"""Convert raw DROID session folders to a robo_dataset store for training.

Raw DROID sessions contain trajectory.h5 + recordings/ and are not directly
usable as a high-throughput training dataset.  This script converts them to
any robo_dataset writable backend — defaulting to VideoStore.

Camera serials are discovered automatically from the first session unless
overridden with --camera.  By default all camera views (wrist + exterior) are
included, following the DROID RLDS naming convention.

Output formats
--------------
video         (default) Per-episode MP4s + zarr/lz4 for lowdim.
              Load with: robo_dataset.VideoStore(path)
zarr          zarr DirectoryStore with JPEG images + lz4 lowdim.
              Load with: robo_dataset.ZarrStore(path)
zarr_sharded  zarr v3 ShardingCodec store (requires zarr >= 3.0).
              Load with: robo_dataset.ZarrShardedStore(path)
rlds          TFRecord per-frame store (requires tfrecord + crc32c).
              Load with: robo_dataset.RLDSFrameStore(path)

Usage — all cameras, VideoStore (default)::

    python scripts/utils/convert_droid_data.py \\
        --data-dir /data/droid/sessions \\
        --output /data/droid/kendama.video

Usage — lowdim only, no images::

    python scripts/utils/convert_droid_data.py \\
        --data-dir /data/droid/sessions \\
        --output /data/droid/kendama.video \\
        --camera-source none

Usage — RLDS format with manually specified cameras::

    python scripts/utils/convert_droid_data.py \\
        --data-dir /data/droid/sessions \\
        --output /data/droid/kendama.rlds \\
        --format rlds \\
        --camera exterior_image_1_left=14313307 \\
        --camera wrist_image_left=14324001
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "third_party" / "robo_dataset"))

log = logging.getLogger(__name__)

_CAMERA_TYPE_TO_STRING = {0: "wrist", 1: "exterior", 2: "exterior"}


def _discover_cameras(data_dir: Path, camera_source: str) -> dict[str, str]:
    """Auto-discover camera serial → obs_key mapping from the first session.

    Reads camera types from trajectory.h5 and names cameras following the
    DROID RLDS convention (wrist_image_left, exterior_image_N_left).
    """
    import h5py

    session: Path | None = None
    for root, _dirs, files in sorted(os.walk(data_dir, followlinks=True)):
        if "trajectory.h5" in files:
            session = Path(root)
            break
    if session is None:
        raise ValueError(f"No trajectory.h5 sessions found under {data_dir}")

    # Collect serials from the recording directory
    if camera_source == "svo":
        rec_dir = session / "recordings" / "SVO"
        if not rec_dir.is_dir():
            raise FileNotFoundError(f"No recordings/SVO dir in {session}")
        seen: set[str] = set()
        serials: list[str] = []
        for f in sorted(rec_dir.iterdir()):
            if ".svo" in f.name:
                serial = f.name.split(".svo")[0]
                if serial not in seen:
                    seen.add(serial)
                    serials.append(serial)
    elif camera_source == "mp4":
        rec_dir = session / "recordings" / "MP4"
        if not rec_dir.is_dir():
            raise FileNotFoundError(f"No recordings/MP4 dir in {session}")
        serials = [f.stem for f in sorted(rec_dir.iterdir()) if f.suffix == ".mp4"]
    else:
        return {}

    if not serials:
        raise ValueError(f"No camera recordings found under {session / 'recordings'}")

    # Read camera types from trajectory.h5 to assign RLDS-style obs_key names
    with h5py.File(session / "trajectory.h5", "r") as f:
        cam_type_grp = f.get("observation/camera_type")
        if cam_type_grp is not None:
            cam_types = {k: int(cam_type_grp[k][0]) for k in cam_type_grp}
        else:
            cam_types = {}

    cameras: dict[str, str] = {}
    exterior_idx = 0
    for serial in serials:
        ctype = cam_types.get(serial, 1)  # unknown → treat as exterior
        if ctype == 0:
            cameras["wrist_image_left"] = serial
        else:
            exterior_idx += 1
            cameras[f"exterior_image_{exterior_idx}_left"] = serial

    return cameras


_FORMAT_WRITERS = {
    "video": ("robo_dataset.backends.video", "VideoStoreWriter"),
    "zarr": ("robo_dataset.backends.zarr_backend", "ZarrStoreWriter"),
    "zarr_sharded": ("robo_dataset.backends.zarr_sharded", "ZarrShardedStoreWriter"),
    "rlds": ("robo_dataset.backends.rlds", "RLDSFrameStoreWriter"),
}


def _build_writer(fmt: str, path: str, src, image_keys: list[str], args: argparse.Namespace):
    """Instantiate the correct WritableTrajectoryStore for the chosen format."""
    import importlib

    mod_name, cls_name = _FORMAT_WRITERS[fmt]
    cls = getattr(importlib.import_module(mod_name), cls_name)

    key_shapes = {k: src.frame_shape(k) for k in src.keys}
    key_dtypes = {k: src.frame_dtype(k) for k in src.keys}
    common = dict(
        path=path,
        key_shapes=key_shapes,
        key_dtypes=key_dtypes,
        image_keys=image_keys,
        schemas=src.schemas,
    )

    if fmt == "video":
        return cls.create(**common, fps=args.fps, crf=args.crf)
    if fmt == "zarr_sharded":
        return cls.create(**common, chunk_frames=args.chunk_frames, shard_frames=args.shard_frames)
    # zarr and rlds accept the common kwargs without extras
    return cls.create(**common)


def _merge_shards(shard_paths, output):
    """Concatenate per-episode VideoStore shards into one contiguous store.

    The episode-indexed VideoStore layout is self-contained per episode
    (videos/<key>/episode_<idx>.mp4 plus data/<idx>/<key> zarr dirs) and carries
    no episode_ends.npy — episode lengths are recovered at read time. Merging is
    therefore a renumber-and-copy of each shard's episodes into one index space;
    per-shard corrupt-skips already produced gap-free local indices, so global
    indices stay gap-free too. tasks.json is remapped with the same counter.
    """
    import json
    import shutil

    output = Path(output)
    with open(shard_paths[0] / "meta.json") as f:
        meta = json.load(f)

    output.mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(exist_ok=True)
    for vk in meta["video_keys"]:
        (output / "videos" / vk).mkdir(parents=True, exist_ok=True)

    global_ep = 0
    merged_tasks = {}
    for shard_path in shard_paths:
        with open(shard_path / "meta.json") as f:
            n_shard_eps = int(json.load(f)["n_episodes"])
        shard_tasks_path = shard_path / "tasks.json"
        shard_tasks = {}
        if shard_tasks_path.exists():
            with open(shard_tasks_path) as f:
                shard_tasks = json.load(f)

        for local_ep in range(n_shard_eps):
            for vk in meta["video_keys"]:
                shutil.copy2(
                    shard_path / "videos" / vk / f"episode_{local_ep:06d}.mp4",
                    output / "videos" / vk / f"episode_{global_ep:06d}.mp4",
                )
            src_data = shard_path / "data" / f"{local_ep:06d}"
            if src_data.is_dir():
                shutil.copytree(src_data, output / "data" / f"{global_ep:06d}")
            if str(local_ep) in shard_tasks:
                merged_tasks[str(global_ep)] = shard_tasks[str(local_ep)]
            global_ep += 1

    if merged_tasks:
        with open(output / "tasks.json", "w") as f:
            json.dump(merged_tasks, f, indent=2)

    meta["n_episodes"] = global_ep
    with open(output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert raw DROID sessions to a robo_dataset store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Root of the raw DROID session tree (directories containing trajectory.h5).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output store directory.",
    )
    parser.add_argument(
        "--format",
        choices=list(_FORMAT_WRITERS),
        default="video",
        help="Output store format (default: video).",
    )
    parser.add_argument(
        "--camera-source",
        choices=("none", "svo", "mp4"),
        default="svo",
        help="Camera decode source: 'none' for lowdim only, 'svo' for ZED SVO files, "
        "'mp4' for recorded MP4s via the droid package (default: svo).",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="OBS_KEY=ZED_SERIAL",
        help="Explicit camera mapping, e.g. exterior_image_1_left=14313307. "
        "Repeat for multiple cameras. If omitted, all cameras are discovered "
        "automatically from the first session.",
    )
    parser.add_argument(
        "--action-space",
        default="joint_velocity",
        help="DROID action space key (default: joint_velocity).",
    )
    parser.add_argument(
        "--image-size",
        default="180x320",
        metavar="HxW",
        help="Output image resolution as HxW (default: 180x320).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Video frame rate for encoded MP4s — video format only (default: 15).",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="libx264 constant rate factor — video format only; lower = higher quality "
        "(default: 23).",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=8,
        metavar="N",
        help="Inner decompression chunk size — zarr_sharded format only (default: 8).",
    )
    parser.add_argument(
        "--shard-frames",
        type=int,
        default=1024,
        metavar="N",
        help="Outer shard file size — zarr_sharded format only (default: 1024).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N episodes (useful for testing).",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        default=False,
        help="Silently skip sessions with zero movement-enabled steps instead of raising.",
    )
    parser.add_argument(
        "--skip-corrupt",
        action="store_true",
        default=False,
        help="Silently skip episodes where camera decode fails (e.g. SVO frame-count mismatch).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker subprocesses (default: 1).",
    )
    parser.add_argument(
        "--session-offset",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N valid sessions (used internally by worker subprocesses).",
    )
    parser.add_argument(
        "--zed-settings-path",
        default=None,
        help="Directory containing SN<serial>.conf ZED calibration files "
        "(required for --camera-source svo on most systems).",
    )
    parser.add_argument(
        "--droid-package-path",
        default="~/src_droid",
        help="Path to the droid package for --camera-source mp4 when not installed.",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="Comma-separated list of session date dirs to include, e.g. "
        "2026-06-24,2026-06-26 (matches the .../<split>/<date>/<timestamp>/ "
        "component). Sessions on other dates are ignored.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    dates = {d.strip() for d in args.dates.split(",") if d.strip()} if args.dates else None

    if args.workers > 1:
        n_raw = sum(
            1
            for root, _dirs, files in os.walk(args.data_dir, followlinks=True)
            if "trajectory.h5" in files
            and (dates is None or os.path.basename(os.path.dirname(root)) in dates)
        )
        log.info("found %d raw sessions; launching %d workers", n_raw, args.workers)

        chunk = math.ceil(n_raw / args.workers)
        shard_dir = args.output.parent / (args.output.name + "_shards")
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_paths = []
        procs = []
        worker_logs: list[Path] = []
        log_handles = []
        try:
            for i in range(args.workers):
                offset = i * chunk
                size = min(chunk, n_raw - offset)
                if size <= 0:
                    break
                shard_out = shard_dir / f"shard_{i:02d}"
                shard_paths.append(shard_out)
                worker_argv = [
                    sys.executable,
                    __file__,
                    "--data-dir",
                    str(args.data_dir),
                    "--output",
                    str(shard_out),
                    "--format",
                    args.format,
                    "--camera-source",
                    args.camera_source,
                    "--action-space",
                    args.action_space,
                    "--image-size",
                    args.image_size,
                    "--fps",
                    str(args.fps),
                    "--crf",
                    str(args.crf),
                    "--chunk-frames",
                    str(args.chunk_frames),
                    "--shard-frames",
                    str(args.shard_frames),
                    "--workers",
                    "1",
                    "--session-offset",
                    str(args.session_offset + offset),
                    "--max-episodes",
                    str(size),
                ]
                if args.skip_empty:
                    worker_argv.append("--skip-empty")
                if args.skip_corrupt:
                    worker_argv.append("--skip-corrupt")
                for cam_spec in args.camera:
                    worker_argv += ["--camera", cam_spec]
                if args.zed_settings_path is not None:
                    worker_argv += ["--zed-settings-path", args.zed_settings_path]
                if args.droid_package_path != "~/src_droid":
                    worker_argv += ["--droid-package-path", args.droid_package_path]
                if args.dates is not None:
                    worker_argv += ["--dates", args.dates]
                # stderr to a file per worker, not a pipe: nothing reads these
                # pipes until every worker has exited, so a worker that writes
                # more than the ~64 KB pipe buffer blocks forever and the wait()
                # below never returns. A conversion that hangs after an hour with
                # no output is the worst possible way to learn that.
                log_path = shard_dir / f"shard_{i:02d}.stderr.log"
                handle = log_path.open("wb")
                log_handles.append(handle)  # before the spawn, so cleanup sees it
                procs.append(subprocess.Popen(worker_argv, stderr=handle))
                worker_logs.append(log_path)  # after it, so this stays 1:1 with procs

            for proc in procs:
                proc.wait()
        finally:
            # Reached however the block above ends, including a Popen that fails
            # partway through the loop: workers already running would otherwise
            # carry on unsupervised while their log files stayed open.
            for proc in procs:
                if proc.poll() is None:
                    proc.kill()
            for handle in log_handles:
                handle.close()

        for i, (proc, log_path) in enumerate(zip(procs, worker_logs, strict=True)):
            if proc.returncode != 0:
                stderr = log_path.read_text(errors="replace")
                raise RuntimeError(f"Worker {i} exited with {proc.returncode}:\n{stderr}")

        log.info("merging %d shards -> %s", len(shard_paths), args.output)
        _merge_shards(shard_paths, args.output)
        import shutil as _shutil

        _shutil.rmtree(shard_dir)
        return 0

    # Parse explicit camera overrides
    cameras: dict[str, str] = {}
    for spec in args.camera:
        obs_key, _, serial = spec.partition("=")
        if not serial:
            parser.error(f"--camera expects OBS_KEY=ZED_SERIAL, got {spec!r}")
        cameras[obs_key] = serial

    # Auto-discover cameras when not explicitly provided
    if args.camera_source != "none" and not cameras:
        log.info("discovering cameras from first session in %s ...", args.data_dir)
        cameras = _discover_cameras(args.data_dir, args.camera_source)
        if cameras:
            log.info("discovered cameras: %s", cameras)
        else:
            log.warning("no cameras found; falling back to lowdim only")

    # Parse image size
    try:
        h_str, w_str = args.image_size.lower().split("x")
        image_size = (int(h_str), int(w_str))
    except ValueError:
        parser.error(f"--image-size must be HxW, e.g. 180x320, got {args.image_size!r}")

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output {args.output} already exists and is not empty. Remove it first.")

    from robo_dataset.backends.droid import DroidStore

    log.info(
        "opening DroidStore at %s (camera_source=%s, cameras=%s)",
        args.data_dir,
        args.camera_source,
        list(cameras) or "none",
    )
    src = DroidStore(
        path=str(args.data_dir),
        cameras=cameras or None,
        camera_source=args.camera_source,
        image_size=image_size,
        action_space=args.action_space,
        max_episodes=args.max_episodes,
        session_offset=args.session_offset,
        droid_package_path=args.droid_package_path,
        zed_settings_path=args.zed_settings_path,
        skip_empty=args.skip_empty,
        skip_corrupt=args.skip_corrupt,
        dates=dates,
    )
    log.info(
        "%d episodes, %d frames, keys: %s",
        src.n_episodes,
        src.n_frames,
        src.keys,
    )

    image_keys = list(cameras.keys())
    dst = _build_writer(args.format, str(args.output), src, image_keys, args)
    log.info("converting to %s -> %s", args.format, args.output)
    # Episode-wise conversion: iter_episodes() decodes one episode at a time and
    # (with --skip-corrupt) drops any that fail to decode rather than aborting
    # the whole run. Survivors are written contiguously via sequential
    # put_episode, so output episode indices have no gaps and _merge_shards can
    # re-index shards cleanly. src._yielded_sessions records the session for each
    # written episode, in order, for the tasks.json sidecar below.
    from tqdm import tqdm as _tqdm

    for _ep in _tqdm(src.iter_episodes(), total=src.n_episodes, desc="Converting"):
        dst.put_episode(_ep)
    dst.finalize()

    import json as _json

    import h5py as _h5

    _tasks = {}
    for _i, _sess in enumerate(src._yielded_sessions):
        with _h5.File(os.path.join(_sess, "trajectory.h5"), "r") as _f:
            _tasks[str(_i)] = str(_f.attrs.get("current_task", ""))
    with open(args.output / "tasks.json", "w") as _fp:
        _json.dump(_tasks, _fp, indent=2)

    log.info(
        "done: %d episodes, %d frames -> %s",
        src.n_episodes,
        src.n_frames,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
