"""Assemble a sync-mode collector's recordings/ tree into the canonical rollout
layout (episodes/media/policy_embeddings), in place.

Serves both collectors that write that tree: diffusion_policy's
``scripts/droid_collect_rollouts.py`` and rgrl's
``rgrl.inference.collect_rollouts`` (which generalized beyond DROID and so also
covers robomimic sim sessions). Despite the name, nothing here is
DROID-specific: it never reads ``trajectory.h5``, only episode timestamps, the
online embeddings, the videos and each episode's metadata -- which is exactly
the set a sim session writes too.

Sync-mode collection saves an online per-episode embedding
(``embeddings_<layer>.npz``) alongside each episode, so no offline extraction
checkpoint is needed — this just sorts episodes by timestamp, concatenates
their embeddings, and copies over videos and per-episode metadata (note,
collector args) via ``write_rollout_dir``. For async-streaming collections
with no online embeddings, use ``droid_extract_embeddings.py`` instead.

Every camera the collector recorded is exported by default, one
``media/<camera>/`` directory per view, so the review players can show them side
by side; pass ``--camera`` to narrow that. Views always keep their recorded name,
a one-camera export included — the flat ``media/*.mp4`` form this script used to
write for a single view reads back as an unnamed default view, which costs it
anything keyed by camera (the ``cameras:`` geometry in ``configs/data/``, the
sidebar picker's label). Readers still accept that form, so datasets assembled
before this keep working.

Usage::

    python scripts/utils/assemble_droid_sync_recordings.py \\
        data/rollouts/set_table_baseline_v2_plusmiddle_sync --layer bottleneck_plan_t0

    # one view only (still media/exterior_image_1_left/)
    python scripts/utils/assemble_droid_sync_recordings.py \\
        data/rollouts/set_table_baseline_v2_plusmiddle_sync \\
        --camera exterior_image_1_left
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_graphs.envs.trajectory_dir import write_rollout_dir  # noqa: E402

RECORDINGS_DIR = "recordings"


def _transcode(src: Path, dst: Path, max_width: int, crf: int) -> None:
    """Re-encode *src* into *dst*: capped resolution, H.264/CRF, faststart.

    Raw DROID captures run ~11 Mbps at 720p — far more bitrate than a review
    player needs. This typically cuts file size by ~100x with no visible
    quality loss at review-player sizes, and faststart makes the result
    seekable by browsers/``st.video`` without downloading the whole file.
    """
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale='min({max_width},iw)':-2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(dst),
        ],
        check=True,
    )


def _discover_cameras(ep_dirs: list[Path]) -> list[str]:
    """Camera view names across the collection, from the per-session MP4 names.

    The union rather than the first session's set: a camera that only starts
    recording partway through a collection should still get its own view instead
    of being dropped, and ``write_rollout_dir`` takes clips sparsely.
    """
    return sorted({video.stem for ep_dir in ep_dirs for video in ep_dir.glob("*.mp4")})


def _usable(video: Path, ep_name: str) -> bool:
    """True when *video* is there and non-empty, saying so when it is not.

    A truncated (0-byte) capture counts as absent: ffmpeg would abort on it, and
    ``write_rollout_dir`` takes ``video_paths`` sparsely, so the episode still
    contributes its embeddings without that clip. Named, because otherwise the
    dataset just quietly ends up with fewer clips than episodes.
    """
    if not video.exists():
        print(f"  {ep_name}: no {video.name}; episode keeps no clip for this view")
        return False
    if not video.stat().st_size:
        print(f"  {ep_name}: {video.name} is 0 bytes; episode keeps no clip for this view")
        return False
    return True


def _missing_inputs(ep_dir: Path, layer: str) -> bool:
    """True when *ep_dir* lacks the metadata or embeddings to be assembled.

    An interrupted transfer leaves 0-byte files rather than absent ones, which
    is why size is checked and not just existence.
    """
    for name in ("metadata.yaml", f"embeddings_{layer}.npz"):
        path = ep_dir / name
        if not path.exists() or path.stat().st_size == 0:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rollout_dir", type=Path)
    # Repeatable, like --camera. The collector can capture several activation
    # taps in one pass and rgrl_env.py emits one --layer per entry; as a
    # single-valued option argparse kept only the LAST, so a two-layer
    # collection wrote both embeddings_*.npz in step 1 and assembled one of
    # them, leaving a rollout dir that looks complete and is missing a layer.
    parser.add_argument("--layer", action="append", dest="layers")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Assemble the usable episodes instead of erroring when some are "
        "missing/truncated (default: error, so partial data is never silent)",
    )
    parser.add_argument(
        "--no-transcode",
        action="store_true",
        help="Copy source videos as-is instead of transcoding (default: transcode)",
    )
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        metavar="NAME",
        help="Camera view to export, named after its MP4 (e.g. wrist_image). "
        "Repeat for several; default: every view the collector recorded. Each "
        "lands in media/<camera>/, under the name it was recorded with",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=640,
        help="Cap transcoded video width in pixels, preserving aspect ratio (default: 640)",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=28,
        help="x264 constant-rate-factor; higher = smaller file, lower quality (default: 28)",
    )
    args = parser.parse_args(argv)

    layers = args.layers or ["bottleneck_plan_t0"]
    rollout_dir = args.rollout_dir
    outputs = [rollout_dir / sub for sub in ("episodes", "media", "policy_embeddings")]
    if not args.overwrite:
        for existing in outputs:
            if existing.exists():
                parser.error(f"{existing} already exists; pass --overwrite to replace")

    ep_dirs = sorted((rollout_dir / RECORDINGS_DIR).glob("ep_*"))
    incomplete = sorted(
        {ep for layer in layers for ep in ep_dirs if _missing_inputs(ep, layer)}
    )
    if incomplete and not args.skip_incomplete:
        names = "/".join(f"embeddings_{layer}.npz" for layer in layers)
        parser.error(
            f"{len(incomplete)} of {len(ep_dirs)} episodes are missing or have an "
            f"empty metadata.yaml/{names}, starting with "
            f"{incomplete[0]}. Pass --skip-incomplete to assemble the rest."
        )
    for ep_dir in incomplete:
        print(f"skipping incomplete episode {ep_dir.name}")
    ep_dirs = [ep for ep in ep_dirs if ep not in incomplete]
    if not ep_dirs:
        parser.error(
            f"no usable episodes under {rollout_dir / RECORDINGS_DIR} "
            f"(found {len(incomplete)} incomplete)"
        )
    cameras = args.camera or _discover_cameras(ep_dirs)
    if not cameras:
        print(f"no camera MP4s under {rollout_dir / RECORDINGS_DIR}; assembling without clips")
    else:
        print(f"exporting {len(cameras)} camera view(s): {', '.join(cameras)}")

    lengths, successes, extras, video_paths, embeddings = [], [], [], {}, {}

    with tempfile.TemporaryDirectory(prefix="assemble_droid_") as tmp:
        tmp_dir = Path(tmp)
        for i, ep_dir in enumerate(ep_dirs):
            meta = yaml.safe_load((ep_dir / "metadata.yaml").read_text())
            lengths.append(int(meta["length"]))
            successes.append(meta.get("success"))
            extras.append(
                {k: v for k, v in meta.items() if k not in ("index", "length", "success")}
            )
            clips: dict[str, Path] = {}
            for camera in cameras:
                video = ep_dir / f"{camera}.mp4"
                if not _usable(video, ep_dir.name):
                    continue
                if args.no_transcode:
                    clips[camera] = video
                else:
                    out = tmp_dir / f"ep{i:04d}_{camera}.mp4"
                    _transcode(video, out, args.max_width, args.crf)
                    clips[camera] = out
            if clips:
                # Always under the camera's recorded name, one view or six. The
                # flat media/ form drops it (it reads back as DEFAULT_CAMERA), and
                # everything keyed by camera goes with it — a `--camera
                # wrist_image_left` export used to render upside down because the
                # `cameras:` rotation in configs/data could no longer find it.
                # Readers still accept the flat form, so datasets assembled
                # before this are unaffected.
                video_paths[i] = clips
            for layer in layers:
                with np.load(ep_dir / f"embeddings_{layer}.npz") as data:
                    embeddings.setdefault(layer, []).append(
                        np.asarray(data["embeddings"], dtype=np.float32)
                    )

        stacked = {layer: np.concatenate(parts) for layer, parts in embeddings.items()}

        # Only now is the existing dataset replaceable: everything above can
        # fail (unreadable metadata, a corrupt npz, ffmpeg) and must leave the
        # previous assembly intact. Clearing rather than writing over matters
        # because media/ is keyed by episode index *and* outcome
        # (epNNNN_success.mp4), so assembling a different set of episodes
        # renames clips instead of replacing them, and a stale
        # ep0002_success.mp4 would outlive the ep0002_failure.mp4 that
        # supersedes it.
        for existing in outputs:
            if existing.exists():
                shutil.rmtree(existing)

        reader = write_rollout_dir(
            rollout_dir,
            lengths,
            successes,
            stacked,
            video_paths=video_paths,
            source={"kind": "droid_rollouts", "raw": str(rollout_dir)},
            episode_extras=extras,
        )

    episodes = reader.episodes()
    if len(video_paths) != len(episodes):
        print(
            f"WARNING: {len(episodes) - len(video_paths)} of {len(episodes)} episodes "
            "have no review clip (see the per-episode notes above)"
        )
    n_succ = sum(1 for ep in episodes if ep.success is True)
    n_fail = sum(1 for ep in episodes if ep.success is False)
    print(
        f"{rollout_dir}: {len(episodes)} episodes — {n_succ} success, {n_fail} failure, "
        f"{len(episodes) - n_succ - n_fail} unlabeled; layers {', '.join(layers)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
