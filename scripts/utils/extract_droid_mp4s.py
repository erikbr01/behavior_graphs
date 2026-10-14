#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["h5py", "mcap"]
# ///
"""Export human-viewable MP4 clips from raw DROID sessions, without the ZED SDK.

Companion to ``convert_droid_data.py``, with a different purpose: that script
builds a training-ready VideoStore and needs ``pyzed`` + the ZED SDK + per-camera
``SN<serial>.conf`` calibration files (see docs/droid_workflow.md).  This one just
pulls a few demonstrations per language instruction out to MP4 so you can *watch*
them — picking user-study stimuli, sanity-checking a collection session, sharing
examples — and runs anywhere ffmpeg is on PATH.

How it avoids pyzed
-------------------
A ``.svo2`` file is an **MCAP** container (the open Foxglove/ROS 2 log format; the
first bytes are ``\\x89MCAP0\\r\\n`` followed by ``libmcap stereolabs``).  Video
lives on the ``Camera_SN<serial>/side_by_side`` channel, one message per frame::

    [4B uint32 total_size][4B uint32 payload_size][payload: HEVC Annex-B][56B trailer]

Concatenating just the payloads yields a plain HEVC elementary stream that ffmpeg
decodes directly.  Both size fields are relative to the whole message
(``total - payload_size == 64`` on every frame observed, i.e. the 8-byte header
plus a fixed 56-byte per-frame trailer); leaving the trailer in still decodes but
makes ffmpeg log ``Failed to parse header of NALU (type 0)`` for every frame.

Frames are a 2560x720 stereo pair, so the left eye alone is 1280x720; ``--stereo``
keeps both.  Frame rate is derived from MCAP ``log_time`` (~14.36 Hz on the
workstation recordings, i.e. nominal 15 Hz control minus dropped frames).

Verified against SVO format 2.0.5 / ZED SDK 5.2.3 recordings.  The older ``.svo``
files in the public DROID release come from ZED SDK 3.x and are *not* MCAP, so
they will fail in ``make_reader`` — use ``convert_droid_data.py`` for those.

Camera roles come from ``observation/camera_type`` in ``trajectory.h5``
(0 = wrist, 1 = exterior, 2 = fixed), matching the workstation serials:
``14313307`` wrist, ``36716034`` exterior.

Usage — 5 successful episodes per instruction variant, exterior camera only::

    scripts/utils/extract_droid_mp4s.py --per-variant 5 --cameras exterior

Usage — 5 per *pattern*, spread across that pattern's variants, both cameras::

    scripts/utils/extract_droid_mp4s.py --per-pattern 5

Usage — a different task family, to a scratch dir, preview only::

    scripts/utils/extract_droid_mp4s.py --patterns 'kendama - *' --per-variant 3 \\
        --out /tmp/kendama --dry-run

The shebang runs it under ``uv`` with h5py + mcap resolved on the fly, so no repo
env is needed.  To use the repo venv instead: ``uv sync --extra droid`` and then
``python scripts/utils/extract_droid_mp4s.py``.

Writes ``<out>/<task_slug>/<episode>__<camera>_<serial>.mp4`` plus a
``manifest.json`` recording the instruction, outcome, camera and frame count of
every clip.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
from mcap.reader import make_reader

REPO = Path(__file__).resolve().parents[2]

DEFAULT_ROOT = Path("/Volumes/ebauerdrive/stanford/droid_data/data")
DEFAULT_OUT = REPO / "data" / "droid_data_collection_examples"
DEFAULT_PATTERNS = ["prepare plate - *", "prepare tray - *", "cooking - *"]

HEADER_BYTES = 8  # total_size, payload_size
TRAILER_BYTES = 56  # fixed ZED per-frame trailer, not part of the bitstream
CAMERA_TYPES = {0: "wrist", 1: "exterior", 2: "fixed"}


# ---------------------------------------------------------------- episode scanning


@dataclass
class Episode:
    path: Path
    task: str
    success: bool
    date: str
    user: str
    scene_id: int
    svos: list[Path] = field(default_factory=list)
    cam_types: dict[str, str] = field(default_factory=dict)

    @property
    def episode_id(self) -> str:
        return self.path.name


def read_episode(ep_dir: Path) -> Episode | None:
    """Read instruction + camera metadata from trajectory.h5. None if unreadable."""
    h5_path = ep_dir / "trajectory.h5"
    try:
        with h5py.File(h5_path, "r") as f:
            attrs = f.attrs
            cam_types = {}
            if "observation" in f and "camera_type" in f["observation"]:
                for serial, ds in f["observation"]["camera_type"].items():
                    try:
                        code = int(ds[0]) if ds.shape else int(ds[()])
                    except Exception:
                        continue
                    cam_types[serial] = CAMERA_TYPES.get(code, f"type{code}")
            return Episode(
                path=ep_dir,
                task=str(attrs.get("current_task", "")).strip(),
                success=bool(attrs.get("success", False)),
                date=ep_dir.parent.name,
                user=str(attrs.get("user", "")),
                scene_id=int(attrs.get("scene_id", -1)),
                svos=sorted((ep_dir / "recordings" / "SVO").glob("*.svo2")),
                cam_types=cam_types,
            )
    except Exception as exc:  # 0-byte / truncated files exist in the raw dataset
        print(f"  ! skipping unreadable {h5_path}: {exc}", file=sys.stderr)
        return None


def scan_dataset(root: Path, index_path: Path | None) -> list[Episode]:
    """Scan <root>/{success,failure}/<date>/<episode>/. Cached in a JSON index."""
    if index_path and index_path.exists():
        raw = json.loads(index_path.read_text())
        print(f"Loaded index of {len(raw)} episodes from {index_path}")
        return [
            Episode(path=Path(d.pop("path")), svos=[Path(s) for s in d.pop("svos")], **d)
            for d in raw
        ]

    ep_dirs = sorted(p.parent for p in root.glob("*/*/*/trajectory.h5"))
    print(f"Scanning {len(ep_dirs)} episode directories under {root} ...")
    episodes = [ep for ep in (read_episode(d) for d in ep_dirs) if ep is not None]
    print(f"Read metadata for {len(episodes)} episodes.")

    if index_path:
        index_path.write_text(
            json.dumps(
                [
                    {**e.__dict__, "path": str(e.path), "svos": [str(s) for s in e.svos]}
                    for e in episodes
                ],
                indent=1,
            )
        )
        print(f"Wrote index to {index_path}")
    return episodes


# ---------------------------------------------------------------- selection


def select(
    episodes: list[Episode],
    patterns: list[str],
    n: int,
    per_variant: bool,
    require_success: bool = True,
) -> dict[str, list[Episode]]:
    """Pick episodes per pattern.

    per_variant=False: n episodes per pattern, round-robin over the matching
        sub-instructions, so 'prepare plate - *' yields a spread over milk
        left/middle/right rather than n takes of whichever variant sorts first.
    per_variant=True: n episodes for *each* distinct instruction the pattern matches.

    Selection within a variant is the first n by sorted path, i.e. deterministic
    across reruns.
    """
    chosen: dict[str, list[Episode]] = {}
    for pattern in patterns:
        matches = [
            e
            for e in episodes
            if fnmatch.fnmatch(e.task, pattern) and e.svos and (e.success or not require_success)
        ]
        by_task: dict[str, list[Episode]] = {}
        for e in matches:
            by_task.setdefault(e.task, []).append(e)
        for lst in by_task.values():
            lst.sort(key=lambda e: str(e.path))

        picked: list[Episode] = []
        if per_variant:
            for task in sorted(by_task):
                take = by_task[task][:n]
                picked.extend(take)
                if len(take) < n:
                    print(f"  ! only {len(take)} of {n} available for {task!r}")
        else:
            round_idx = 0
            while len(picked) < n and any(len(v) > round_idx for v in by_task.values()):
                for task in sorted(by_task):
                    if len(picked) >= n:
                        break
                    if len(by_task[task]) > round_idx:
                        picked.append(by_task[task][round_idx])
                round_idx += 1
            if len(picked) < n:
                print(f"  ! only {len(picked)} of {n} available for {pattern!r}")

        chosen[pattern] = picked
        variants = ", ".join(sorted({e.task for e in picked}))
        print(f"{pattern!r}: {len(matches)} candidates -> picked {len(picked)} [{variants}]")
    return chosen


# ---------------------------------------------------------------- svo2 -> mp4


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "unnamed"


def svo_frames(svo: Path):
    """Yield (log_time_ns, hevc_payload) for the side_by_side channel."""
    with svo.open("rb") as fh:
        reader = make_reader(fh)
        topic = None
        for channel in reader.get_summary().channels.values():
            if channel.topic.endswith("/side_by_side"):
                topic = channel.topic
                break
        if topic is None:
            raise RuntimeError(f"no side_by_side channel in {svo}")
        for _schema, _channel, msg in reader.iter_messages(topics=[topic]):
            size = int.from_bytes(msg.data[4:8], "little")
            yield msg.log_time, msg.data[HEADER_BYTES : HEADER_BYTES + size]


def convert(
    svo: Path,
    out_mp4: Path,
    crf: int,
    stereo: bool,
    fps_override: float | None,
    max_frames: int | None,
) -> dict:
    """Stream one .svo2 through ffmpeg into an mp4. Returns manifest info."""
    payloads, times = [], []
    for log_time, payload in svo_frames(svo):
        times.append(log_time)
        payloads.append(payload)
        if max_frames and len(payloads) >= max_frames:
            break
    if not payloads:
        raise RuntimeError(f"no frames in {svo}")

    span = (times[-1] - times[0]) / 1e9
    fps = fps_override or (round((len(times) - 1) / span, 3) if span > 0 else 15.0)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "hevc",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
    ]
    if not stereo:  # left eye only
        cmd += ["-filter:v", "crop=iw/2:ih:0:0"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out_mp4),
    ]

    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            for payload in payloads:
                proc.stdin.write(payload)
        finally:
            proc.stdin.close()
        if proc.wait() != 0:
            err.seek(0)
            raise RuntimeError(f"ffmpeg failed for {svo}:\n{err.read().decode(errors='replace')}")

    return {
        "mp4": str(out_mp4),
        "svo2": str(svo),
        "frames": len(payloads),
        "fps": fps,
        "duration_s": round(len(payloads) / fps, 2),
        "bytes": out_mp4.stat().st_size,
    }


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ROOT,
        help="raw DROID root holding {success,failure}/<date>/<episode>/",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--patterns",
        nargs="+",
        default=DEFAULT_PATTERNS,
        help="glob patterns matched against the h5 'current_task' attribute",
    )
    ap.add_argument(
        "--per-pattern",
        type=int,
        default=5,
        help="episodes per pattern, spread across its variants",
    )
    ap.add_argument(
        "--per-variant",
        type=int,
        default=None,
        help="episodes for EACH distinct instruction a pattern matches (overrides --per-pattern)",
    )
    ap.add_argument("--cameras", choices=["all", "wrist", "exterior", "fixed"], default="all")
    ap.add_argument("--include-failures", action="store_true")
    ap.add_argument("--stereo", action="store_true", help="keep the full side-by-side pair")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument(
        "--fps", type=float, default=None, help="override the fps derived from timestamps"
    )
    ap.add_argument(
        "--max-frames", type=int, default=None, help="truncate each clip (for quick tests)"
    )
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--index",
        type=Path,
        default=None,
        help="cache/reuse the episode metadata scan at this JSON path",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    episodes = scan_dataset(args.data_root, args.index)
    chosen = select(
        episodes,
        args.patterns,
        args.per_variant or args.per_pattern,
        per_variant=args.per_variant is not None,
        require_success=not args.include_failures,
    )

    jobs = []
    for pattern, eps in chosen.items():
        for ep in eps:
            for svo in ep.svos:
                serial = svo.stem
                cam = ep.cam_types.get(serial, "cam")
                if args.cameras != "all" and cam != args.cameras:
                    continue
                out_mp4 = args.out / slugify(ep.task) / f"{ep.episode_id}__{cam}_{serial}.mp4"
                jobs.append((pattern, ep, cam, serial, svo, out_mp4))

    print(f"\n{len(jobs)} clip(s) to write into {args.out}")
    for _pattern, ep, cam, _serial, _svo, out_mp4 in jobs:
        print(f"  {ep.task:<32} {ep.date}/{ep.episode_id} [{cam}] -> {out_mp4}")
    if args.dry_run:
        return 0

    manifest, failures = [], []

    def run(job):
        pattern, ep, cam, serial, svo, out_mp4 = job
        info = convert(svo, out_mp4, args.crf, args.stereo, args.fps, args.max_frames)
        return {
            "pattern": pattern,
            "task": ep.task,
            "episode_id": ep.episode_id,
            "date": ep.date,
            "outcome": "success" if ep.success else "failure",
            "user": ep.user,
            "scene_id": ep.scene_id,
            "camera": cam,
            "serial": serial,
            "stereo": args.stereo,
            **info,
        }

    print()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for job, result in zip(jobs, pool.map(lambda j: _safe(run, j), jobs)):
            if isinstance(result, Exception):
                failures.append((str(job[5]), repr(result)))
                print(f"FAIL {job[5]}: {result}", file=sys.stderr)
            else:
                manifest.append(result)
                print(
                    f"ok   {result['mp4']}  ({result['frames']} frames, "
                    f"{result['duration_s']}s @ {result['fps']}fps, "
                    f"{result['bytes'] / 1e6:.1f} MB)"
                )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} mp4(s) + manifest.json to {args.out}")
    if failures:
        print(f"{len(failures)} failure(s):", file=sys.stderr)
        for path, err in failures:
            print(f"  {path}: {err}", file=sys.stderr)
    return 1 if failures else 0


def _safe(fn, job):
    try:
        return fn(job)
    except Exception as exc:
        return exc


if __name__ == "__main__":
    sys.exit(main())
