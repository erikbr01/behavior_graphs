#!/usr/bin/env python
"""Pre-build web-sized video proxies + poster frames for rollout directories.

The apps transcode on first view (~0.5 s per clip), which is fine
interactively but shows up as a spinner the first time anyone opens a node.
Running this once per dataset makes every clip instant from the first click.

Usage:
    python scripts/prewarm_media.py /path/to/rollouts/set_table_baseline_v2_sync
    python scripts/prewarm_media.py /path/to/rollouts --all
    python scripts/prewarm_media.py /path/to/rollouts --all --workers 16

Honours the same environment variables as the apps:
    BEHAVIOR_GRAPHS_MEDIA_CACHE, BEHAVIOR_GRAPHS_PROXY_HEIGHT,
    BEHAVIOR_GRAPHS_PROXY_CRF, BEHAVIOR_GRAPHS_DISPLAY_DATA.

The last one because a camera's configured rotation (``cameras:`` in
``configs/data/<group>.yaml``) is baked into its proxy and keys the cache entry:
a clip under ``media/<camera>/`` is warmed in the orientation the app will
request, which is why the group the app reads matters here too.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from behavior_graphs.envs.trajectory_dir import DEFAULT_CAMERA, MEDIA_DIR  # noqa: E402
from behavior_graphs.viz import media  # noqa: E402


def _with_transforms(videos: list[Path]) -> list[tuple[Path, dict | None]]:
    """Pair each clip with the transform its camera is configured with.

    The camera is the ``media/<camera>/`` directory a clip sits in — or
    :data:`DEFAULT_CAMERA` for the flat ``media/`` form, which is how the app
    reads those back, so a group's ``cameras: {default: …}`` reaches them here
    too. The dataset is the rollout dir above ``media/``, the same key
    ``configs/data/<group>.yaml`` uses; the lookup is memoised because it walks
    that group's whole ``defaults:`` tree and a dataset has one answer for all
    of its clips. A clip that is not under ``media/`` at all (the ``--all``
    rescue path, raw ``recordings/``) has no camera to look up and is warmed as
    recorded.
    """
    from behavior_graphs.viz.streamlit.data import camera_transforms

    by_dataset: dict[str, dict[str, dict]] = {}
    out: list[tuple[Path, dict | None]] = []
    for v in videos:
        if v.parent.name == MEDIA_DIR:
            camera, dataset = DEFAULT_CAMERA, v.parent.parent.name
        elif v.parent.parent.name == MEDIA_DIR:
            camera, dataset = v.parent.name, v.parent.parent.parent.name
        else:
            out.append((v, None))
            continue
        if dataset not in by_dataset:
            by_dataset[dataset] = camera_transforms(dataset)
        out.append((v, by_dataset[dataset].get(camera) or None))
    return out


def _discover(root: Path, every_dataset: bool) -> list[Path]:
    """The clips worth warming under *root*: what the apps serve, and only that.

    ``media/**`` at either depth, so a multi-camera dataset's ``media/<camera>/``
    views are covered — and, with ``--all``, so are the ``media/`` dirs of every
    dataset below *root* without the raw ``recordings/`` captures beside them.
    Those are what the collector wrote, not what anything reads: warming all 2594
    of them spent most of a run's ffmpeg time on 30 MB sources no page will ever
    ask for, and left the cache holding entries nothing can reach.

    The bare ``**/*.mp4`` rescue stays for a layout with no ``media/`` at all,
    where anything else would warm nothing.
    """
    pattern = f"**/{MEDIA_DIR}/**/*.mp4" if every_dataset else f"{MEDIA_DIR}/**/*.mp4"
    return sorted(root.glob(pattern)) or sorted(root.glob("**/*.mp4"))


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", type=Path, help="Rollout dir, or a parent of several with --all")
    ap.add_argument("--all", action="store_true", help="Recurse into every subdirectory of ROOT")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel ffmpeg processes (default: min(8, cpu_count))",
    )
    ap.add_argument("--no-thumbs", action="store_true", help="Skip poster frames")
    ap.add_argument(
        "--max-cache-gb",
        type=float,
        default=0.0,
        help=(
            "After pre-warming, trim the media cache to this size, "
            "least-recently-used first. The cache is content-addressed and "
            "append-only, so nothing else reclaims superseded entries."
        ),
    )
    ap.add_argument(
        "--strip-thumbs",
        action="store_true",
        help="Also build the small tiles used by the episode strip",
    )
    args = ap.parse_args()

    if media.ffmpeg_exe() is None:
        print("error: ffmpeg not found on PATH (and imageio-ffmpeg unavailable)", file=sys.stderr)
        return 2

    videos = _discover(args.root, args.all)
    if not videos:
        print(f"No .mp4 files under {args.root}", file=sys.stderr)
        return 1

    src_bytes = sum(v.stat().st_size for v in videos)
    widths: list[int] = []
    if not args.no_thumbs:
        widths.append(media.POSTER_WIDTH)
    if args.strip_thumbs:
        widths.append(media.STRIP_THUMB_WIDTH)

    print(f"{len(videos)} videos ({_fmt_bytes(src_bytes)}) → {media.cache_dir()}")
    print(f"proxy: {media.proxy_height()}p, CRF {media.proxy_crf()}")

    # A configured rotation is part of the cache key, so a clip has to be warmed
    # in the orientation the app will ask for — otherwise every rotated view
    # transcodes on first click, which is what this script exists to prevent.
    items = _with_transforms(videos)
    xformed = [spec for _, spec in items if spec]
    if xformed:
        specs = ", ".join(sorted({str(dict(spec)) for spec in xformed}))
        print(f"  {len(xformed)} of them rotated/mirrored: {specs}")

    done = 0
    done_lock = threading.Lock()
    t0 = time.perf_counter()

    def _tick(_path: Path) -> None:
        # Called from prewarm's ffmpeg worker threads: `done += 1` is a
        # read-modify-write, so without the lock increments are lost and the
        # progress line freezes short of the total on a finished run.
        nonlocal done
        with done_lock:
            done += 1
            n = done
        if n % 10 == 0 or n == len(videos):
            elapsed = time.perf_counter() - t0
            rate = n / elapsed if elapsed else 0
            print(f"  {n}/{len(videos)}  ({rate:.1f}/s)", end="\r", flush=True)

    n_ok, n_total = media.prewarm(
        items,
        thumb_widths=tuple(widths),
        workers=args.workers,
        on_done=_tick,
    )
    print()

    proxy_bytes = 0
    for v, spec in items:
        proxy = media.proxy_path(v, transform=spec)
        if proxy.exists():
            proxy_bytes += proxy.stat().st_size
    elapsed = time.perf_counter() - t0
    print(f"done: {n_ok}/{n_total} in {elapsed:.1f}s")
    if proxy_bytes:
        print(
            f"proxies: {_fmt_bytes(proxy_bytes)} "
            f"({src_bytes / max(proxy_bytes, 1):.0f}× smaller than source)"
        )
    if args.max_cache_gb > 0:
        n_pruned, freed = media.prune(int(args.max_cache_gb * 1024**3))
        if n_pruned:
            print(f"pruned: {n_pruned} stale files ({_fmt_bytes(freed)} freed)")

    n_files, total = media.cache_stats()
    print(f"cache now holds {n_files} files ({_fmt_bytes(total)})")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
