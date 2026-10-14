"""Analyze and trim idle (zero-velocity) beginnings from DROID episodes.

Hypothesis: at the start of many episodes the arm is stationary because
in the raw data the operator had not yet enabled movement, and the model
lacks enough history to decide when to start moving.

This script:
  1. Loads a converted DROID VideoStore.
  2. Reads per-episode joint-velocity action data and computes per-frame
     velocity norms to identify idle prefixes.
  3. Reports statistics (supports/refutes the hypothesis).
  4. Optionally writes a new VideoStore with idle prefixes trimmed.

Idle definition
---------------
A frame is "idle" if the L2 norm of the 7 joint-velocity dimensions of the
action vector is below --vel-threshold.  The idle prefix is the maximal
leading run of idle frames.  A minimum episode length of --min-length is
enforced: if trimming would leave fewer frames, the episode is kept as-is.

Usage — analysis only::

    python scripts/utils/trim_droid_idle_beginnings.py \\
        --store ${DATA_ROOT}/droid_converted \\
        --tasks-json ${DATA_ROOT}/droid_converted/tasks.json \\
        --task-filter "prepare plate - milk right" "prepare plate - milk middle"

Usage — with trimming::

    python scripts/utils/trim_droid_idle_beginnings.py \\
        --store ${DATA_ROOT}/droid_converted \\
        --tasks-json ${DATA_ROOT}/droid_converted/tasks.json \\
        --task-filter "prepare plate - milk right" "prepare plate - milk middle" \\
        --output ${DATA_ROOT}/droid_converted_trimmed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
# robo_dataset lives in the main repo's third_party; git worktrees share the
# object database but don't reinitialize submodules, so walk up to find it.
for _candidate in [REPO, *REPO.parents]:
    _rd = _candidate / "third_party" / "robo_dataset"
    if (_rd / "robo_dataset").is_dir():
        sys.path.insert(0, str(_rd))
        break

log = logging.getLogger(__name__)


# ── Idle detection ────────────────────────────────────────────────────────────


def _joint_vel_norms(action: np.ndarray) -> np.ndarray:
    """Per-frame L2 norm of joint-velocity dims (first 7 of action vector)."""
    return np.linalg.norm(action[:, :7], axis=1)


def find_idle_prefix_length(
    action: np.ndarray,
    vel_threshold: float,
    smooth_window: int = 3,
) -> int:
    """Return the number of idle frames at the start of an episode.

    A frame is active if its joint-velocity norm >= vel_threshold.  We use a
    small running-mean window to avoid single-frame noise spikes triggering
    an early end to the idle prefix.

    Returns 0 if the episode is immediately active.
    """
    norms = _joint_vel_norms(action)
    T = len(norms)
    if T == 0:
        return 0

    # Smooth with a small causal window (mean of preceding frames including self)
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        # 'full' convolution then truncate to length T
        smoothed = np.convolve(norms, kernel, mode="full")[:T]
    else:
        smoothed = norms

    for i in range(T):
        if smoothed[i] >= vel_threshold:
            return i
    return T  # entire episode is idle


# ── Per-store analysis ────────────────────────────────────────────────────────


def analyze_store(
    store_path: str,
    tasks_json_path: str | None,
    task_filter: list[str] | None,
    vel_threshold: float,
    smooth_window: int,
) -> dict:
    """Return analysis results dict.

    Returns
    -------
    dict with keys:
      n_total_eps, n_filtered_eps,
      idle_lengths: list of int (one per filtered episode),
      episode_lengths: list of int,
      trim_starts: dict[ep_idx -> trim_start],
      task_names: dict[ep_idx -> str],
    """
    from robo_dataset.backends.video import VideoStore

    # Read per-episode action arrays via the store API (episode-indexed layout;
    # the old concatenated data/action + episode_ends.npy no longer exist). The
    # action key is lowdim, so this is a fast zarr read — no video decode.
    store = VideoStore(store_path)
    n_total = store.n_episodes

    tasks: dict[int, str] = {}
    if tasks_json_path and Path(tasks_json_path).exists():
        with open(tasks_json_path) as f:
            raw = json.load(f)
        tasks = {int(k): v for k, v in raw.items()}

    results = {
        "n_total_eps": n_total,
        "n_filtered_eps": 0,
        "idle_lengths": [],
        "episode_lengths": [],
        "trim_starts": {},
        "task_names": tasks,
        "vel_threshold": vel_threshold,
    }

    for ep_idx in range(n_total):
        task_name = tasks.get(ep_idx, "")
        if task_filter and not any(f in task_name for f in task_filter):
            continue

        ep_len = store.episode_length(ep_idx)
        action = store.get_frames(ep_idx, "action", 0, ep_len)
        idle = find_idle_prefix_length(action, vel_threshold, smooth_window)

        results["n_filtered_eps"] += 1
        results["idle_lengths"].append(idle)
        results["episode_lengths"].append(ep_len)
        results["trim_starts"][ep_idx] = idle

    return results


def print_analysis(results: dict, fps: float = 15.0) -> None:
    idle = np.array(results["idle_lengths"])
    ep_len = np.array(results["episode_lengths"])
    n = results["n_filtered_eps"]
    threshold = results["vel_threshold"]

    log.info("─" * 60)
    log.info("DROID idle-beginning analysis")
    log.info("  total episodes in store : %d", results["n_total_eps"])
    log.info("  task-filtered episodes  : %d", n)
    log.info("  vel_threshold           : %.4f", threshold)
    log.info("─" * 60)

    if n == 0:
        log.warning("No episodes matched the task filter.")
        return

    has_idle = idle > 0
    log.info(
        "Episodes with any idle prefix : %d / %d (%.1f%%)", has_idle.sum(), n, 100 * has_idle.mean()
    )
    log.info("Idle prefix length (frames) — all filtered episodes:")
    log.info("  min  : %d (%.2fs)", idle.min(), idle.min() / fps)
    log.info("  p25  : %.1f (%.2fs)", np.percentile(idle, 25), np.percentile(idle, 25) / fps)
    log.info("  p50  : %.1f (%.2fs)", np.percentile(idle, 50), np.percentile(idle, 50) / fps)
    log.info("  p75  : %.1f (%.2fs)", np.percentile(idle, 75), np.percentile(idle, 75) / fps)
    log.info("  p90  : %.1f (%.2fs)", np.percentile(idle, 90), np.percentile(idle, 90) / fps)
    log.info("  max  : %d (%.2fs)", idle.max(), idle.max() / fps)
    log.info("  mean : %.1f (%.2fs)", idle.mean(), idle.mean() / fps)

    trim_frac = idle / ep_len
    log.info("Idle fraction of episode length:")
    log.info("  mean : %.3f", trim_frac.mean())
    log.info("  p50  : %.3f", np.percentile(trim_frac, 50))
    log.info("  p90  : %.3f", np.percentile(trim_frac, 90))

    nonempty_idle = idle[idle > 0]
    if len(nonempty_idle):
        log.info("Non-zero idle durations (%.0f%% of episodes):", 100 * has_idle.mean())
        log.info("  mean : %.1f frames (%.2fs)", nonempty_idle.mean(), nonempty_idle.mean() / fps)
        log.info(
            "  p50  : %.1f frames (%.2fs)",
            np.percentile(nonempty_idle, 50),
            np.percentile(nonempty_idle, 50) / fps,
        )

    log.info("─" * 60)

    # Distribution histogram (ASCII)
    counts, bins = np.histogram(idle, bins=10)
    log.info("Histogram of idle prefix lengths (frames):")
    for i, c in enumerate(counts):
        bar = "█" * min(c, 40)
        log.info("  [%4.0f-%4.0f] %s (%d)", bins[i], bins[i + 1], bar, c)
    log.info("─" * 60)


# ── Trimmed dataset writer ────────────────────────────────────────────────────


def _reencode_trimmed_episode(
    src_path: str,
    out_path: str,
    key: str,
    ep_idx: int,
    trim: int,
    ep_len: int,
    fps: float,
    crf: int,
    kfi: int,
) -> None:
    """Decode an MP4 episode, skip the first `trim` frames, re-encode to out_path."""
    import fractions

    import av as av_mod

    src_mp4 = os.path.join(src_path, "videos", key, f"episode_{ep_idx:06d}.mp4")
    with av_mod.open(src_mp4, "r") as ic:
        video_stream = ic.streams.video[0]
        all_frames = [f.to_ndarray(format="rgb24") for f in ic.decode(video_stream)]

    if len(all_frames) != ep_len:
        raise ValueError(
            f"Episode {ep_idx}, key {key!r}: expected {ep_len} frames, got {len(all_frames)}"
        )
    trimmed = all_frames[trim:]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rate = fractions.Fraction(fps).limit_denominator(1000)
    codec_opts = {"crf": str(crf), "g": str(kfi), "preset": "fast"}
    with av_mod.open(out_path, "w") as oc:
        H, W = trimmed[0].shape[:2]
        stream = oc.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.options = codec_opts
        for arr in trimmed:
            for p in stream.encode(av_mod.VideoFrame.from_ndarray(arr, format="rgb24")):
                oc.mux(p)
        for p in stream.encode():
            oc.mux(p)


def write_trimmed_store(
    src_path: str,
    output_path: str,
    trim_starts: dict[int, int],
    tasks_json_path: str | None,
    min_length: int,
) -> None:
    """Copy src VideoStore to output_path, trimming episode starts per trim_starts.

    Episodes not in trim_starts (non-filtered tasks) are copied unmodified.
    If trimming would leave fewer than min_length frames, episode is kept as-is.

    Fast path: untrimmed episodes have their MP4 files hard-linked (or copied)
    directly without decode/encode.  Only trimmed episodes go through the
    full decode → trim → re-encode pipeline.
    """
    import json as _json
    import shutil

    from robo_dataset.backends.video import VideoStore, VideoStoreWriter

    src = VideoStore(src_path)
    output_path_obj = Path(output_path)

    if output_path_obj.exists() and any(output_path_obj.iterdir()):
        raise RuntimeError(
            f"Output path {output_path} already exists and is non-empty. "
            "Remove it first to avoid overwriting data."
        )

    with open(os.path.join(src_path, "meta.json")) as f:
        meta = _json.load(f)

    # Re-encode every episode through the episode-indexed VideoStoreWriter,
    # dropping each episode's idle prefix. This keeps the trimmed store in the
    # same per-episode layout convert_droid_data.py produces (all keys are read
    # back through the store API, so old vs new source layout doesn't matter).
    dst = VideoStoreWriter.create(
        path=str(output_path_obj),
        key_shapes={k: src.frame_shape(k) for k in src.keys},
        key_dtypes={k: src.frame_dtype(k) for k in src.keys},
        image_keys=list(src._video_keys),
        schemas=src.schemas,
        vcodec=meta.get("vcodec", "libx264"),
        crf=int(meta.get("crf", 23)),
        kfi=int(meta.get("kfi", 1)),
        fps=float(meta.get("fps", 15.0)),
    )

    n_trimmed = 0
    n_kept_too_short = 0
    total_frames_removed = 0

    from tqdm import tqdm

    for ep_idx in tqdm(range(src.n_episodes), desc="Writing trimmed store"):
        ep_len = src.episode_length(ep_idx)
        trim = trim_starts.get(ep_idx, 0)
        if trim > 0 and (ep_len - trim) < min_length:
            trim = 0
            n_kept_too_short += 1

        ep = {k: src.get_frames(ep_idx, k, trim, ep_len) for k in src.keys}
        dst.put_episode(ep)

        if trim > 0:
            n_trimmed += 1
            total_frames_removed += trim

    dst.finalize()

    # Copy tasks.json if present (episode indices are unchanged — 1-to-1 copy).
    if tasks_json_path and Path(tasks_json_path).exists():
        shutil.copy2(tasks_json_path, output_path_obj / "tasks.json")

    log.info("Trimmed store written to: %s", output_path)
    log.info("  total episodes          : %d", src.n_episodes)
    log.info("  episodes trimmed        : %d", n_trimmed)
    log.info("  kept as-is (too short)  : %d", n_kept_too_short)
    log.info("  total frames removed    : %d", total_frames_removed)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze and trim idle beginnings from a DROID VideoStore.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--store", required=True, help="Path to converted VideoStore.")
    parser.add_argument(
        "--tasks-json",
        default=None,
        help="Path to tasks.json sidecar. Defaults to <store>/tasks.json.",
    )
    parser.add_argument(
        "--task-filter",
        nargs="*",
        default=None,
        help="Analyze/trim only episodes whose task name contains one of these strings. "
        "If omitted, all episodes are analyzed.",
    )
    parser.add_argument(
        "--vel-threshold",
        type=float,
        default=0.03,
        help="Joint-velocity L2-norm threshold below which a frame is 'idle' (default: 0.03).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=3,
        help="Causal mean window size for velocity smoothing before threshold test (default: 3).",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=8,
        help="Minimum episode length (frames) after trimming. Episodes that would fall below "
        "this are kept as-is (default: 8).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="If set, write trimmed VideoStore to this path. Skipped if not provided.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        default=False,
        help="Run analysis only, do not write output even if --output is given.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    tasks_json = args.tasks_json or str(Path(args.store) / "tasks.json")

    log.info("Analyzing store: %s", args.store)
    log.info("Task filter: %s", args.task_filter)
    log.info("vel_threshold=%.4f  smooth_window=%d", args.vel_threshold, args.smooth_window)

    results = analyze_store(
        store_path=args.store,
        tasks_json_path=tasks_json,
        task_filter=args.task_filter,
        vel_threshold=args.vel_threshold,
        smooth_window=args.smooth_window,
    )

    # Load FPS from meta.json for time-based reporting
    meta_path = Path(args.store) / "meta.json"
    fps = 15.0
    if meta_path.exists():
        with open(meta_path) as f:
            fps = float(json.load(f).get("fps", 15.0))

    print_analysis(results, fps=fps)

    if args.output and not args.analyze_only:
        log.info("Writing trimmed store to: %s", args.output)
        write_trimmed_store(
            src_path=args.store,
            output_path=args.output,
            trim_starts=results["trim_starts"],
            tasks_json_path=tasks_json,
            min_length=args.min_length,
        )
    elif not args.output:
        log.info("(Pass --output <path> to write a trimmed dataset)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
