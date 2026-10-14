"""Merge several canonical trajectory directories into one.

Works for any canonical-layout directories — featurized demos, collected
rollouts, or a mix — re-indexing episodes in argument order::

    # merge featurized demo dirs (output of extract_trajectory_features.py):
    python scripts/utils/merge_droid_dirs.py \\
        /data/demos/lift_abs /data/demos/square_abs \\
        --output /data/demos/merged

    # merge DROID rollout sessions:
    python scripts/utils/merge_droid_dirs.py \\
        /data/rollouts/kendama_jun15 /data/rollouts/kendama_jun16 \\
        --output /data/rollouts/kendama_all

Carries over embeddings, observations, actions, success labels, notes, and
videos. DROID raw recordings (``recordings/ep*/``) are copied along and
re-indexed automatically when present, so embeddings can still be
(re-)extracted from the merged directory later.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_graphs.envs.trajectory_dir import (  # noqa: E402
    TrajectoryDirectory,
    merge_trajectory_dirs,
)

RECORDINGS_DIR = "recordings"


def merge_recordings(sources: list[Path], dest: Path) -> int:
    """Copy raw per-episode recordings along, re-indexed like the episodes."""
    new_idx = 0
    n_copied = 0
    for src in sources:
        n_episodes = len(TrajectoryDirectory(src).episodes())
        for old_idx in range(n_episodes):
            rec = Path(src) / RECORDINGS_DIR / f"ep{old_idx:04d}"
            if rec.is_dir():
                shutil.copytree(rec, dest / RECORDINGS_DIR / f"ep{new_idx:04d}")
                n_copied += 1
            new_idx += 1
    return n_copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge canonical trajectory dirs into one.",
    )
    parser.add_argument(
        "sources", nargs="+", type=Path, help="source directories, in episode order"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory {args.output} exists and is not empty")

    merged = merge_trajectory_dirs(args.sources, args.output)
    episodes = merged.episodes()
    n_recordings = merge_recordings(args.sources, args.output)

    n_succ = sum(1 for ep in episodes if ep.success is True)
    n_fail = sum(1 for ep in episodes if ep.success is False)
    summary = (
        f"{args.output}: {len(episodes)} episodes from {len(args.sources)} "
        f"source(s) — {n_succ} success, {n_fail} failure, "
        f"{len(episodes) - n_succ - n_fail} unlabeled; "
        f"layers: {merged.embedding_layers() or 'none'}"
    )
    if n_recordings:
        summary += f"; {n_recordings} raw recordings carried over"
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
