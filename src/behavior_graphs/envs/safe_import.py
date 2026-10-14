"""Shared helpers for ``scripts/utils/import_safe_{widowx,franka}.py``.

Both scripts convert one arm/policy's slice of SAFE's public release (Gu et
al., "Multitask Failure Detection for Vision-Language-Action Models,"
NeurIPS 2025, arXiv:2506.09937) into the canonical trajectory layout,
differing only in how they read that arm's archive format (pkl-per-episode
for WidowX vs. a zipped env/policy-record pair for Franka) -- everything
after that (task-name slugging, the ``source={...}`` provenance dict shape,
the per-task "wrote N episodes" log line) is identical, so it lives here
once instead of twice.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory, write_trajectory_dir

SAFE_SOURCE_URL = "https://github.com/vla-safe/SAFE"


def slugify(description: str) -> str:
    """``"Pick up the knife"`` -> ``"pick_up_the_knife"``, for task directory names."""
    return re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")


def write_safe_task(
    log: logging.Logger,
    dest: Path,
    lengths: list[int],
    successes: list[bool],
    embeddings: dict[str, np.ndarray],
    dataset_label: str,
    task_id: int,
    task_description: str,
    episode_extras: list[dict[str, Any]],
) -> TrajectoryDirectory:
    """Write one SAFE task's trajectory dir + provenance, logging the episode split."""
    written = write_trajectory_dir(
        dest,
        lengths,
        successes,
        embeddings,
        source={
            "kind": "safe_release",
            "dataset": dataset_label,
            "url": SAFE_SOURCE_URL,
            "task_id": task_id,
            "task_description": task_description,
        },
        episode_extras=episode_extras,
    )
    n_success = sum(successes)
    log.info(
        "wrote %s: %d episodes (%d success / %d failure)",
        dest,
        len(lengths),
        n_success,
        len(lengths) - n_success,
    )
    return written
