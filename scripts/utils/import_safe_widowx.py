"""Convert SAFE's public WidowX/OpenVLA release into the canonical trajectory layout.

SAFE (Gu et al., "Multitask Failure Detection for Vision-Language-Action
Models," NeurIPS 2025, arXiv:2506.09937) publishes real, labeled rollout
data for a WidowX arm running OpenVLA -- per episode: an mp4, a per-step
CSV of OpenVLA's own action-token entropy/probability statistics, and a
pkl with the policy's per-timestep hidden states (list of ``(7, 4096)``
bfloat16 tensors, one row per discretized action token) plus
``episode_success``/``task_description``/``task_id``. Source (not hosted
here, ~1.3GB): https://drive.google.com/file/d/1EwaccasZjnlM9L6SEYyWqTd7d6-BR9zp

This exists so ``behavior_graphs.outcome_classifier`` can be evaluated on the
same real rollout data used by SAFE -- a genuinely faithful comparison
axis (same episodes, same policy, same labels) without needing to train
or run any model ourselves, unlike a from-scratch reimplementation of
SAFE's own method would require.

Requires ``torch`` to unpickle the bfloat16 hidden-state tensors (not a
runtime dependency of this repo otherwise -- install with
``uv run --with torch`` or the ``data-import`` extra). Everything written
to disk afterward is plain float32 npz; no further torch dependency.

The archive's per-task folders are named ``task_<slug>_<session>`` (e.g.
``task_lift_red_bottle_1``, ``_2``, ``_3``, ``_4``) -- multiple collection
sessions of the IDENTICAL task (same ``task_id``/``task_description``
inside every episode's pkl, verified by inspection), not distinct tasks.
This script merges all sessions sharing a ``task_id`` into one dataset,
since splitting them would leave several tasks with under 10 episodes --
too few for a meaningful train/held-out split.

The per-timestep hidden state is ``(7, 4096)`` -- one row per OpenVLA
action token -- mean-pooled over the 7 tokens to a single 4096-dim vector
per timestep, written as embedding layer ``openvla_hidden_mean``.

Usage::

    uv run --with torch python scripts/utils/import_safe_widowx.py \\
        --archive-dir /path/to/extracted/openvla_widowx \\
        --output-dir $DATA_PATH/rollouts
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_graphs.envs.safe_import import slugify, write_safe_task  # noqa: E402

log = logging.getLogger("import_safe_widowx")

EMBEDDING_LAYER = "openvla_hidden_mean"


def _load_episode(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def discover_episodes(archive_dir: Path) -> dict[int, list[Path]]:
    """Group every episode pkl in the archive by its (identical-across-sessions) task_id."""
    by_task: dict[int, list[Path]] = defaultdict(list)
    task_names: dict[int, str] = {}
    for pkl_path in sorted(archive_dir.glob("*/*.pkl")):
        data = _load_episode(pkl_path)
        task_id = int(data["task_id"])
        by_task[task_id].append(pkl_path)
        task_names.setdefault(task_id, data["task_description"])
    for task_id, paths in sorted(by_task.items()):
        log.info(
            "task_id=%d %-32s %d episodes across %d session folder(s)",
            task_id,
            task_names[task_id],
            len(paths),
            len({p.parent.name for p in paths}),
        )
    return by_task


def convert_task(task_id: int, pkl_paths: list[Path], output_dir: Path) -> Path:
    data = [_load_episode(p) for p in pkl_paths]
    description = data[0]["task_description"]
    name = f"safe_widowx_{slugify(description)}"
    dest = output_dir / name

    lengths, successes, extras, embedding_rows = [], [], [], []
    for pkl_path, episode in zip(pkl_paths, data):
        hidden_states = episode["hidden_states"]  # list of (7, 4096) bfloat16 torch tensors
        # .float() before .numpy(): numpy has no native bfloat16 dtype.
        per_step = np.stack([h.float().numpy().mean(axis=0) for h in hidden_states], axis=0).astype(
            np.float32
        )
        lengths.append(per_step.shape[0])
        successes.append(bool(episode["episode_success"]))
        embedding_rows.append(per_step)
        extras.append(
            {
                "safe_session_dir": pkl_path.parent.name,
                "safe_episode_idx": int(episode["eposide_idx"]),
            }
        )

    embeddings = {EMBEDDING_LAYER: np.concatenate(embedding_rows, axis=0)}
    write_safe_task(
        log,
        dest,
        lengths,
        successes,
        embeddings,
        dataset_label="SAFE WidowX/OpenVLA (arXiv:2506.09937)",
        task_id=task_id,
        task_description=description,
        episode_extras=extras,
    )
    return dest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        required=True,
        help="Path to the extracted 'openvla_widowx' directory (task_* subfolders of pkl/csv/mp4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write one 'safe_widowx_<task>' trajectory dir per task into.",
    )
    args = parser.parse_args()

    by_task = discover_episodes(args.archive_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_id, pkl_paths in sorted(by_task.items()):
        convert_task(task_id, pkl_paths, args.output_dir)


if __name__ == "__main__":
    main()
