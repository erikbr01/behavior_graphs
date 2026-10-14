"""Convert SAFE's public Franka/pi0-FAST-DROID release into the canonical trajectory layout.

SAFE (Gu et al., "Multitask Failure Detection for Vision-Language-Action
Models," NeurIPS 2025, arXiv:2506.09937) publishes real, labeled rollout
data for a Franka arm running pi0-FAST-DROID. Source (not hosted here,
~25.5GB): https://drive.google.com/file/d/13z_cdwnaJota2iHkZbhYgVALujZwtM3b

Unlike the WidowX/OpenVLA release (see ``import_safe_widowx.py``), this
archive lays episodes out as two parallel record types per session folder
(``rollouts_all/<session>/<task_slug>/``):

- ``env_records/task{T}--ep{E}--succ{S}--meta.pkl``: one per episode --
  ``task_id``, ``episode_idx``, ``episode_success``, ``replan_steps``.
- ``policy_records/step_<global>--<run_note>--task_<T>--ep_<E>--t_<local>--meta.pkl``:
  one per *replan step* within an episode (the policy replans every
  ``replan_steps`` raw env timesteps, not every timestep) -- each holds
  ``pre_logits``/``encoded``/``logits`` of shape ``(n_action_tokens, 2048)``,
  the pi0-FAST backbone's per-token internal representation for that
  replan call. ``pre_logits`` is used here (mean-pooled over tokens) as
  the closest analogue to WidowX's raw ``hidden_states`` -- the backbone's
  representation before the final action-token unembedding, not the
  post-unembedding token scores (``logits``).

Episode indices reset to 0 within each session folder, so episodes are
keyed by ``(session_dir, episode_idx)``, not ``episode_idx`` alone --
verified by inspection (the same ``task_id``/``episode_idx`` pair recurs
across multiple session folders collected on different dates). As with
the WidowX import, multiple session folders sharing a ``task_id`` are
merged into one dataset.

Reads directly from the zip archive (``zipfile``, streaming one member at
a time) rather than extracting first -- the archive is dominated by mp4
video that's never needed here, so extracting everything first would cost
much more disk and time for no benefit. No ``torch`` dependency needed
(unlike the WidowX import): these pkls hold plain numpy arrays.

Usage::

    python scripts/utils/import_safe_franka.py \\
        --zip-path /path/to/pi0fast_droid_0510_all.zip \\
        --output-dir $DATA_PATH/rollouts
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_graphs.envs.safe_import import slugify, write_safe_task  # noqa: E402

log = logging.getLogger("import_safe_franka")

EMBEDDING_LAYER = "pi0fast_prelogits_mean"

_ENV_RE = re.compile(r"env_records/task(\d+)--ep(\d+)--succ(\d+)--meta\.pkl$")
_POLICY_RE = re.compile(
    r"policy_records/step_\d+--[^-]+--task_(\d+)--ep_(\d+)--t_(\d+)--meta\.pkl$"
)


def _session_dir(member_name: str) -> str:
    """``rollouts_all/<session>/<task_folder>/{env,policy}_records/...`` -> ``<session>/<task_folder>``."""
    parts = member_name.split("/")
    return "/".join(parts[1:3])


def discover(zf: zipfile.ZipFile) -> tuple[dict, dict]:
    """Group member names by ``(session_dir, episode_idx)``, keyed separately for env/policy records."""
    env_by_episode: dict[tuple[str, int], str] = {}
    policy_by_episode: dict[tuple[str, int], list[tuple[int, str]]] = defaultdict(list)
    for name in zf.namelist():
        m = _ENV_RE.search(name)
        if m:
            _task_id, ep_idx, _succ = m.groups()
            env_by_episode[(_session_dir(name), int(ep_idx))] = name
            continue
        m = _POLICY_RE.search(name)
        if m:
            _task_id, ep_idx, t = m.groups()
            policy_by_episode[(_session_dir(name), int(ep_idx))].append((int(t), name))
    return env_by_episode, policy_by_episode


def load_episode(
    zf: zipfile.ZipFile, env_name: str, policy_steps: list[tuple[int, str]]
) -> tuple[np.ndarray, bool, int, str]:
    with zf.open(env_name) as f:
        env_meta = pickle.load(f)
    steps = sorted(set(policy_steps))  # de-dupe: identical (t, name) pairs are the same read
    rows = []
    for _t, name in steps:
        with zf.open(name) as f:
            step_meta = pickle.load(f)
        rows.append(step_meta["pre_logits"].mean(axis=0).astype(np.float32))
    embedding = np.stack(rows, axis=0) if rows else np.zeros((0, 2048), dtype=np.float32)
    return (
        embedding,
        bool(env_meta["episode_success"]),
        int(env_meta["task_id"]),
        str(env_meta["task_description"]),
    )


def convert(zip_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        env_by_episode, policy_by_episode = discover(zf)
        log.info(
            "found %d episodes (env_records) / %d with policy_records",
            len(env_by_episode),
            len(policy_by_episode),
        )

        by_task: dict[int, dict] = defaultdict(
            lambda: {"lengths": [], "successes": [], "rows": [], "extras": [], "description": None}
        )
        skipped_no_policy = 0
        for key, env_name in sorted(env_by_episode.items()):
            policy_steps = policy_by_episode.get(key)
            if not policy_steps:
                skipped_no_policy += 1
                continue
            session_dir, ep_idx = key
            embedding, success, task_id, description = load_episode(zf, env_name, policy_steps)
            if embedding.shape[0] == 0:
                skipped_no_policy += 1
                continue
            bucket = by_task[task_id]
            bucket["description"] = description
            bucket["lengths"].append(embedding.shape[0])
            bucket["successes"].append(success)
            bucket["rows"].append(embedding)
            bucket["extras"].append({"safe_session_dir": session_dir, "safe_episode_idx": ep_idx})

        if skipped_no_policy:
            log.warning("skipped %d episode(s) with no policy_records steps", skipped_no_policy)

        output_dir.mkdir(parents=True, exist_ok=True)
        for task_id, bucket in sorted(by_task.items()):
            name = f"safe_franka_{slugify(bucket['description'])}"
            dest = output_dir / name
            embeddings = {EMBEDDING_LAYER: np.concatenate(bucket["rows"], axis=0)}
            write_safe_task(
                log,
                dest,
                bucket["lengths"],
                bucket["successes"],
                embeddings,
                dataset_label="SAFE Franka/pi0-FAST-DROID (arXiv:2506.09937)",
                task_id=task_id,
                task_description=bucket["description"],
                episode_extras=bucket["extras"],
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert(args.zip_path, args.output_dir)


if __name__ == "__main__":
    main()
