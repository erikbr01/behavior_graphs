#!/usr/bin/env python3
"""Generate and write a frozen episode index list for the prepare_tray_baseline config.

Mirrors ``generate_set_table_episode_selection.py``: the episode indices are
written directly into the Hydra config YAML as ``task.dataset.episodes_per_task``
so training is fully reproducible without any runtime sampling.

Selection logic
---------------
For each task: ``sorted(rng(seed).choice(all_task_eps, size=60, replace=False))``

The store itself must contain only successful episodes (convert from
``droid_raw/success/...``, not the split root) — this script asserts that
via the `reward` channel as a safety check, it does not filter.

Usage
-----
    python scripts/utils/generate_prepare_tray_episode_selection.py \\
        --store ${DATA_ROOT}/droid_converted_prepare_tray_trimmed \\
        --config-dir third_party/diffusion_policy/configs/image/droid \\
        --seed 42

Run with --dry-run to print indices without modifying any files.
Run with --verify-only to check the existing config without regenerating.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _candidate in [REPO, *REPO.parents]:
    _rd = _candidate / "third_party" / "robo_dataset"
    if (_rd / "robo_dataset").is_dir():
        sys.path.insert(0, str(_rd))
        break

TASKS = ["prepare tray - top", "prepare tray - bottom"]
RUN_NAME = "prepare_tray_baseline"
EPISODES_PER_TASK = 60

# ---------------------------------------------------------------------------
# Index generation
# ---------------------------------------------------------------------------


def load_task_episodes(store_path: Path) -> dict[str, list[int]]:
    tasks_file = store_path / "tasks.json"
    if not tasks_file.exists():
        sys.exit(f"ERROR: {tasks_file} not found — has the store been converted?")
    with open(tasks_file) as f:
        tasks_map = json.load(f)
    task_to_eps: dict[str, list[int]] = defaultdict(list)
    for ep_idx_str, task in tasks_map.items():
        task_to_eps[task].append(int(ep_idx_str))
    return {k: sorted(v) for k, v in task_to_eps.items()}


def assert_all_successful(store_path: Path) -> None:
    """Fail loudly if the store contains any non-success episode.

    `convert_droid_data.py` writes `reward[-1, 0] = float(success)` from the
    raw session's `trajectory.h5` attrs (see robo_dataset/backends/droid.py).
    The store is expected to be converted from `droid_raw/success/...` only
    (not the split root), so every episode should have reward 1.0; this is a
    safety check against accidentally training on failed demonstrations, not
    a filter.
    """
    from robo_dataset.backends.video import VideoStore

    store = VideoStore(str(store_path))
    failures = []
    for ep in range(store.n_episodes):
        ep_len = store.episode_length(ep)
        reward = store.get_frames(ep, "reward", ep_len - 1, ep_len)
        if reward[0, 0] <= 0.5:
            failures.append(ep)
    if failures:
        sys.exit(
            f"ERROR: {store_path} contains {len(failures)} non-success episode(s) "
            f"(e.g. {failures[:5]}) — convert from droid_raw/success/ only, not the split root."
        )


def base_select(eps: list[int], cap: int, seed: int) -> list[int]:
    """Stable base selection: sorted choice without replacement."""
    arr = np.array(eps)
    if cap >= len(arr):
        return sorted(arr.tolist())
    return sorted(np.random.default_rng(seed).choice(arr, size=cap, replace=False).tolist())


def generate_selection(task_to_eps: dict, seed: int) -> dict[str, list[int]]:
    return {task: base_select(task_to_eps[task], EPISODES_PER_TASK, seed) for task in TASKS}


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------


def read_config(path: Path) -> str:
    return path.read_text()


def write_episodes_per_task(config_text: str, eps_per_task: dict[str, list[int]]) -> str:
    lines = ["    episodes_per_task:"]
    for task, indices in eps_per_task.items():
        lines.append(f"      {task}:")
        lines.append(f"        {json.dumps(indices)}")
    block = "\n".join(lines)

    # Anchored to line starts so this only consumes lines nested under the
    # header (>= 6 spaces indent), not the first "4 spaces + word" substring
    # it happens to find inside a deeply-indented nested line.
    config_text = re.sub(
        r"^    episodes_per_task:\n(?: {6,}.*\n)*",
        "",
        config_text,
        flags=re.MULTILINE,
    )
    config_text = re.sub(r"^    task_filter:\n(?: {6,}.*\n)*", "", config_text, flags=re.MULTILINE)
    config_text = re.sub(r"^    max_episodes_per_task:.*\n", "", config_text, flags=re.MULTILINE)

    config_text = config_text.replace("    val_ratio:", block + "\n    val_ratio:")
    return config_text


def read_episodes_per_task(config_text: str) -> dict[str, list[int]]:
    block_m = re.search(r"episodes_per_task:(.*?)val_ratio", config_text, re.DOTALL)
    if not block_m:
        return {}
    result = {}
    for m in re.finditer(r"(prepare tray - \w+):\s*\n\s*(\[.*?\])", block_m.group(1)):
        result[m.group(1)] = json.loads(m.group(2))
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_selection(eps_per_task: dict, task_to_eps: dict) -> bool:
    ok = True
    for task, indices in eps_per_task.items():
        if len(indices) != len(set(indices)):
            print(f"  FAIL {task}: duplicate indices")
            ok = False
        max_ep = max(task_to_eps.get(task, [0]))
        invalid = [i for i in indices if i > max_ep]
        if invalid:
            print(f"  FAIL {task}: indices out of range: {invalid}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--store", required=True, help="Path to the converted VideoStore (must contain tasks.json)"
    )
    parser.add_argument(
        "--config-dir",
        required=True,
        help="Path to configs/image/droid/ in the diffusion_policy submodule",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print indices but do not write config files"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Read the existing config and verify without regenerating",
    )
    args = parser.parse_args()

    store_path = Path(args.store)
    config_dir = Path(args.config_dir)
    assert_all_successful(store_path)
    task_to_eps = load_task_episodes(store_path)

    cfg_path = config_dir / RUN_NAME / "config.yaml"
    if args.verify_only:
        print("Reading existing config...")
        eps_per_task = read_episodes_per_task(read_config(cfg_path))
    else:
        print(f"Generating episode selection (seed={args.seed})...")
        eps_per_task = generate_selection(task_to_eps, args.seed)

    print("\nVerifying...")
    ok = verify_selection(eps_per_task, task_to_eps)
    if ok:
        print("  All checks passed ✓")
    else:
        print("  VERIFICATION FAILED")
        sys.exit(1)

    counts = {t: len(v) for t, v in eps_per_task.items()}
    print(f"  {RUN_NAME}: {counts}")

    if args.verify_only or args.dry_run:
        if args.dry_run:
            print("\n[dry-run] no files written")
        return

    print("\nWriting config...")
    updated = write_episodes_per_task(read_config(cfg_path), eps_per_task)
    cfg_path.write_text(updated)
    print(f"  wrote {cfg_path}")

    print("\nDone. Commit the updated config to record the selection.")


if __name__ == "__main__":
    main()
