#!/usr/bin/env python3
"""Generate and write frozen episode index lists for the set_table_v2 training configs.

The episode indices are written directly into the Hydra config YAML files as
``task.dataset.episodes_per_task`` so that training is fully reproducible without
any runtime sampling.

Selection logic
---------------
Base-60 (right and middle, shared by ALL runs):
    sorted(rng(seed).choice(all_task_eps, size=60, replace=False))

Extra-60 (right for plusright, middle for plusmiddle):
    rng(seed).permutation(all_task_eps - base_set)[:60]

The permutation approach for extra indices guarantees that base_set ⊆ total_set
regardless of cap size — choice() does not guarantee this across different sizes.

Usage
-----
    python scripts/generate_set_table_episode_selection.py \\
        --store ${DATA_ROOT}/droid_converted_set_table_trimmed \\
        --config-dir third_party/diffusion_policy/configs/image/droid \\
        --seed 42

Run with --dry-run to print indices without modifying any files.
Run with --verify-only to check existing configs without regenerating.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

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


def base_select(eps: list[int], cap: int, seed: int) -> list[int]:
    """Stable base selection: sorted choice without replacement."""
    arr = np.array(eps)
    if cap >= len(arr):
        return sorted(arr.tolist())
    return sorted(np.random.default_rng(seed).choice(arr, size=cap, replace=False).tolist())


def extra_select(eps: list[int], already_used: set[int], n: int, seed: int) -> list[int]:
    """Extra selection from episodes NOT in already_used, via permutation for nestability."""
    remaining = np.array([e for e in eps if e not in already_used])
    if n >= len(remaining):
        return sorted(remaining.tolist())
    return sorted(np.random.default_rng(seed).permutation(remaining)[:n].tolist())


def generate_selections(task_to_eps: dict, seed: int) -> dict[str, dict[str, list[int]]]:
    right = task_to_eps["prepare plate - milk right"]
    middle = task_to_eps["prepare plate - milk middle"]
    left = task_to_eps["prepare plate - milk left"]

    base_right = base_select(right, 60, seed)
    base_middle = base_select(middle, 60, seed)
    base_left = base_select(left, 60, seed)

    extra_right = extra_select(right, set(base_right), 60, seed)
    extra_middle = extra_select(middle, set(base_middle), 60, seed)

    return {
        "set_table_v2_baseline": {
            "prepare plate - milk right": base_right,
            "prepare plate - milk middle": base_middle,
        },
        "set_table_v2_plusleft": {
            "prepare plate - milk right": base_right,
            "prepare plate - milk middle": base_middle,
            "prepare plate - milk left": base_left,
        },
        "set_table_v2_plusright": {
            "prepare plate - milk right": base_right + extra_right,
            "prepare plate - milk middle": base_middle,
        },
        "set_table_v2_plusmiddle": {
            "prepare plate - milk right": base_right,
            "prepare plate - milk middle": base_middle + extra_middle,
        },
    }


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

    # Remove existing episodes_per_task block if present
    config_text = re.sub(
        r"    episodes_per_task:.*?(?=    \w)",
        "",
        config_text,
        flags=re.DOTALL,
    )
    # Remove legacy task_filter / max_episodes_per_task if present
    config_text = re.sub(r"    task_filter:.*?(?=    \w|\Z)", "", config_text, flags=re.DOTALL)
    config_text = re.sub(r"    max_episodes_per_task:.*?\n", "", config_text)

    # Insert before val_ratio
    config_text = config_text.replace("    val_ratio:", block + "\n    val_ratio:")
    return config_text


def read_episodes_per_task(config_text: str) -> dict[str, list[int]]:
    block_m = re.search(r"episodes_per_task:(.*?)val_ratio", config_text, re.DOTALL)
    if not block_m:
        return {}
    result = {}
    for m in re.finditer(r"(prepare plate - milk \w+):\s*\n\s*(\[.*?\])", block_m.group(1)):
        result[m.group(1)] = json.loads(m.group(2))
    return result


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_selections(selections: dict, task_to_eps: dict) -> bool:
    base_right = set(selections["set_table_v2_baseline"]["prepare plate - milk right"])
    base_middle = set(selections["set_table_v2_baseline"]["prepare plate - milk middle"])

    ok = True
    for run, eps_per_task in selections.items():
        for task, indices in eps_per_task.items():
            # uniqueness
            if len(indices) != len(set(indices)):
                print(f"  FAIL {run}/{task}: duplicate indices")
                ok = False
            # valid store indices
            max_ep = max(task_to_eps.get(task, [0]))
            invalid = [i for i in indices if i > max_ep]
            if invalid:
                print(f"  FAIL {run}/{task}: indices out of range: {invalid}")
                ok = False

        # baseline subset check
        right_indices = set(eps_per_task.get("prepare plate - milk right", []))
        middle_indices = set(eps_per_task.get("prepare plate - milk middle", []))
        if not base_right.issubset(right_indices):
            print(f"  FAIL {run}: base right not subset of config right")
            ok = False
        if not base_middle.issubset(middle_indices):
            print(f"  FAIL {run}: base middle not subset of config middle")
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
        help="Read existing configs and verify without regenerating",
    )
    args = parser.parse_args()

    store_path = Path(args.store)
    config_dir = Path(args.config_dir)
    task_to_eps = load_task_episodes(store_path)

    if args.verify_only:
        print("Reading existing configs...")
        selections = {}
        for run in [
            "set_table_v2_baseline",
            "set_table_v2_plusleft",
            "set_table_v2_plusright",
            "set_table_v2_plusmiddle",
        ]:
            cfg_path = config_dir / run / "config.yaml"
            selections[run] = read_episodes_per_task(read_config(cfg_path))
    else:
        print(f"Generating episode selections (seed={args.seed})...")
        selections = generate_selections(task_to_eps, args.seed)

    print("\nVerifying...")
    ok = verify_selections(selections, task_to_eps)
    if ok:
        print("  All checks passed ✓")
    else:
        print("  VERIFICATION FAILED")
        sys.exit(1)

    for run, eps_per_task in selections.items():
        counts = {t.split("milk ")[-1]: len(v) for t, v in eps_per_task.items()}
        print(f"  {run}: {counts}")

    if args.verify_only or args.dry_run:
        if args.dry_run:
            print("\n[dry-run] no files written")
        return

    print("\nWriting configs...")
    for run, eps_per_task in selections.items():
        cfg_path = config_dir / run / "config.yaml"
        updated = write_episodes_per_task(read_config(cfg_path), eps_per_task)
        cfg_path.write_text(updated)
        print(f"  wrote {cfg_path}")

    print("\nDone. Commit the updated configs to record the selection.")


if __name__ == "__main__":
    main()
