"""Walk a raw DROID dataset directory and print per-task success/failure counts."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections import defaultdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        required=True,
        type=str,
        help="Root of the raw DROID session tree.",
    )
    args = parser.parse_args(argv)

    import h5py

    success_counts: dict[str, int] = defaultdict(int)
    failure_counts: dict[str, int] = defaultdict(int)

    for root, _dirs, files in sorted(os.walk(args.data_dir, followlinks=True)):
        if "trajectory.h5" not in files:
            continue
        h5_path = os.path.join(root, "trajectory.h5")
        try:
            with h5py.File(h5_path, "r") as f:
                task = str(f.attrs.get("current_task", ""))
                success = bool(f.attrs.get("success", False))
        except Exception as exc:
            warnings.warn(f"skipping {h5_path}: {exc}")
            continue
        if success:
            success_counts[task] += 1
        else:
            failure_counts[task] += 1

    all_tasks = set(success_counts) | set(failure_counts)
    rows = sorted(
        all_tasks,
        key=lambda t: success_counts.get(t, 0),
        reverse=True,
    )

    col_w = max((len(t) for t in all_tasks), default=4)
    col_w = max(col_w, 4)
    header = f"{'Task':<{col_w}}  {'success':>7}  {'failure':>7}"
    print(header)
    print("-" * len(header))
    for task in rows:
        s = success_counts.get(task, 0)
        f = failure_counts.get(task, 0)
        print(f"{task:<{col_w}}  {s:>7}  {f:>7}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
