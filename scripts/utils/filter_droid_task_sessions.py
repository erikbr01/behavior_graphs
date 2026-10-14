"""Symlink only one task's raw DROID sessions into a filtered tree.

`convert_droid_data.py` has no task/success filter of its own -- it converts
every `trajectory.h5` under `--data-dir`. To convert just one language
instruction (e.g. adding a new instruction variant to an existing store, see
docs/droid_workflow.md's "Adding a new instruction variant" section), build a
filtered raw-session tree first and point the converter at that instead. This
only symlinks matching session dirs (no copying), so it's cheap to build and
safe to point straight at the raw dataset.

Like the converter, this refuses a non-empty `--output` (a second run into the
same tree would union two tasks' sessions, and the converter would then
silently produce a mixed-task store), refuses an `--output` nested in
`--data-dir` (the walk follows symlinks, so it would re-walk its own output),
and fails on an unreadable `trajectory.h5` unless `--skip-unreadable` is
passed. Matching zero sessions is an error, not an empty tree.

Usage::

    python scripts/utils/filter_droid_task_sessions.py \\
        --data-dir ${DATA_ROOT}/droid_raw \\
        --output   /tmp/cooking_carrot_sessions \\
        --task "cooking - carrot"

Task strings must match `fetch_droid_language_stats.py`'s output exactly.
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir", required=True, type=str, help="Root of the raw DROID session tree."
    )
    parser.add_argument(
        "--output", required=True, type=str, help="Filtered tree to create (symlinks only)."
    )
    parser.add_argument(
        "--task", required=True, type=str, help="Exact `current_task` string to keep."
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        default=False,
        help="Also link unsuccessful sessions for this task (default: successes only).",
    )
    parser.add_argument(
        "--skip-unreadable",
        action="store_true",
        default=False,
        help="Warn and continue past sessions whose trajectory.h5 cannot be opened (default: abort).",
    )
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    if not data_dir.is_dir():
        parser.error(f"--data-dir {data_dir} is not a directory")
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output {output} already exists and is not empty. Remove it first.")
    if data_dir == output or data_dir in output.parents or output in data_dir.parents:
        parser.error(
            f"--output {output} and --data-dir {data_dir} are nested -- "
            "the session walk follows symlinks and would descend into its own output"
        )

    import h5py

    n = 0
    n_sessions = 0
    n_task_seen = 0
    n_task_failed = 0
    skipped = 0
    for root, _dirs, files in sorted(os.walk(data_dir, followlinks=True)):
        if "trajectory.h5" not in files:
            continue
        n_sessions += 1
        h5_path = os.path.join(root, "trajectory.h5")
        try:
            with h5py.File(h5_path, "r") as f:
                task = str(f.attrs.get("current_task", ""))
                success = bool(f.attrs.get("success", False))
        except Exception as exc:
            if not args.skip_unreadable:
                raise SystemExit(
                    f"cannot read {h5_path}: {exc} -- pass --skip-unreadable to link the "
                    "readable sessions anyway"
                ) from exc
            warnings.warn(f"skipping {h5_path}: {exc}")
            skipped += 1
            continue
        if task != args.task:
            continue
        n_task_seen += 1
        if not success and not args.include_failures:
            n_task_failed += 1
            continue
        rel = os.path.relpath(root, data_dir)
        link_path = output / rel
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.lexists(link_path):
            # --output was verified empty above, and each session maps to a
            # unique relative path, so anything here is unexpected.
            raise FileExistsError(f"refusing to overwrite existing {link_path}")
        os.symlink(os.path.abspath(root), link_path)
        n += 1

    if n == 0:
        if n_task_seen == 0:
            raise SystemExit(
                f"none of the {n_sessions} sessions under {data_dir} has current_task == "
                f"{args.task!r} -- task strings must match fetch_droid_language_stats.py's "
                f"output exactly (skipped {skipped} unreadable)"
            )
        raise SystemExit(
            f"all {n_task_seen} {args.task!r} sessions are unsuccessful -- "
            "pass --include-failures to link them anyway"
        )

    print(
        f"linked {n} '{args.task}' sessions into {output}, "
        f"skipped {n_task_failed} unsuccessful and {skipped} unreadable"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
