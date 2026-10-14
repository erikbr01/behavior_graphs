"""Reproduce the ablation table in the paper -- those numbers and no others.

The paper reports exactly twenty-four numbers from this sweep: nine cells of
representation x reduction, each a mean and standard deviation over the three
DROID tasks, plus a mean row per representation marginalised over the three
reductions. The prose quotes three of them again ($F_1$ of 0.226, 0.132, 0.113)
and bolds two cells as the per-metric maxima. This script computes those and
stops. It deliberately does not print density, silhouette, the uniform-null
floors, per-task cells, winning $K$ values or best-cell links: all of those were
part of deciding what to report, none of them are in the paper, and a
reproduction script that prints more than the paper makes the reader guess which
numbers mattered.

**Why the numbers are a maximum, and why that is only fair if the grid is
shared.** A cell is the best segmental $F_1$ its option reaches over the
identical nuisance grid of window width, stride, $K$ and smoothing -- so a
cell with more cells behind it wins on search budget rather than merit. Two
pins enforce the shared grid: `k_allow` fixes $K$ to {5, 10, 15} for all three
tasks (a per-task `k_floor` would give them different ranges) and
`n_components=10` keeps UMAP and PCA from being searched over component counts
that `identity` has no analogue for. The script then *checks* the result rather
than trusting it: if any row group has a different cell count from the others,
the comparison is a maximum over unequal searches and it refuses to print.

Selection runs through `report.per_task.rows`, the same path the app's Figures
page and `study.py report` use, so this cannot drift from what the website
shows. It reads `rescored.json` and recomputes nothing -- a stale rescore
reports stale numbers rather than silently disagreeing with the sweep.

Usage::

    python scripts/ablation_study/paper_ablation.py              # print and verify
    python scripts/ablation_study/paper_ablation.py --latex      # the table body
    python scripts/ablation_study/paper_ablation.py --tolerance 0.0005

Exits non-zero if any number has drifted from the published value, so it works
as a regression test on the sweep artefacts.
"""

from __future__ import annotations

import argparse
import statistics
import sys

from behavior_graphs.ablation_study import config as CFG
from behavior_graphs.ablation_study.report import per_task

#: The three DROID deployments the user study drew its baseline policies from,
#: in the order the paper's columns average over. Each has gripper-derived
#: ground truth; see `build_gripper_gt.py` for what that proxy does and does
#: not measure.
TASKS = ("settable", "preparetray", "cooking")

#: Layer names, in the paper's row order.
REPRESENTATIONS = ("state", "cond_proj", "dinov3")

#: Row labels as the paper prints them, which are not the codebase's
#: `rep_label` strings ("Policy conditioning", "DINOv3 (frozen)"). Kept here so
#: the LaTeX this emits can be pasted over Tab. I without a rename, and so a
#: reader comparing the two sees the same words.
PAPER_LABEL = {
    "state": "Proprioception",
    "cond_proj": "Policy embeddings",
    "dinov3": "DINOv3",
}

#: Reduction labels as `per_task.reduction_label` emits them. `identity`
#: becomes "none" and is the baseline the other two have to beat.
REDUCTIONS = ("none", "PCA 10", "UMAP 10")

#: $K$ values shared by all three tasks. Pinned rather than floored -- see the
#: module docstring.
K_ALLOW = (5, 10, 15)

#: Component count pinned for the reductions that take one.
N_COMPONENTS = 10

#: What the paper prints, keyed by (layer, reduction) with "mean" for the
#: marginal row, as (F1, F1 s.d., IoU, IoU s.d.). Transcribed from
#: `ieee.tex`'s Tab. I rather than generated, so that a change in the sweep
#: shows up as a mismatch here instead of quietly rewriting both sides.
PAPER: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("state", "none"): (0.093, 0.075, 0.523, 0.061),
    ("state", "PCA 10"): (0.120, 0.121, 0.547, 0.098),
    ("state", "UMAP 10"): (0.182, 0.022, 0.598, 0.033),
    ("state", "mean"): (0.132, 0.072, 0.556, 0.063),
    ("cond_proj", "none"): (0.264, 0.108, 0.633, 0.048),
    ("cond_proj", "PCA 10"): (0.254, 0.121, 0.644, 0.059),
    ("cond_proj", "UMAP 10"): (0.160, 0.031, 0.567, 0.022),
    ("cond_proj", "mean"): (0.226, 0.086, 0.615, 0.032),
    ("dinov3", "none"): (0.088, 0.030, 0.462, 0.029),
    ("dinov3", "PCA 10"): (0.082, 0.023, 0.462, 0.022),
    ("dinov3", "UMAP 10"): (0.169, 0.086, 0.566, 0.088),
    ("dinov3", "mean"): (0.113, 0.019, 0.497, 0.014),
}

#: The two cells the paper bolds, and the prose claim that both are policy
#: rows. Checked, not assumed: bolding a maximum that is no longer the maximum
#: is the one error a reader cannot catch.
PAPER_BOLD = {"F1": ("cond_proj", "none"), "IoU": ("cond_proj", "PCA 10")}


def per_task_cells(task: str) -> dict[tuple[str, str], dict[str, float]]:
    """One task's nine cells, keyed by (layer, reduction label).

    Raises if the task's row groups were searched over different numbers of
    cells, or if any of the nine is missing -- both make the cross-task mean a
    different quantity from the published one, and a partial table that looks
    plausible is worse than no table.
    """
    cfg = CFG.load(task)
    rows = per_task.rows(
        cfg,
        k_allow=K_ALLOW,
        representations=REPRESENTATIONS,
        n_components=N_COMPONENTS,
    )
    cells = {(r["layer"], r["reduction"]): r for r in rows}

    missing = [(a, r) for a in REPRESENTATIONS for r in REDUCTIONS if (a, r) not in cells]
    if missing:
        raise SystemExit(
            f"{task}: missing {len(missing)} of 9 cells: {missing}\n"
            f"  sweep `configs/grid/ablation_droid.yaml` for this task, then "
            f"`python scripts/ablation_study/study.py rescore --dataset {task}`"
        )

    counts = {cells[(a, r)]["n_cells"] for a in REPRESENTATIONS for r in REDUCTIONS}
    if len(counts) != 1:
        raise SystemExit(
            f"{task}: row groups hold {sorted(counts)} cells, so the best-per-row "
            f"table is a maximum over unequal searches. Refusing to report it."
        )
    return cells


def aggregate() -> tuple[dict[tuple[str, str], tuple[float, float, float, float]], int]:
    """The paper's twenty-four numbers, and the per-row cell count behind them.

    Means run over the levels of the collapsed axis *within* a task and only
    then across tasks, so a mean row is the mean of the cells printed above it
    and the s.d. is the spread across the three tasks -- not across cells,
    which would mix two sources of variation.
    """
    by_task = {task: per_task_cells(task) for task in TASKS}
    budget = {c["n_cells"] for cells in by_task.values() for c in cells.values()}

    out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    for arm in REPRESENTATIONS:
        for red in REDUCTIONS:
            f1 = [by_task[t][(arm, red)]["F1"] for t in TASKS]
            iou = [by_task[t][(arm, red)]["IoU"] for t in TASKS]
            out[(arm, red)] = (
                statistics.fmean(f1),
                statistics.stdev(f1),
                statistics.fmean(iou),
                statistics.stdev(iou),
            )
        # The marginal: collapse reductions inside each task, then across tasks.
        f1 = [statistics.fmean(by_task[t][(arm, r)]["F1"] for r in REDUCTIONS) for t in TASKS]
        iou = [statistics.fmean(by_task[t][(arm, r)]["IoU"] for r in REDUCTIONS) for t in TASKS]
        out[(arm, "mean")] = (
            statistics.fmean(f1),
            statistics.stdev(f1),
            statistics.fmean(iou),
            statistics.stdev(iou),
        )
    return out, min(budget)


def maxima(got: dict[tuple[str, str], tuple[float, float, float, float]]) -> dict[str, tuple]:
    """Which of the nine cells leads each metric; the mean rows do not compete."""
    cells = [k for k in got if k[1] != "mean"]
    return {
        "F1": max(cells, key=lambda k: got[k][0]),
        "IoU": max(cells, key=lambda k: got[k][2]),
    }


def report(got, budget: int, tolerance: float) -> int:
    """Print the table the paper prints, flagging any value that has drifted."""
    bad = 0
    print(f"{'representation':18s} {'reduction':9s} {'F1(0.75)':>16s} {'mean IoU':>16s}")
    for arm in REPRESENTATIONS:
        label = PAPER_LABEL[arm]
        for red in (*REDUCTIONS, "mean"):
            f1, f1s, iou, ious = got[(arm, red)]
            want = PAPER[(arm, red)]
            flags = [
                "" if abs(value - target) <= tolerance else f" != {target:.3f}"
                for value, target in zip((f1, f1s, iou, ious), want, strict=True)
            ]
            bad += sum(1 for f in flags if f)
            print(
                f"{label:18s} {red:9s} "
                f"{f1:>8.3f} ± {f1s:.3f}{flags[0]}{flags[1]} "
                f"{iou:>8.3f} ± {ious:.3f}{flags[2]}{flags[3]}"
            )
            label = ""
    print(f"\n{budget} cells behind every row, identical across the three tasks.")

    for metric, cell in maxima(got).items():
        want = PAPER_BOLD[metric]
        mark = "" if cell == want else f"  != paper, which bolds {want}"
        print(f"{metric} maximum: {PAPER_LABEL[cell[0]]} / {cell[1]}{mark}")
        bad += bool(mark)
    return bad


def latex(got) -> None:
    """The table body, in the paper's own markup, for pasting over Tab. I."""
    for arm in REPRESENTATIONS:
        # Four rows a block, counting the mean. `ieee.tex` currently says 3,
        # which centres the label against the first three only.
        print(f"\\multirow{{{len(REDUCTIONS) + 1}}}{{*}}{{{PAPER_LABEL[arm]}}}")
        for red in (*REDUCTIONS, "mean"):
            f1, f1s, iou, ious = got[(arm, red)]
            name = "\\emph{mean}" if red == "mean" else red.split()[0]
            cells = []
            for value, sd, metric in ((f1, f1s, "F1"), (iou, ious, "IoU")):
                body = f"{value:.3f}"
                if (arm, red) == PAPER_BOLD[metric]:
                    body = f"\\mathbf{{{body}}}"
                cells.append(f"${body} \\pm {sd:.3f}$")
            print(f"  & {name} & {cells[0]} & {cells[1]} \\\\")
        if arm != REPRESENTATIONS[-1]:
            print("\\midrule")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latex", action="store_true", help="also emit the table body as LaTeX")
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.0005,
        # The paper prints three decimals, so anything inside half of the last
        # place rounds to the same string and is not a drift.
        help="largest difference from the published value that still agrees (default: 0.0005)",
    )
    args = ap.parse_args()

    got, budget = aggregate()
    bad = report(got, budget, args.tolerance)
    if args.latex:
        print()
        latex(got)
    if bad:
        print(f"\n{bad} value(s) differ from the published table.", file=sys.stderr)
        return 1
    print("\nAll 24 published values reproduce, and both bolded maxima still hold.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
