"""One task's ablation as a representation x reduction table, paper-ready.

:mod:`behavior_graphs.ablation_study.report.latex` lays a task out as one row per
option of every axis, which is the right shape for "which dial matters". This is
the other shape: the two *reported* axes crossed, so a reader can see that a
reduction's effect flips between representations rather than reading a pooled
reduction row that averages the two directions away. On the cooking collection
those two readings genuinely disagree -- PCA takes proprioception and UMAP takes
the policy tap -- and the crossed table is the only one that shows it.

**Three columns, and the third is why.** $F_1$ at a fixed IoU threshold rises
with segment density, so a row can win by predicting more segments rather than
better-placed ones. ``density`` (predicted over ground-truth count) says whether
that happened, and ``miou`` (mean overlap of the matched segments,
threshold-free) says whether the boundaries landed. Density has no better
direction -- 1.0 is the target and both 0.3 and 3.0 are failures -- so it is
listed in :data:`behavior_graphs.viz.latex_table.NOT_RANKABLE` and never bolded.

**The K floor is a modelling choice, not a filter.** A task whose ground truth
averages ~10.9 segments an episode cannot be represented by fewer than ~11
clusters, so cells below that answer a different question and their scores are
about how a too-coarse partition happens to land. Pooling them produced a
reduction effect on cooking that reversed once they were dropped, which is why
the floor is a named argument that ends up in the caption rather than something
applied silently upstream.

Reads ``rescored.json`` through :func:`behavior_graphs.ablation_study.cells.load`, so
what this prints and what ``study.py report`` prints cannot disagree -- and a
sweep whose rescore is stale shows stale numbers rather than recomputing behind
the caller's back.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from collections.abc import Sequence
from typing import Any

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.viz.latex_table import booktabs_table
from behavior_graphs.viz.paper_figures import rep_label

#: Reduction levels in reporting order, coarsest first. ``identity`` leads
#: because "no reduction" is the baseline the other four have to beat, not a
#: variant of them.
REDUCTION_ORDER = ("identity", "pca", "cuml_umap", "umap")


def reduction_label(reduction: str, n_components: Any) -> str:
    """``pca`` + 10 -> ``PCA 10``; identity has no width to report."""
    if reduction == "identity":
        return "none"
    name = {"pca": "PCA", "cuml_umap": "UMAP", "umap": "UMAP"}.get(reduction, reduction)
    return f"{name} {n_components}" if n_components else name


def _sort_key(label: str) -> tuple[int, int]:
    order = {"none": 0, "PCA": 1, "UMAP": 2}
    head = label.split()[0]
    width = int(label.split()[1]) if len(label.split()) > 1 else 0
    return order.get(head, 9), width


def _silhouette(job_dir: str) -> float:
    """The cell's silhouette, from its own metrics.json.

    Not in `rescored.json`, which carries the boundary metrics only -- the
    re-score recomputes those from cached labels and has no reason to touch an
    unsupervised score. Read per winning row rather than for every cell, so a
    nine-row table costs nine small reads. NaN when absent, which prints as
    `--` rather than as a zero.
    """
    try:
        raw = json.loads((pathlib.Path(job_dir) / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError):
        return float("nan")
    value = (raw.get("cluster_quality") or {}).get("silhouette")
    return float(value) if value is not None else float("nan")


def rows(
    cfg: DatasetConfig,
    *,
    k_min: int | None = None,
    k_allow: Sequence[int] | None = None,
    representations: Sequence[str] | None = None,
    n_components: int | None = None,
) -> list[dict[str, Any]]:
    """Best cell per (representation, reduction), with its IoU and density.

    The best *by the criterion*, and its IoU and density are that same cell's --
    not each column's own maximum, which would describe a configuration that
    does not exist.

    *representations* restricts the arms, by layer name, and keeps the order
    given: a table of three arms should not imply the sweep ran three.

    *n_components* pins the component count rather than maximising over it.
    That is a balance requirement, not a preference -- identity takes no
    component count, so folding UMAP@10 and UMAP@100 into one "UMAP" level would
    search UMAP and PCA over twice as many cells as identity, and a
    best-per-level table over unequal search spaces favours whichever level was
    searched hardest. Pinning leaves every level the same 96 cells on cooking.

    *k_allow* pins the $K$ values rather than flooring them, which is what a
    table spanning several tasks needs: each task declares its own floor, so a
    floor leaves the three searched over different $K$ ranges and their cells
    stop being the same grid. Passing the shared set makes the design identical
    across tasks, at the price of ignoring a task's own floor -- `k_min` is the
    single-task reading, `k_allow` the cross-task one. They are exclusive.
    """
    if k_allow is not None and k_min:
        raise ValueError("pass k_min or k_allow, not both: they disagree about K by design")
    # None means "whatever this task declares", which is the point of having
    # declared it -- a caller that has to remember the floor will forget it.
    if k_min is None and k_allow is None:
        k_min = int(cfg.evaluation.k_floor or 0)
    loaded = list(C.load(cfg))
    if k_allow is not None:
        allowed = {int(k) for k in k_allow}
        loaded = [c for c in loaded if (c.k or 0) in allowed]
    elif k_min:
        loaded = [c for c in loaded if (c.k or 0) >= k_min]
    if representations is not None:
        wanted = list(representations)
        loaded = [c for c in loaded if c.layer in wanted]
    if n_components is not None:
        # identity has no component count and must survive the pin, or the
        # baseline the other two are measured against disappears.
        loaded = [
            c
            for c in loaded
            if c.reduction == "identity" or str(c.n_components) == str(n_components)
        ]
    # `min_segment` is swept inside the evaluator rather than being a dial, so
    # every cell appears once per value; the design's own cells are the
    # unfiltered ones and the rest would count each configuration five times.
    loaded = [c for c in loaded if str(c.min_segment) in ("0.0", "0", "None", "")]

    grouped: dict[tuple[str, str], list[C.Cell]] = {}
    for cell in loaded:
        key = (cell.layer, reduction_label(cell.reduction, cell.n_components))
        grouped.setdefault(key, []).append(cell)

    out = []
    for (layer, reduction), group in grouped.items():
        best = max(group, key=lambda c: c.score)
        # Flat: `cells.load` has already reduced each record to its primary
        # metric dict, so there is no task/primary level to walk here.
        primary = best.metrics or {}
        out.append(
            {
                "representation": rep_label(layer),
                "layer": layer,
                "reduction": reduction,
                "F1": float(best.score),
                "IoU": float(primary.get("mean_matched_iou", float("nan"))),
                "density": float(best.density),
                "silhouette": _silhouette(best.dir),
                "K": best.k_label(),
                "median_F1": statistics.median(c.score for c in group),
                "n_cells": len(group),
            }
        )
    out.sort(key=lambda r: (r["representation"], _sort_key(r["reduction"])))
    return out


def default_caption(cfg: DatasetConfig, table_rows: list[dict[str, Any]], k_min: int) -> str:
    """What a reader needs and cannot recover from the numbers."""
    tau = cfg.evaluation.criterion.split("@")[-1]
    arms = len({r["representation"] for r in table_rows})
    per_cell = min((r["n_cells"] for r in table_rows), default=0)
    floor = (
        f" $K$ is floored at {k_min}, since a partition into fewer clusters cannot "
        "represent the segment count the ground truth carries."
        if k_min
        else ""
    )
    return (
        f"Subtask-boundary ablation on {cfg.name}, {arms} representations against "
        f"{len(REDUCTION_ORDER) and len({r['reduction'] for r in table_rows})} reduction "
        f"settings. Each cell is the best segmental $F_1$ at IoU {tau} the option reaches "
        f"over an identical nuisance grid of window width, stride, $K$ and smoothing "
        f"({per_cell} cells per row).{floor} IoU is the mean overlap of the matched "
        f"segments and is threshold-free, so it says whether the boundaries landed; "
        f"density is predicted segments over ground-truth segments, where "
        f"\\emph{{1.0}} is the target and a larger value is over-segmentation rather than "
        f"a better score. IoU and density are the winning cell's own, not each "
        f"column's maximum. Silhouette is unsupervised and reported for contrast "
        f"only: it anti-correlates with $F_1$ against hand annotation on seven of "
        f"eight annotated tasks, and is measured in each reduction's own space, so "
        f"it is comparable neither to the other columns nor down a block. "
        f"$K$, window width, stride and smoothing are searched "
        f"identically for every row and not reported: the best $K$ follows the "
        f"task's own segment count rather than the design."
    )


def table(
    cfg: DatasetConfig,
    *,
    k_min: int | None = None,
    representations: Sequence[str] | None = None,
    n_components: int | None = None,
    caption: str | None = None,
    label: str | None = None,
    compact: bool = True,
) -> str:
    """The crossed table as booktabs LaTeX, grouped by representation."""
    if k_min is None:
        k_min = int(cfg.evaluation.k_floor or 0)
    table_rows = rows(cfg, k_min=k_min, representations=representations, n_components=n_components)
    if not table_rows:
        return (
            f"% no cells for {cfg.name}"
            + (f" at K >= {k_min}" if k_min else "")
            + " -- run `study.py rescore --dataset "
            + f"{cfg.name}` after sweeping\n"
        )

    import pandas as pd

    # No K column: it is searched identically for every row and then left out,
    # because the best K follows the task's segment count rather than the
    # design. In a table of design choices it would read as a recommendation.
    frame = pd.DataFrame(table_rows)[
        ["representation", "reduction", "F1", "IoU", "density", "silhouette"]
    ]
    return booktabs_table(
        frame,
        # density is a measurement, so it earns a score column's decimals;
        # NOT_RANKABLE is what keeps bold_best off it.
        score_columns=("F1", "IoU", "density", "silhouette"),
        group_column="representation",
        caption=caption or default_caption(cfg, table_rows, k_min),
        label=label or f"tab:ablation-{cfg.name}",
        bold_best=True,
        compact=compact,
    )


def run(cfg: DatasetConfig, **kwargs) -> str:
    """Text form for `study.py report --which per_task`."""
    k_min = int(kwargs.get("k_min") or cfg.evaluation.k_floor or 0)
    table_rows = rows(cfg, k_min=k_min)
    if not table_rows:
        return f"{cfg.name}: no cells{f' at K >= {k_min}' if k_min else ''}"

    lines = [
        f"{cfg.name}: best cell per (representation, reduction)"
        + (f", K >= {k_min}" if k_min else ""),
        f"criterion {cfg.evaluation.criterion}  |  density target 1.0",
        "",
        f"  {'representation':28s} {'reduction':10s} {'F1':>6} {'IoU':>6} "
        f"{'dens':>6} {'K':>4} {'med F1':>7} {'n':>5}",
    ]
    last = None
    for r in table_rows:
        name = r["representation"] if r["representation"] != last else ""
        last = r["representation"]

        def num(v: float, places: int = 3) -> str:
            return f"{'--':>6}" if v != v else f"{v:>6.{places}f}"

        lines.append(
            f"  {name:28s} {r['reduction']:10s} {num(r['F1'])} {num(r['IoU'])} "
            f"{num(r['density'], 2)} {r['K']:>4} {r['median_F1']:>7.3f} {r['n_cells']:>5}"
        )
    return "\n".join(lines)
