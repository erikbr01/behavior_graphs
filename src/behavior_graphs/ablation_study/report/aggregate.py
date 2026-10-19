"""The ablation pooled across datasets: one table instead of one table per task.

:mod:`behavior_graphs.ablation_study.report.latex` writes a table per dataset, which is
the right unit while a task is being understood and the wrong one in a paper with
eight of them -- eight ``table*`` floats say "here is a lot of data" where the
claim is "these orderings hold across tasks".

So this pools the same rows. For every (stage, option) it reports the mean of the
per-dataset held-out $F_1$, the spread across datasets, and how many datasets that
option was the best of its axis on -- which is the part a mean hides. An option
that wins narrowly everywhere and an option that wins hugely on one task and
loses on seven can have the same mean, and only the win count separates them.

**Why a cache.** ``rows_for`` takes the argmax per level on the selection fold and
then re-scores it on the reserved episodes, which is a pipeline scoring run per
level -- minutes per dataset, and the point of selecting on one fold and
reporting on another is lost if it is skipped. Eight of those is far past what a
Streamlit page can do on a click, so the per-dataset rows are computed once by
``study.py cache_rows`` and written beside the sweep as ``ablation_rows.json``.
:func:`pooled` reads only caches, names the datasets that have none, and never
computes -- so a page built on it is instant and can say what is missing rather
than hanging.

The cache records the criterion and the split sizes it was built with, and
:func:`pooled` refuses to mix rows from two different criteria: a table averaging
``segmental_f1@0.75`` on some tasks and ``segmental_f1@0.50`` on others would be
a number with no definition.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import statistics
from typing import Any

from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.report import latex
from behavior_graphs.viz.latex_table import booktabs_table, escape

#: Bumped when the cached shape changes, so a stale file is refused rather than
#: read as if it meant something.
#: 2: entries carry {f1, miou, density} instead of a bare F1. A v1 file
#: would be read as if its float were the whole metric set.
CACHE_VERSION = 2

CACHE_NAME = "ablation_rows.json"

#: Pooled for every level, in reporting order. `f1` is the criterion and the
#: only one with a better direction; `miou` is the threshold-free mean overlap
#: of matched segments; `density` is predicted over ground-truth segment count,
#: where 1.0 is the target and both 0.3 and 3.0 are failures. Keeping all three
#: is what stops an F1 bought with extra segments reading as a better method.
METRICS = ("f1", "miou", "density")


#: Axes a balanced grid may vary. An axis that turns out single-valued is
#: dropped automatically: it is not a choice, and would print as a row with no
#: alternative beside it.
#:
#: ``reduction`` here is the *setting*, not the kind -- reduction crossed with
#: its component count, so the levels read none / PCA 10 / PCA 100 / UMAP 10 /
#: UMAP 100. Two reasons. It is balanced: `identity` takes no `n_components`, so
#: as separate axes the kind column had 960/480/960 cells and a best-per-level
#: maximum would favour the two levels searched twice as hard. And it is the
#: honest reading: `n_components` is nested inside `reduction` rather than
#: orthogonal to it, and 10 components means a different operation on a 14-d
#: proprioception vector than on 6144-d DINOv3, and nothing at all on identity.
BALANCED_AXES = (
    "layer",
    "reduction",
    "clusterer",
    "k",
    "width",
    "stride",
    "smoothing",
)


def _level(cell, axis: str) -> str:
    """The level *cell* sits at on *axis*, with reduction crossed by width."""
    if axis == "reduction":
        from behavior_graphs.ablation_study.report import per_task

        return per_task.reduction_label(cell.reduction, cell.n_components)
    return str(cell.axis(axis))


def balanced_entries(cfg: DatasetConfig, *, k_min: int | None = None):
    """``(axis, level, metrics)`` for a full-cross-product grid that is not ablation_droid.

    The same contract as :func:`latex.rows_for` -- argmax per level on the
    selection fold, re-scored on the reserved episodes -- but the design comes
    from the cells instead of a grid file, so it works for any dataset whose
    sweep was a complete cross product.

    Balance is checked, not assumed. For a cross product every level of an axis
    appears in the same number of cells; if it does not, this raises with the
    counts rather than reporting a maximum taken over unequal search spaces,
    which is the failure the balanced design exists to prevent.
    """
    import collections

    from behavior_graphs.ablation_study import cells as C
    from behavior_graphs.ablation_study import splits
    from behavior_graphs.ablation_study.scoring import Scorer

    if k_min is None:
        k_min = int(cfg.evaluation.k_floor or 0)
    rows = [c for c in C.load(cfg) if str(c.min_segment) in ("0.0", "0", "None", "")]
    rows = [c for c in rows if not k_min or (c.k or 0) >= k_min]
    if not rows:
        raise SystemExit(f"{cfg.name}: no cells at K >= {k_min}; sweep and rescore first")

    varying = []
    for axis in BALANCED_AXES:
        levels = {_level(c, axis) for c in rows}
        if len(levels) > 1:
            varying.append(axis)

    for axis in varying:
        counts = collections.Counter(_level(c, axis) for c in rows)
        if len(set(counts.values())) != 1:
            raise SystemExit(
                f"{cfg.name}: axis {axis!r} is not balanced -- "
                + ", ".join(f"{lv}:{n}" for lv, n in sorted(counts.items()))
                + ". A best-per-level table over unequal search spaces favours "
                "whichever level was searched hardest, so it is refused."
            )

    named = splits.resolve(cfg)
    scorer = Scorer(cfg)
    criterion = cfg.evaluation.criterion

    entries = []
    for axis in varying:
        for level in sorted({_level(c, axis) for c in rows}):
            here = [c for c in rows if _level(c, axis) == level]
            pick = max(here, key=lambda c: c.score)
            scored = scorer.score_cell(pick, named[splits.TEST])
            entries.append(
                (
                    axis,
                    level,
                    {
                        "f1": float(scored[criterion]),
                        "miou": float(scored.get("mean_matched_iou", float("nan"))),
                        "density": float(scored.get("density", float("nan"))),
                    },
                )
            )

    spread = {}
    for axis in varying:
        vals = [m["f1"] for a, _, m in entries if a == axis]
        spread[axis] = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")

    best = max(rows, key=lambda c: c.score)
    on_test = scorer.score_cell(best, named[splits.TEST])
    meta = {
        "n_cells": len(rows),
        "n_per_rep": len(rows) // max(1, len({c.layer for c in rows})),
        "n_train": len(named[splits.TRAIN]),
        "n_val": len(named[splits.VAL]),
        "n_test": len(named[splits.TEST]),
        "best_test": float(on_test[criterion]),
        "uniform": float(on_test.get(f"segmental_uniform_f1@{criterion.split('@')[1]}", 0.0)),
        "noise_floor": cfg.evaluation.noise_floor,
        "k_floor": k_min,
        "spread": spread,
        "source": "balanced_grid",
    }
    return entries, spread, meta


def cache_path(cfg: DatasetConfig) -> pathlib.Path:
    return cfg.sweep_dir / CACHE_NAME


def write_cache(cfg: DatasetConfig) -> pathlib.Path:
    """Compute this dataset's held-out ablation rows and cache them.

    The slow call, run deliberately from the CLI. Everything downstream reads
    the file.
    """
    # ablation_droid when the dataset ran it, otherwise the dataset's own grid --
    # verified balanced rather than assumed. A task swept on `ablation*`
    # has no `rescored_core.json` and would otherwise be uncacheable.
    if (cfg.sweep_dir / "rescored_core.json").is_file():
        entries, spread, meta = latex.rows_for(cfg)
    else:
        entries, spread, meta = balanced_entries(cfg)
    payload = {
        "cache_version": CACHE_VERSION,
        "dataset": cfg.name,
        "task": cfg.task,
        "criterion": cfg.evaluation.criterion,
        "written_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "entries": [{"axis": a, "level": lv, **m} for a, lv, m in entries],
        "spread": spread,
        "meta": meta,
    }
    path = cache_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_cache(cfg: DatasetConfig) -> dict[str, Any] | None:
    """The cached rows, or None when absent or written by an older shape."""
    path = cache_path(cfg)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if int(payload.get("cache_version", 0)) != CACHE_VERSION:
        return None
    return payload


class Pooled:
    """The aggregated rows, plus what they were built from.

    A small class rather than a tuple because every consumer -- the text report,
    the LaTeX table, the app -- wants the provenance next to the numbers, and a
    table whose caption cannot say which datasets it covers is not quotable.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]],
        datasets: list[str],
        missing: list[str],
        criterion: str,
    ) -> None:
        self.rows = rows
        self.datasets = datasets
        self.missing = missing
        self.criterion = criterion

    @property
    def axes(self) -> list[str]:
        seen: list[str] = []
        for row in self.rows:
            if row["axis"] not in seen:
                seen.append(row["axis"])
        return seen

    def spread_of(self, axis: str) -> float:
        """Mean across datasets of the within-axis range -- the cost of choosing wrong.

        Averaged from each dataset's own spread rather than taken over the pooled
        means, because the mean of the maxima is not the maximum of the means and
        the quantity wanted is "how much does this axis cost on a typical task".
        """
        values = [
            row["spread"][axis]
            for row in self._per_dataset
            if axis in row["spread"] and row["spread"][axis] == row["spread"][axis]
        ]
        return statistics.fmean(values) if values else float("nan")


def pooled(cfgs: list[DatasetConfig]) -> Pooled:
    """Aggregate every cached dataset among *cfgs*. Computes nothing."""
    payloads = []
    missing = []
    for cfg in cfgs:
        payload = read_cache(cfg)
        (payloads.append(payload) if payload else missing.append(cfg.name))

    if not payloads:
        return Pooled([], [], missing, "")

    criteria = {p["criterion"] for p in payloads}
    if len(criteria) > 1:
        raise ValueError(
            "cached rows disagree on the criterion "
            f"({', '.join(sorted(criteria))}); a pooled mean over two "
            "definitions of F1 has no definition. Re-cache the odd ones out."
        )
    criterion = criteria.pop()

    # (axis, level) -> {dataset: f1}, keeping the axis and level order the
    # per-dataset tables use so the pooled table reads like they do.
    order: list[tuple[str, str]] = []
    # (axis, level) -> metric -> dataset -> value. Three levels rather than two
    # because every metric is pooled across the same datasets and a reader needs
    # them side by side: an F1 is not interpretable without the density that
    # produced it.
    scores: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for payload in payloads:
        for entry in payload["entries"]:
            key = (entry["axis"], entry["level"])
            if key not in scores:
                scores[key] = {m: {} for m in METRICS}
                order.append(key)
            for metric in METRICS:
                value = entry.get(metric)
                if value is not None and value == value:
                    scores[key][metric][payload["dataset"]] = float(value)

    # Best level per (dataset, axis), which is what the win count counts.
    wins: dict[tuple[str, str], int] = dict.fromkeys(order, 0)
    for payload in payloads:
        by_axis: dict[str, list[tuple[str, float]]] = {}
        for entry in payload["entries"]:
            by_axis.setdefault(entry["axis"], []).append(
                (entry["level"], float(entry.get("f1") or 0.0))
            )
        for axis, levels in by_axis.items():
            # Won on the criterion, not on IoU or density: density has no better
            # direction at all (1.0 is the target), so "best" is only defined
            # for F1.
            best_level = max(levels, key=lambda p: p[1])[0]
            wins[(axis, best_level)] = wins.get((axis, best_level), 0) + 1

    rows = []
    for axis, level in order:
        per = scores[(axis, level)]["f1"]
        values = list(per.values())
        row = {
            "axis": axis,
            "level": level,
        }
        for metric in METRICS:
            vals = list(scores[(axis, level)][metric].values())
            row[metric] = statistics.fmean(vals) if vals else float("nan")
            row[f"{metric}_sd"] = statistics.pstdev(vals) if len(vals) > 1 else float("nan")
        rows.append(
            {
                **row,
                "mean": statistics.fmean(values),
                # Population SD over the datasets present, not a standard error:
                # these are eight particular tasks, not a sample from a
                # population of tasks, so the spread is descriptive.
                "sd": statistics.pstdev(values) if len(values) > 1 else float("nan"),
                "min": min(values),
                "max": max(values),
                "n": len(values),
                "wins": wins.get((axis, level), 0),
                "per_dataset": per,
            }
        )

    out = Pooled(rows, [p["dataset"] for p in payloads], missing, criterion)
    out._per_dataset = payloads  # noqa: SLF001 -- spread_of reads it
    return out


def _num(value: float, width: int = 6, places: int = 3) -> str:
    """*value* right-aligned, or ``--`` when it is NaN.

    A metric absent from a cache is not zero, and printing `nan` invites a
    reader to treat it as one.
    """
    return f"{'--':>{width}}" if value != value else f"{value:>{width}.{places}f}"


def dataframe_rows(agg: Pooled) -> list[dict[str, Any]]:
    """The pooled rows in the shape a table wants, one row per level.

    ``std_F1`` rather than ``SD``: :func:`booktabs_table` collapses a
    ``mean_x``/``std_x`` pair into one ``x`` cell, and naming the column its way
    is what buys that for free. ``best on`` is a fraction and deliberately not a
    score column -- it is a count, and bolding the option scored on the most
    tasks would be nonsense.
    """
    return [
        {
            "stage": latex.AXIS_LABELS.get(r["axis"], r["axis"]),
            "option": latex._level_label(r["axis"], r["level"]),  # noqa: SLF001
            "F1": r["f1"],
            "std_F1": r["f1_sd"],
            "IoU": r["miou"],
            "density": r["density"],
            "best on": f"{r['wins']}/{r['n']}",
        }
        for r in agg.rows
    ]


def table(
    agg: Pooled,
    *,
    caption: str | None = None,
    label: str | None = None,
    compact: bool = True,
) -> str:
    """The pooled ablation as a booktabs table.

    One column group, unlike the per-dataset table: pooling eight tables into
    one collapses the row count enough that four columns fit a single column of
    a twocolumn layout, so this needs no ``table*``. ``group_column`` turns the
    stage into ``\\multirow`` blocks with a rule between them, which is the
    layout the per-dataset table hand-rolls.

    No bolding, for the reason the per-dataset table gives: there is no default
    level to mark, and a bold mark in an $F_1$ column reads as "the best" when
    the honest answer is the ``best on`` count beside it -- an option can hold
    the highest mean while winning on two tasks of eight.
    """
    if not agg.rows:
        return "% no cached ablation rows -- run `study.py cache_rows --dataset <name>`\n"

    import pandas as pd

    frame = pd.DataFrame(dataframe_rows(agg))[
        ["stage", "option", "F1", "std_F1", "IoU", "density", "best on"]
    ]
    return booktabs_table(
        frame,
        # `density` is a score column so it gets a measurement's decimals, and
        # `latex_table.NOT_RANKABLE` keeps `bold_best` off it regardless: its
        # target is 1.0, so neither direction is "best".
        score_columns=("F1", "std_F1", "IoU", "density"),
        group_column="stage",
        caption=caption or default_caption(agg),
        label=label or "tab:ablation-pooled",
        compact=compact,
        bold_best=False,
    )


def default_caption(agg: Pooled) -> str:
    """The caption the export panel offers, and what `table` falls back to.

    A function rather than a literal inside :func:`table` because the shared
    LaTeX export panel puts the caption in an editable box, and a default it
    cannot reach is a default nobody uses.
    """
    tau = agg.criterion.split("@")[-1]
    return (
        f"Ablation pooled over {len(agg.datasets)} tasks "
        f"({', '.join(escape(d) for d in sorted(agg.datasets))}). Each cell is the mean "
        f"over tasks of the best segmentation $F_1$ that option reaches at IoU {tau}, "
        f"chosen on each task's selection episodes and scored on its reserved ones, "
        f"$\\pm$ the spread across tasks. That spread is descriptive, not a standard "
        f"error: these are particular tasks, not a sample from a population of them. "
        f"IoU is the mean overlap of the matched segments, which is threshold-free and so "
        f"says whether an $F_1$ was earned by placement; density is predicted segments over "
        f"ground-truth segments, where \\emph{{1.0}} is the target and a larger value is "
        f"over-segmentation rather than a better score --- at a fixed threshold $F_1$ rises "
        f"with density, so the three columns are read together. "
        f"``best on'' counts the tasks where the option was the best of its stage on $F_1$, which "
        f"the mean hides --- an option winning narrowly everywhere and one winning "
        f"hugely on a single task can share a mean."
    )


def run(cfg: DatasetConfig, **_) -> str:
    """Text form, for `study.py report --which aggregate`.

    Takes a cfg like every other report so the dispatch table stays uniform, and
    ignores it beyond using its directory layout to find the sibling caches --
    the table is about every dataset, not this one.
    """
    from behavior_graphs.ablation_study import config as C

    cfgs = [C.load(name) for name in C.available()]
    agg = pooled(cfgs)
    if not agg.rows:
        return (
            "no cached ablation rows anywhere. Run\n"
            "  python scripts/ablation_study/study.py cache_rows --dataset <name>\n"
            f"for each of: {', '.join(C.available())}"
        )

    lines = [
        f"pooled over {len(agg.datasets)} datasets: {', '.join(sorted(agg.datasets))}",
        f"criterion {agg.criterion}",
    ]
    if agg.missing:
        lines.append(f"NOT cached, so absent from every number below: {', '.join(agg.missing)}")
    lines += [
        "",
        f"  {'stage':16s} {'option':22s} {'F1':>6} {'SD':>6} {'IoU':>6} {'dens':>6} {'best on':>8}",
        # The density header carries its target, because a column of numbers
        # around 2.0 reads as a score unless something says 1.0 is the aim.
        f"  {'':16s} {'':22s} {'':6s} {'':6s} {'':6s} {'(->1.0)':>6}",
    ]
    last = None
    for row in agg.rows:
        stage = latex.AXIS_LABELS.get(row["axis"], row["axis"]) if row["axis"] != last else ""
        last = row["axis"]
        lines.append(
            f"  {stage:16s} {latex._level_label(row['axis'], row['level'])[:22]:22s} "
            f"{_num(row['f1'])} {_num(row['f1_sd'])} {_num(row['miou'])} "
            f"{_num(row['density'], places=2)} {row['wins']:>4}/{row['n']}"
        )
    lines += ["", "mean cost of choosing the wrong level, per axis:"]
    for axis in agg.axes:
        value = agg.spread_of(axis)
        shown = "--" if value != value else f"{value:.3f}"
        lines.append(f"  {latex.AXIS_LABELS.get(axis, axis):16s} {shown}")
    return "\n".join(lines)
