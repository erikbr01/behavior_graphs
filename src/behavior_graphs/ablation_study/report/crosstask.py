"""Findings averaged over tasks, with the per-task numbers beside them.

A section written around two tasks says "on bottles the DiT layer wins, on legos
the joint angles do" and leaves a reader with an anecdote. What generalises is
the average and the spread across tasks: whether an axis matters *anywhere*, and
whether its best level is stable. So this aggregates first and lists per task
second, and it takes however many datasets have a scored core rather than a
fixed pair.

Two aggregates, because they answer different questions:

``mean``
    The average of a level's best $F_1$ over tasks. What you would expect from
    that level on a new task.
``range``
    Max minus min over tasks. Large means the option's value is task-specific, so
    a recipe fixing it will not transfer -- which is the interesting failure and
    is invisible in any single-task table.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence

import numpy as np

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import config as study_config
from behavior_graphs.ablation_study import core, splits
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.scoring import Scorer

CORE_ARTEFACT = "rescored_core.json"

#: Stages the LaTeX table omits. Both stay in the design, in the sweep and in
#: the eta^2 ranking the caption reports -- they are simply not worth a block of
#: rows a reader has to scan past. `min_segment` is a post-hoc merge applied
#: after clustering rather than a graph-construction choice, and `smoothing` is
#: binary, so its contribution is already stated as one eta^2 figure.
TABLE_OMIT = ("smoothing", "min_segment")

#: IoU thresholds for the strictness table. The scorer also emits 0.10, which
#: is dropped: a prediction overlapping a tenth of a segment counts as a hit
#: there, so every option passes and the column carries no signal.
STRICTNESS = (0.25, 0.50, 0.75)


def _numeric(v: str) -> bool:
    try:
        float(v)
    except ValueError:
        return False
    return True


def datasets() -> list[str]:
    """Datasets with a scored core, in config order."""
    out = []
    for name in study_config.available():
        cfg = study_config.load(name)
        if (cfg.sweep_dir / CORE_ARTEFACT).exists():
            out.append(name)
    return out


def per_task(
    name: str,
) -> tuple[
    dict[tuple[str, str], float],
    dict[str, float],
    dict,
    dict[tuple[str, str], dict[float, float]],
]:
    """``(axis, level) -> test F1``, ``axis -> eta^2``, summary, and per-threshold F1.

    The fourth value re-runs the argmax *per threshold*: for each option and
    each IoU, the configuration selected is the one that scores best at that
    IoU on the selection split, then re-scored on the reporting split at the
    same IoU.

    Selecting once at 0.5 and re-scoring elsewhere would answer a different
    question -- how the 0.5-optimal recipe happens to fare under a stricter
    test -- and would bias every strict column downward for any option whose
    best-at-0.5 configuration is not its best-at-0.75 one. This table is a
    best-per-option table, so each cell has to be the best that option can do
    at that threshold. Cells across a row already come from different
    configurations, exactly as they do down a column.
    """
    cfg: DatasetConfig = study_config.load(name)
    rows = core.select(C.load(cfg, path=cfg.sweep_dir / CORE_ARTEFACT))
    named = splits.resolve(cfg)
    scorer = Scorer(cfg)
    sel = cfg.evaluation.criterion

    # One test-set scoring per distinct configuration, however many thresholds
    # select it: score_cell returns every threshold at once and is the
    # expensive call here.
    on_test_cache: dict[str, dict] = {}

    def on_test(cell: C.Cell) -> dict:
        if cell.dir not in on_test_cache:
            on_test_cache[cell.dir] = scorer.score_cell(cell, named[splits.TEST])
        return on_test_cache[cell.dir]

    scores: dict[tuple[str, str], float] = {}
    by_threshold: dict[tuple[str, str], dict[float, float]] = {}
    for axis in core.ablation_axes(rows):
        for level in C.levels(rows, axis):
            here = [c for c in rows if str(c.axis(axis)) == level]
            if not here:
                continue
            scores[(axis, level)] = float(on_test(max(here, key=lambda c: c.score))[sel])

            at: dict[float, float] = {}
            for t in STRICTNESS:
                key = f"segmental_f1@{t:g}"
                # Selection uses the *selection split*, which is what
                # `cell.metrics` holds; the reported value comes from the
                # reporting split. Never select on the split being reported.
                eligible = [c for c in here if key in c.metrics]
                if not eligible:
                    continue
                pick = max(eligible, key=lambda c: c.metrics[key])
                scored = on_test(pick)
                if key in scored:
                    at[t] = float(scored[key])
            by_threshold[(axis, level)] = at

    per_job = C.per_job(rows)
    eta = {
        a: C.eta_squared(list(rows if a == "min_segment" else per_job), a)
        for a in core.ablation_axes(rows)
    }

    best = max(rows, key=lambda c: c.score)
    on_test = scorer.score_cell(best, named[splits.TEST])
    summary = {
        "additive_r2": additive_r2(rows),
        "best": float(on_test[sel]),
        "uniform": float(on_test[f"segmental_uniform_f1@{sel.split('@')[1]}"]),
        "n_test": len(named[splits.TEST]),
        "n_val": len(named[splits.VAL]),
        "n_per_rep": len(per_job) // max(1, len({c.layer for c in per_job})),
    }
    return scores, eta, summary, by_threshold


def additive_r2(rows: Sequence[C.Cell]) -> float:
    """Variance explained by an additive (main-effects-only) model of the axes.

    Complements :math:`\\eta^2`, which asks how much *one* axis explains on its
    own. This asks how much *all* of them explain together assuming they do not
    interact, so ``1 - additive_r2`` is the share of the score that lives in
    combinations of levels rather than in any level by itself. A recipe is only
    transferable as a list of independent choices to the extent this is high.

    One-hot over every core axis plus min_segment, least squares, no intercept
    term of its own (the constant column carries it). Fitted on the same
    selection-split scores as ``eta_squared``, so the two are comparable.
    """
    axes = core.ablation_axes(rows)
    y = np.asarray([c.score for c in rows], dtype=float)
    ok = np.isfinite(y)
    y, rows = y[ok], [c for c, keep in zip(rows, ok, strict=True) if keep]
    if len(rows) < 2:
        return float("nan")

    cols = [np.ones(len(rows))]
    for axis in axes:
        levels = sorted({str(c.axis(axis)) for c in rows})
        # Drop one level per axis: with all of them the design matrix is
        # collinear with the constant column and the fit is not identifiable.
        for level in levels[1:]:
            cols.append(np.asarray([float(str(c.axis(axis)) == level) for c in rows]))
    x = np.stack(cols, axis=1)

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sst = float(((y - y.mean()) ** 2).sum())
    if sst <= 0:
        return float("nan")
    return float(1.0 - (resid**2).sum() / sst)


def run(cfg: DatasetConfig | None = None, **_) -> str:
    """Cross-task summary. ``cfg`` is ignored: this spans every dataset."""
    names = datasets()
    if len(names) < 2:
        return f"only {len(names)} dataset(s) with a scored core; nothing to average"

    scores, etas, summaries, _thr = {}, {}, {}, {}
    for name in names:
        scores[name], etas[name], summaries[name], _thr[name] = per_task(name)

    # Axes in the study's display order, keeping only those some task scored.
    seen = {a for n in names for (a, _) in scores[n]}
    axes = [a for a in C.AXES if a in seen]

    w = max(len(n) for n in names)
    out = [
        f"{len(names)} tasks: {', '.join(names)}",
        "",
        "per-task headline",
        f"  {'task':<{w}} {'best':>7} {'uniform':>8} {'head':>7} {'cfgs/rep':>9} "
        f"{'test eps':>9} {'add. R2':>8}",
    ]
    for name in names:
        s = summaries[name]
        out.append(
            f"  {name:<{w}} {s['best']:>7.3f} {s['uniform']:>8.3f} "
            f"{s['best'] - s['uniform']:>+7.3f} {s['n_per_rep']:>9} {s['n_test']:>9} "
            f"{s['additive_r2']:>8.3f}"
        )

    out += ["", "eta^2 per stage, averaged over tasks (high = this stage decides the score)"]
    out.append(f"  {'stage':<14} {'mean':>7} {'min':>7} {'max':>7}")
    for a in sorted(axes, key=lambda a: -statistics.fmean(etas[n].get(a, 0.0) for n in names)):
        vals = [etas[n].get(a, float("nan")) for n in names]
        out.append(f"  {a:<14} {statistics.fmean(vals):>7.3f} {min(vals):>7.3f} {max(vals):>7.3f}")

    out += [
        "",
        "best F1 per option, averaged over tasks",
        "  `range` is max minus min across tasks: large means the option's value is",
        "  task-specific, so a recipe that fixes it will not transfer.",
        f"  {'stage':<14} {'option':<20} {'mean':>7} {'range':>7}  "
        + " ".join(f"{n[:8]:>8}" for n in names),
    ]
    for a in axes:
        # Numeric levels sort numerically; K=10 must not follow K=1.
        levels = sorted(
            {lv for n in names for (ax, lv) in scores[n] if ax == a},
            key=lambda v: (0, float(v)) if _numeric(v) else (1, v),
        )
        for lv in levels:
            vals = [scores[n].get((a, lv)) for n in names]
            got = [v for v in vals if v is not None]
            if not got:
                continue
            out.append(
                f"  {a:<14} {lv:<20} {statistics.fmean(got):>7.3f} "
                f"{max(got) - min(got):>7.3f}  "
                + " ".join(f"{'-' if v is None else f'{v:.3f}':>8}" for v in vals)
            )
    return "\n".join(out)


def table(full_width: bool = True) -> str:
    """The cross-task table as LaTeX: mean and range first, per task after.

    One table for every task rather than one per task. That is both more compact
    and the right emphasis: a per-task table invites a reader to read off one
    task's winner, which is exactly the anecdote the average exists to replace.
    """
    from behavior_graphs.ablation_study.report.latex import AXIS_LABELS, _level_label

    names = datasets()
    if len(names) < 2:
        raise SystemExit(f"only {len(names)} scored core(s); need at least two to average")

    scores, etas, summaries, _thr = {}, {}, {}, {}
    for name in names:
        scores[name], etas[name], summaries[name], _thr[name] = per_task(name)
    seen = {a for n in names for (a, _) in scores[n]}
    axes = [a for a in C.AXES if a in seen and a not in TABLE_OMIT]

    # "stage"/"option" rather than the ANOVA "factor"/"level": these rows are
    # pipeline stages and the settings they were given, and a reader should not
    # need the vocabulary of experimental design to parse the first two columns.
    #
    # Mean and standard deviation over datasets, not mean and range: with eight
    # tasks a range is set by whichever two are extreme, so it says less than a
    # spread using all of them. Per-task columns are gone -- eight would not
    # fit, and inviting a reader to read off one task's winner is the anecdote
    # the average exists to replace.
    #
    # eta^2 is a property of the *stage*, not of an option. As a column it was
    # empty on every row but the first of each block, which reads as missing
    # data rather than as "this does not vary by option". It now heads its own
    # spanning row, which also gives the blocks a visible break.
    spec = "@{}l rr@{}"
    head = "option & $F_1$ mean & $F_1$ std \\\\"

    body = []
    for a in axes:
        levels = sorted(
            {lv for n in names for (ax, lv) in scores[n] if ax == a},
            key=lambda v: (0, float(v)) if _numeric(v) else (1, v),
        )
        eta_mean = statistics.fmean([etas[n].get(a, 0.0) for n in names])
        rows = []
        for lv in levels:
            got = [v for n in names if (v := scores[n].get((a, lv))) is not None]
            if not got:
                continue
            rows.append(
                " & ".join(
                    [
                        f"\\quad {_level_label(a, lv)}",
                        f"{statistics.fmean(got):.3f}",
                        f"{statistics.stdev(got):.3f}" if len(got) > 1 else "--",
                    ]
                )
                + " \\\\"
            )
        if not rows:
            continue
        if body:
            body.append("\\addlinespace")
        body.append(
            f"\\multicolumn{{3}}{{@{{}}l@{{}}}}{{\\textit{{{AXIS_LABELS.get(a, a)}}}"
            f" \\quad $\\eta^2 = {eta_mean:.3f}$}} \\\\"
        )
        body.extend(rows)

    eta_line = ", ".join(
        f"{AXIS_LABELS.get(a, a)} {statistics.fmean([etas[n].get(a, 0.0) for n in names]):.3f}"
        for a in sorted(
            axes, key=lambda a: -statistics.fmean([etas[n].get(a, 0.0) for n in names])
        )[:3]
    )
    # Summarised, not enumerated: one "x against y" per task reads fine for two
    # and is an unreadable list of eight.
    gains = [summaries[n]["best"] - summaries[n]["uniform"] for n in names]
    heads = (
        f"{statistics.fmean([summaries[n]['best'] for n in names]):.3f} on average "
        f"against {statistics.fmean([summaries[n]['uniform'] for n in names]):.3f}, "
        f"a margin of {statistics.fmean(gains):+.3f} "
        f"(min {min(gains):+.3f}, max {max(gains):+.3f}),"
    )
    caption = (
        f"Best segmentation $F_1(0.5)$ per option, over {len(names)} tasks on a "
        f"balanced design of {summaries[names[0]]['n_per_rep']} configurations per "
        f"representation each. Mean and standard deviation are over tasks, so a "
        f"large std means the option's value is task-specific and a recipe that "
        f"fixes it will not transfer. $\\eta^2$ is the share of grid-wide variance "
        f"the whole stage explains, averaged over tasks; it is a property of the "
        f"stage rather than of any one option, so it heads each block, and "
        f"it ranks them {eta_line}. Best configurations score {heads} for a "
        f"count-matched uniform segmenter emitting each configuration's own "
        f"segment count."
    )

    env = "table*" if full_width else "table"
    return "\n".join(
        [
            f"\\begin{{{env}}}[t]",
            "\\centering\\small",
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
            head,
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\label{tab:ablation}",
            f"\\end{{{env}}}",
        ]
    )


def table_strictness(full_width: bool = True) -> str:
    """The same rows, but columns are IoU thresholds rather than tasks.

    A companion to :func:`table`, not a replacement. That one asks whether an
    option's value transfers between tasks; this one asks whether it survives a
    stricter definition of a correct segment. Per-task columns are dropped
    because both questions in one table needs tasks x thresholds columns, which
    does not fit and does not read.

    Each cell re-runs the argmax at its own threshold, so it reports what the
    option can achieve there rather than how its 0.5-optimal recipe happens to
    degrade.
    """
    from behavior_graphs.ablation_study.report.latex import AXIS_LABELS, _level_label

    names = datasets()
    if not names:
        raise SystemExit("no scored cores")

    thr: dict[str, dict[tuple[str, str], dict[float, float]]] = {}
    for name in names:
        *_, thr[name] = per_task(name)

    seen = {a for n in names for (a, _) in thr[n]}
    axes = [a for a in C.AXES if a in seen and a not in TABLE_OMIT]

    # Mean first, then the breakdown: it is the summary a reader wants before
    # the per-threshold detail.
    spec = "@{}ll r@{\\hskip 1.2em}" + " ".join(["r"] * len(STRICTNESS)) + "@{}"
    head = "stage & option & mean & " + " & ".join(f"$F_1({t:g})$" for t in STRICTNESS) + " \\\\"

    body = []
    for a in axes:
        levels = sorted(
            {lv for n in names for (ax, lv) in thr[n] if ax == a},
            key=lambda v: (0, float(v)) if _numeric(v) else (1, v),
        )
        for i, lv in enumerate(levels):
            per_t = []
            for t in STRICTNESS:
                vals = [
                    thr[n][(a, lv)][t] for n in names if (a, lv) in thr[n] and t in thr[n][(a, lv)]
                ]
                per_t.append(statistics.fmean(vals) if vals else None)
            got = [v for v in per_t if v is not None]
            cells = [
                AXIS_LABELS.get(a, a) if i == 0 else "",
                _level_label(a, lv),
                f"{statistics.fmean(got):.3f}" if got else "--",
                *[("--" if v is None else f"{v:.3f}") for v in per_t],
            ]
            body.append(" & ".join(cells) + " \\\\")

    env = "table*" if full_width else "table"
    caption = (
        f"Segmentation $F_1$ at three IoU thresholds, averaged over "
        f"{len(names)} task{'s' if len(names) != 1 else ''}. "
        f"Each cell is that option's best achievable $F_1$ at that threshold: "
        f"the configuration is selected at the same IoU it is reported at, on "
        f"the selection split, and scored on the reporting split. Cells are "
        f"therefore not all the same configuration, which is what makes each "
        f"one a fair statement about its own option. ``mean'' averages the "
        f"three thresholds, as COCO's mAP does, and inherits from that the "
        f"fact that each threshold was optimised separately."
    )
    return "\n".join(
        [
            f"\\begin{{{env}}}[t]",
            "\\centering\\small",
            f"\\begin{{tabular}}{{{spec}}}",
            "\\toprule",
            head,
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            "\\label{tab:ablation-strictness}",
            f"\\end{{{env}}}",
        ]
    )
