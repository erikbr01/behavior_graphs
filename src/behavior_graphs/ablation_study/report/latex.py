"""The ablation table as LaTeX, generated rather than transcribed.

The previous table was pasted out of text reports, which is how it came to
carry a bold mark meaning "the default level" in a column a reader would read as
"the best level" -- and the default was not the best in three of the nine
blocks.

Three things changed with it.

**Best per level, not one-factor-at-a-time from a fixed default.** The old table
changed one dial of a single default configuration and left the rest fixed,
which conflates "this level is bad" with "this level is bad *alongside the other
eight defaults*". That is not a hypothetical: the two policy conditioning
encoders scored 0.184 and 0.130 that way and 0.559 and 0.535 given their own
configuration, so the table's headline lived in a paragraph underneath it. Each
cell here is instead the best that level reaches over every combination of the
other axes.

A maximum over configurations is optimistic in proportion to how many it
maximises over, so the levels are only comparable on a balanced design. The wide
sweep is not one: it pools several grid passes, ``n_components`` is nested inside
``reduction`` and ``k`` inside ``clusterer``, and ``k`` is not even the same
quantity across clusterers. So this table is built on ``ablation_droid`` instead --
the largest rectangle inside that space, verified complete and balanced on both
datasets at 36 configurations per representation, 180 in all, every level of
every axis appearing in the same number of them.

Three axes cannot appear as a result. ``n_components``, ``width`` and ``stride``
are single-valued in the core, the first because no value above 10 is a genuine
reduction for both a 14-d and a 3072-d representation, the other two because the
dinov3 and legos arms only ran width 4 and stride 4. They are ablated on the
wide grid instead, where the imbalance can be stated. HDBSCAN is absent for the
same reason it cannot join a ``k`` axis, and ``k`` itself spans only 8 to 12,
since that is all three clusterers share -- its interior optimum needs the
k-means-only range and belongs in a figure.

**Selected on one split, reported on another.** The argmax per level is taken on
the selection fold and then re-scored on the reserved episodes. Taking the
argmax on the reported set would be selecting on it.

**No density column and no bold.** Density answered a question the text answers
better, and there is no longer a default to mark.
"""

from __future__ import annotations

import math

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import core, splits
from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.scoring import Scorer
from behavior_graphs.viz.paper_figures import rep_label

#: Axis -> how it is named in the table. Shorter than the field names, and
#: escaped for LaTeX where needed.
AXIS_LABELS: dict[str, str] = {
    "layer": "representation",
    "reduction": "reduction",
    "n_components": "$n_\\text{components}$",
    "clusterer": "clusterer",
    "k": "$K$",
    "width": "window width",
    "stride": "window stride",
    "smoothing": "smoothing",
    "min_segment": "min.\\ segment",
}

#: Level -> how it is printed. Anything absent prints as-is.
LEVEL_LABELS: dict[str, str] = {
    "cuml_umap": "UMAP",
    "umap": "UMAP (CPU)",
    "pca": "PCA",
    "kmeans": "$k$-means",
    "gmm": "GMM",
    "hsmm": "HSMM",
    "hdbscan": "HDBSCAN",
    "none": "none",
    "sticky": "sticky",
}


def _level_label(axis: str, level: str) -> str:
    if axis == "layer":
        return rep_label(level)
    if axis == "min_segment":
        return f"{float(level):g}\\,s"
    return LEVEL_LABELS.get(level, level)


def _cols(n_entries: int, groups: int) -> int:
    """Rows per column group, so the groups differ by at most one row."""
    return math.ceil(n_entries / groups)


#: What a level's row carries. F1 alone hides how it was earned: at a fixed IoU
#: threshold raw F1 rises with segment density, so a row can win by predicting
#: more segments rather than better-placed ones. `density` (predicted over
#: ground-truth count, so 1.0 is the target and both directions are failures)
#: and `miou` (mean IoU of the matched segments, threshold-free) are what
#: separate those, so they are reported alongside rather than in a footnote.
ROW_METRICS = ("f1", "miou", "density")


def rows_for(
    cfg: DatasetConfig,
) -> tuple[list[tuple[str, str, dict[str, float]]], dict[str, float], dict]:
    """``(axis, level, metrics)`` per level, the per-axis spread, and caption numbers.

    *metrics* carries :data:`ROW_METRICS` -- the criterion, the mean matched IoU
    and the segment density -- every one scored on the reserved episodes.

    The spread is the sensitivity column: the range of this axis's best-per-level
    scores, so it reads as "what picking the wrong level costs once everything
    else is tuned". Derived from the numbers printed beside it rather than from a
    separate statistic, which means a reader can check it.

    Deliberately not $\\eta^2$, which is the study's other sensitivity and asks a
    different question: across the whole grid, how much of the variance in $F_1$
    does this axis account for? The two disagree, and not by a little:

        axis          eta^2   cost of changing it at the optimum
        representation 0.196  -0.104
        K              0.161  -0.246
        clusterer      0.150  -0.009
        width          0.001  -0.131
        stride         0.005  -0.172

    Width and stride explain almost none of the grid's variance yet are among
    the most expensive things to change at the peak; the clusterer explains a lot
    of it yet swapping k-means for an HSMM at the peak costs nothing. The axes
    that decide whether a configuration works *at all* are not the axes that
    matter once it already works -- a bad representation or a bad K is bad
    everywhere in the grid, while width and stride only bite in the narrow region
    where the rest is already right.

    So one column cannot carry both. This one is the local cost, because it is
    derivable from the scores printed beside it. $\\eta^2$ belongs in the prose,
    where the disagreement can be stated rather than implied by a heading.
    """
    rows = core.select(C.load(cfg, path=cfg.sweep_dir / "rescored_core.json"))
    if not rows:
        raise SystemExit(
            f"{cfg.name}: no core cells; run "
            f'`launch_sweep.sh {cfg.name} "" 3 ablation_droid` then rescore'
        )
    named = splits.resolve(cfg)
    scorer = Scorer(cfg)
    sel = cfg.evaluation.criterion

    entries: list[tuple[str, str, float]] = []
    for axis in core.ablation_axes(rows):
        for level in C.levels(rows, axis):
            here = [c for c in rows if str(c.axis(axis)) == level]
            if not here:
                continue
            # argmax on the selection fold, reported on the held-out one.
            pick = max(here, key=lambda c: c.score)
            scored = scorer.score_cell(pick, named[splits.TEST])
            entries.append(
                (
                    axis,
                    level,
                    {
                        "f1": float(scored[sel]),
                        "miou": float(scored.get("mean_matched_iou", float("nan"))),
                        "density": float(scored.get("density", float("nan"))),
                    },
                )
            )

    spread: dict[str, float] = {}
    for axis in core.ablation_axes(rows):
        vals = [m["f1"] for a, _, m in entries if a == axis]
        # One level is no choice, so it has no sensitivity -- not a spread of 0,
        # which would read as "this axis does not matter".
        spread[axis] = (max(vals) - min(vals)) if len(vals) > 1 else float("nan")

    overall = max(rows, key=lambda c: c.score)
    on_test = scorer.score_cell(overall, named[splits.TEST])
    per_job = C.per_job(rows)
    meta = {
        "n_cells": len(rows),
        "n_jobs": len(per_job),
        "n_per_rep": len(per_job) // max(1, len({c.layer for c in per_job})),
        "n_train": len(named[splits.TRAIN]),
        "n_val": len(named[splits.VAL]),
        "n_test": len(named[splits.TEST]),
        "best_test": float(on_test[sel]),
        "uniform": float(on_test[f"segmental_uniform_f1@{sel.split('@')[1]}"]),
        "noise_floor": cfg.evaluation.noise_floor,
        "widest": max(spread, key=lambda a: spread[a] if spread[a] == spread[a] else -1),
        "spread": spread,
    }
    return entries, spread, meta


def table(cfg: DatasetConfig, groups: int = 2, full_width: bool = True) -> str:
    """The table, laid out in ``groups`` side-by-side column groups.

    Side by side because the single-column form was 45 entries of two columns in
    a narrow measure -- mostly whitespace, and 53 rows tall once the
    ``\\addlinespace`` gaps between blocks were counted.

    Measured with pdflatex in a twocolumn article, where a column is 230pt and
    the text 469pt, with the five-digit counts the $n$ column really carries:

        groups=1  251pt  36 rows  22pt over a column
        groups=2  455pt  19 rows  fits the full text width
        groups=3  682pt  13 rows  213pt over even spanning both columns

    So two groups in a ``table*``, the only layout that holds all five columns.
    Each column added closed an option: $\\Delta$ ruled out three groups, and
    $n$ ruled out the single-column fallback, which fitted at 216pt before it.
    If this has to sit in one column, drop $n$ and say in the caption that the
    per-level counts are uneven -- that is what $n$ is there to show.
    """
    entries, spread, meta = rows_for(cfg)
    per = _cols(len(entries), groups)
    chunks = [entries[i * per : (i + 1) * per] for i in range(groups)]

    # Five columns a group now, not four: IoU and density are what say whether
    # an F1 was earned by placement or by segment count, so they travel with it.
    spec = "@{}" + "@{\\hskip 1.0em}".join(["llrrr"] * groups) + "@{}"
    head = " & ".join(["stage & option & $F_1$ & IoU & dens."] * groups) + " \\\\"

    body: list[str] = []
    for r in range(per):
        cells: list[str] = []
        for chunk in chunks:
            if r >= len(chunk):
                cells += ["", "", "", "", ""]
                continue
            axis, level, m = chunk[r]
            # The axis name and its spread print once per run, so a block reads
            # as a block without needing \addlinespace between them. A block
            # split across two column groups prints its name again, which is
            # what a reader needs when the run restarts mid-table.
            first = r == 0 or chunk[r - 1][0] != axis
            cells += [
                AXIS_LABELS.get(axis, axis) if first else "",
                _level_label(axis, level),
                f"{m['f1']:.3f}",
                "--" if m["miou"] != m["miou"] else f"{m['miou']:.3f}",
                "--" if m["density"] != m["density"] else f"{m['density']:.2f}",
            ]
        body.append(" & ".join(cells) + " \\\\")

    # Three sentences: what a cell is, what the extra column is, and the two
    # reference points a reader needs to size the numbers. Everything else that
    # was in here belongs in the text or in the grid config, not under a table.
    caption = (
        f"{cfg.name.capitalize()}: best segmentation $F_1$ per option over a balanced design of "
        f"{meta['n_per_rep']} configurations per representation, chosen on "
        f"{meta['n_val']} selection episodes and scored on {meta['n_test']} reserved ones. "
        f"IoU is the mean overlap of the matched segments and is threshold-free; "
        f"density is predicted segments over ground-truth segments, so 1.0 is the "
        f"target and a larger number is over-segmentation rather than a better "
        f"score. The best configuration scores "
        f"{meta['best_test']:.3f} against {meta['uniform']:.3f} for a count-matched uniform "
        f"segmenter; differences below {meta['noise_floor']:.3f} are within the across-fold "
        f"standard error."
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
            f"\\label{{tab:ablation-{cfg.name}}}",
            f"\\end{{{env}}}",
        ]
    )


def run(cfg: DatasetConfig, **_) -> str:
    return table(cfg)
