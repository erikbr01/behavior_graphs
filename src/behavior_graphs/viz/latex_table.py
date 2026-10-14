"""DataFrame → a ``booktabs`` table you can paste straight into a paper.

An on-screen table and a paper table want different things. The screen wants
every column the sweep produced, four decimals, and ``mean_roc_auc`` /
``std_roc_auc`` side by side; a paper wants three decimals, the two collapsed
into one ``$0.812 \\pm 0.041$`` cell, the winning method bolded, and no method
name smuggling an underscore into LaTeX as a subscript.

This module is that translation and nothing else -- it neither writes files nor
imports Streamlit -- so a page offering the export stays a download button and
an :func:`streamlit.code` call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

#: Metrics where the *smallest* number is the best one, so that ``bold_best``
#: bolds the minimum rather than the maximum. Matched against a column's base
#: name, i.e. after any ``mean_``/``std_`` prefix is stripped, so
#: ``mean_calibrated_brier`` lands here via ``calibrated_brier`` -- hence the
#: prefixed spellings alongside the bare ones.
#: Headers that a compact table sets shorter, keyed case-folded on the base
#: metric name. Only where the short form is the one a paper would print
#: anyway: tau IS Kendall's statistic and AUC is how AUROC is usually written,
#: whereas VOC stays VOC because that is what the work being compared against
#: calls it. Measured on the progress-estimation rubric table, these three
#: headers alone are 30 pt of an IEEE column's 252 -- the headers were wider
#: than the numbers under them.
COMPACT_HEADERS = {
    "kendall": r"$\tau$",
    "auroc": "AUC",
    "pearson": "$r$",
    "spearman": r"$\rho$",
}

#: Inter-column padding a compact table asks for, in points. LaTeX's default is
#: 6 pt, which on a six-column table is 72 pt of pure air.
COMPACT_TABCOLSEP = 4

LOWER_IS_BETTER = frozenset(
    {
        "brier",
        "ece",
        "mce",
        "nll",
        "log_loss",
        "mae",
        "mse",
        "rmse",
        "calibrated_brier",
        "calibrated_ece",
    }
)

#: Metrics with no better direction at all, so ``bold_best`` must skip them
#: however it is configured. They still get a score column's decimals -- they
#: are measurements -- but marking one "best" is a category error rather than a
#: wrong choice of direction.
#:
#: ``density`` is the reason this exists. It is predicted segments over
#: ground-truth segments, so the good value is **1.0** and both 0.3 and 3.0 are
#: failures -- a collapse and an over-segmentation. Neither
#: :data:`LOWER_IS_BETTER` nor its complement expresses that, and bolding the
#: largest density reads as "this row segmented best" when it means the
#: opposite. It happened in a table handed over in review.
#:
#: Keyed on the base metric name, like :data:`LOWER_IS_BETTER`, so
#: ``mean_density`` lands here too.
NOT_RANKABLE = frozenset(
    {
        "density",
        "segment_density",
        "n_pred",
        "n_gt",
        "n_cells",
        "n_tasks",
        # Silhouette has a direction -- higher is more separated -- and is
        # still never the best. It anti-correlates with segmentation F1
        # against hand annotation on seven of eight ABC tasks (-0.17 to
        # -0.71; see probes/silhouette_validity.py), so bolding its maximum
        # asserts the opposite of what was measured. It is also computed in
        # the *reduced* space, so a UMAP cell's and a PCA cell's are
        # different geometries and not comparable down a column.
        "silhouette",
    }
)

#: Tokens a header should shout rather than merely capitalise: ``roc_auc``
#: reads as "ROC AUC", not "Roc auc".
_ACRONYMS = {
    "auc": "AUC",
    "roc": "ROC",
    "pr": "PR",
    "ap": "AP",
    "mi": "MI",
    "ece": "ECE",
    # The progress-estimation metrics, which are acronyms every one: without
    # these, `header_label` sets them "Voc", "Mae" and "Auroc" -- wrong in a
    # paper, and wrong in a way only a reader of the PDF would catch.
    "voc": "VOC",
    "mae": "MAE",
    "mse": "MSE",
    "rmse": "RMSE",
    "auroc": "AUROC",
    "nll": "NLL",
    "mce": "MCE",
    "ari": "ARI",
    "iou": "IoU",
    "f1": "F1",
    "mlp": "MLP",
    "lstm": "LSTM",
    "knn": "kNN",
    "tp": "TP",
    "tn": "TN",
    "fp": "FP",
    "fn": "FN",
    "n": "N",
}

#: Applied as a single translation pass, not as successive replacements: every
#: replacement introduces a backslash and half of them introduce braces, so a
#: second pass would escape the escapes.
_ESCAPES = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)

#: What an undefined metric prints as. A dash rather than ``nan`` or a blank: a
#: paper reader should see "not defined here", and a blank cell reads as an
#: oversight.
MISSING = "--"


def escape(text: str) -> str:
    """LaTeX-escape ``text``, so that ``graph_probe`` stays ``graph_probe``."""
    return text.translate(_ESCAPES)


def base_metric(column: str) -> str:
    """``column`` without its ``mean_``/``std_`` prefix.

    The bit that names *what was measured*, which is what
    :data:`LOWER_IS_BETTER` is keyed on and what a merged mean±std column is
    headed with.
    """
    for prefix in ("mean_", "std_"):
        if column.startswith(prefix):
            return column[len(prefix) :]
    return column


def header_label(column: str) -> str:
    """``mean_roc_auc`` → ``Mean ROC AUC``: underscores out, acronyms up."""
    words = [_ACRONYMS.get(word, word) for word in column.split("_") if word]
    if not words:
        return ""
    first, *rest = words
    if first not in _ACRONYMS.values():
        first = first[:1].upper() + first[1:]
    return escape(" ".join([first, *rest]))


@dataclass(frozen=True)
class _Column:
    """One emitted column: its header, its alignment, and where its cells come from."""

    header: str
    align: str
    #: The frame columns feeding it: ``(mean, std)`` for a merged pair, one
    #: column otherwise.
    keys: tuple[str, ...]
    #: The frame column whose values decide who wins, or ``None`` for a column
    #: nothing can win: identity, context, a denominator, or a spread.
    score_key: str | None


def _is_number(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _fmt_number(value: Any, precision: int) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return MISSING
    return f"{float(value):.{precision}f}"


def _cells(series: pd.Series, precision: int, *, is_score: bool) -> list[str]:
    """``series`` as cell strings: scores to ``precision``, counts as counts."""
    if not _is_number(series):
        return [
            MISSING if value is None or value != value else escape(str(value)) for value in series
        ]
    # A count stays a count: `n_datasets` is 6, never 6.000, and neither is a
    # whole-numbered score like `auc_wins`. An integer dtype settles it; a
    # float column of whole numbers is a count that a NaN somewhere turned
    # float, but only if it is not a score -- a metric that happens to be 1.0
    # on every dataset is still `1.000`.
    if pd.api.types.is_integer_dtype(series):
        return [str(int(value)) for value in series]
    if not is_score:
        defined = series.dropna()
        if len(defined) and all(float(value).is_integer() for value in defined):
            return [MISSING if value != value else str(int(value)) for value in series]
    return [_fmt_number(value, precision) for value in series]


def _best_rows(series: pd.Series, lower_is_better: Sequence[str]) -> set[int]:
    """Row positions holding ``series``'s best value -- all of them, on a tie."""
    numeric = pd.to_numeric(series, errors="coerce")
    defined = numeric.dropna()
    if len(defined) < 2 or defined.nunique() < 2:
        return set()  # nothing to win: one row scored, or every row equal
    # Case-folded: `LOWER_IS_BETTER` is keyed on the metric names as the code
    # spells them (lower case, because they are dict keys and filename stems),
    # but a caller may have renamed a column to the acronym a table should SET
    # -- `MAE` -- and matching case-sensitively then silently ranks a loss
    # upside-down, bolding the worst cell as the winner.
    lower = {name.casefold() for name in lower_is_better}
    best = defined.min() if base_metric(str(series.name)).casefold() in lower else defined.max()
    return {position for position, value in enumerate(numeric) if value == best}


def _groups(series: pd.Series) -> list[tuple[int, int]]:
    """``[(start, length)]`` for each run of equal consecutive values.

    Runs rather than a groupby: the caller's row order is the table's row
    order, and a groupby would silently reorder the blocks (or merge two
    non-adjacent ones into a \\multirow spanning rows that are not adjacent,
    which is not a thing LaTeX can typeset).
    """
    runs: list[tuple[int, int]] = []
    for position, value in enumerate(series):
        if runs and series.iloc[runs[-1][0]] == value:
            start, length = runs[-1]
            runs[-1] = (start, length + 1)
        else:
            runs.append((position, 1))
    return runs


def _mergeable(frame: pd.DataFrame, scores: set[str], *, merge_std: bool) -> dict[str, str]:
    """``mean_x`` -> ``std_x``, for every pair that collapses into one cell.

    A pass of its own because column order is the caller's: nothing says a
    mean precedes its spread, and folding as the loop went would emit a
    leading ``std_x`` both as a column and inside the merged cell.
    """
    if not merge_std:
        return {}
    return {
        name: f"std_{base_metric(name)}"
        for name in frame.columns
        if name.startswith("mean_")
        and name in scores
        and f"std_{base_metric(name)}" in scores
        and f"std_{base_metric(name)}" in frame.columns
    }


def _header(column: str, *, compact: bool) -> str:
    """The column's header, short if a compact table has a short form for it."""
    if compact:
        short = COMPACT_HEADERS.get(base_metric(column).casefold())
        if short is not None:
            return short
    return header_label(column)


def _plan(
    frame: pd.DataFrame, scores: set[str], *, merge_std: bool, compact: bool = False
) -> list[_Column]:
    """The emitted columns, in the frame's own column order.

    ``mean_x``/``std_x`` collapse into a single ``x`` column when both were
    selected as scores and ``merge_std`` is on. A ``std_x`` whose mean is
    absent -- or whose pair was left unmerged -- stays a column of its own,
    since dropping it would silently lose a number the caller asked to see.
    What it never is, is *ranked*: a spread has no better direction, and
    bolding its maximum would advertise the least reproducible method as the
    winner of something.
    """
    merged = _mergeable(frame, scores, merge_std=merge_std)
    folded = set(merged.values())

    columns: list[_Column] = []
    for name in frame.columns:
        if name in folded:
            continue
        if name in merged:
            columns.append(
                _Column(
                    _header(base_metric(name), compact=compact),
                    "r",
                    (name, merged[name]),
                    name,
                )
            )
            continue
        align = "r" if _is_number(frame[name]) else "l"
        rankable = (
            name in scores
            and not name.startswith("std_")
            and base_metric(name).casefold() not in NOT_RANKABLE
        )
        columns.append(
            _Column(_header(name, compact=compact), align, (name,), name if rankable else None)
        )
    return columns


def booktabs_table(
    frame: pd.DataFrame,
    *,
    score_columns: Sequence[str] = (),
    caption: str | None = None,
    label: str | None = None,
    precision: int = 3,
    merge_std: bool = True,
    bold_best: bool = False,
    lower_is_better: Sequence[str] = tuple(LOWER_IS_BETTER),
    group_column: str | None = None,
    compact: bool = False,
    strip_leading_zero: bool = False,
    font_size: str | None = None,
    abbreviate_headers: bool = True,
) -> str:
    """``frame`` as a ``booktabs`` table, ready to paste into a paper.

    ``score_columns`` names the columns that are *measurements* rather than
    identity or denominators. Only they get ``precision`` decimals, get
    collapsed into ``mean ± std`` pairs, and are eligible for ``bold_best`` --
    which is why the caller passes them rather than this function guessing:
    ``n_datasets`` is a number too, and bolding the method that was scored on
    the most datasets would be nonsense. A ``std_`` column is the exception
    that proves the rule: it is a measurement, so it gets the decimals and the
    merge, but a spread is not a thing to win and so is never bolded.

    ``group_column`` turns the named column into ``\\multirow`` blocks: one
    spanning cell per run of equal consecutive values, a ``\\midrule`` between
    runs, and ``bold_best`` then picking a winner WITHIN each block rather than
    down the whole column -- which is the only reading that makes sense once a
    column spans several groups, since the best cell overall would be comparing
    the groups. Needs ``\\usepackage{multirow}`` as well, and the emitted
    source says so.

    ``compact`` is for a table that has to fit a single column: it trims the
    outer ``\\tabcolsep`` with ``@{}``, narrows the rest to
    :data:`COMPACT_TABCOLSEP`, and sets the headers in :data:`COMPACT_HEADERS`
    short. Measured on the progress-estimation rubric table under ``IEEEtran``
    against a 252 pt column: 323 pt as-is, 265 pt compact — the difference
    between overflowing and nearly fitting, with no column dropped and no type
    shrunk.

    ``abbreviate_headers`` is what applies :data:`COMPACT_HEADERS` under
    ``compact``; turn it off to keep the full words. Worth about 20 pt of a
    252 pt column on the rubric table -- the header really is what sets those
    columns' width, since ``Kendall`` is wider than the ``0.930`` under it --
    so full headers need one step more of ``font_size``: measured, they fit at
    ``footnotesize`` (243 pt) and do not at ``small`` (269 pt).

    ``font_size`` is a LaTeX size command without the backslash (``small``,
    ``footnotesize``) applied inside the float, which is how most IEEE tables
    are set anyway. On the rubric table it is worth about 23 pt of a 252 pt
    column at ``small`` -- more than any other knob here, and the one that
    costs no information at all.

    ``strip_leading_zero`` writes ``.945`` for ``0.945``. Correct for a metric
    bounded in [-1, 1] and wrong for one that can reach 1000, so it is a knob
    of its own rather than part of ``compact`` — and the weakest of them: a
    column is as wide as its header or its cells, whichever wins, so once
    ``compact`` has shortened the headers this buys a couple of points. Reach
    for it only when the table is that close.

    The result is a bare ``tabular`` unless a ``caption`` or ``label`` is
    given, in which case it is wrapped in a floating ``table``. Requires
    ``\\usepackage{booktabs}`` either way.
    """
    scores = set(score_columns)
    columns = _plan(frame, scores, merge_std=merge_std, compact=compact and abbreviate_headers)
    if not columns:
        return ""

    rendered = {
        name: _cells(frame[name], precision, is_score=name in scores) for name in frame.columns
    }
    if strip_leading_zero:
        # `.945` rather than `0.945`: correct for a metric bounded in
        # [-1, 1], wrong for one that can reach 1000 -- hence a knob of
        # its own. Only the score columns: a count keeps its digits.
        def _strip_zero(cell: str) -> str:
            # `-0.136` too, or a negative cell would keep a zero its neighbours
            # dropped and the column would look misaligned rather than compact.
            if cell.startswith("0."):
                return cell[1:]
            if cell.startswith("-0."):
                return "-" + cell[2:]
            return cell

        rendered = {
            name: [_strip_zero(cell) if name in scores else cell for cell in cells]
            for name, cells in rendered.items()
        }
    runs = _groups(frame[group_column]) if group_column else [(0, len(frame))]
    best: dict[str, set[int]] = {}
    if bold_best:
        for column in columns:
            if not column.score_key:
                continue
            winners: set[int] = set()
            for start, length in runs:
                block = frame[column.score_key].iloc[start : start + length]
                winners |= {start + position for position in _best_rows(block, lower_is_better)}
            best[column.score_key] = winners

    # Where a \multirow starts, and how tall it is.
    spans = {start: length for start, length in runs} if group_column else {}
    body: list[list[str]] = []
    for row in range(len(frame)):
        cells: list[str] = []
        for column in columns:
            if group_column and column.keys[0] == group_column:
                # One spanning cell per block; the rest of the block's rows
                # leave it empty rather than repeating the label.
                if row not in spans:
                    cells.append("")
                elif spans[row] == 1:
                    # `\multirow{1}` is a no-op wrapper: a one-row block is
                    # just a label, and a plain cell keeps the source legible.
                    cells.append(rendered[group_column][row])
                else:
                    cells.append(
                        rf"\multirow{{{spans[row]}}}{{*}}{{{rendered[group_column][row]}}}"
                    )
                continue
            value = rendered[column.keys[0]][row]
            wins = value != MISSING and row in best.get(column.score_key, set())
            if len(column.keys) == 2:
                spread = rendered[column.keys[1]][row]
                mean = rf"\mathbf{{{value}}}" if wins else value
                # An undefined mean makes the whole cell undefined -- a lone
                # spread around a missing centre says nothing. A defined mean
                # with no spread (one dataset) is just the mean.
                if value == MISSING:
                    cells.append(MISSING)
                else:
                    cells.append(f"${mean} \\pm {spread}$" if spread != MISSING else f"${mean}$")
            else:
                cells.append(rf"\textbf{{{value}}}" if wins else value)
        body.append(cells)

    # Pad to a common width per column so the emitted source is legible in an
    # editor -- LaTeX itself does not care.
    headers = [column.header for column in columns]
    grid = [headers, *body]
    widths = [max(len(row[i]) for row in grid) for i in range(len(headers))]
    rows = [
        " & ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True)).rstrip()
        + r" \\"
        for cells in grid
    ]
    if group_column:
        # A rule between blocks, matching the on-screen table. Inserted from the
        # bottom up so the earlier indices stay valid.
        for start, _ in reversed(runs[1:]):
            rows.insert(start + 1, r"\midrule")

    spec = "".join(column.align for column in columns)
    tabular = [
        r"\begin{tabular}{" + (f"@{{}}{spec}@{{}}" if compact else spec) + "}",
        r"\toprule",
        rows[0],
        r"\midrule",
        *rows[1:],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if compact:
        # Inside the float, so it cannot leak into the document's other tables.
        tabular.insert(0, rf"\setlength{{\tabcolsep}}{{{COMPACT_TABCOLSEP}pt}}")
    if font_size:
        # Same reason it goes inside: a size command is a switch, and one left
        # outside a float would shrink everything after the table too.
        tabular.insert(0, f"\\{font_size.lstrip(chr(92))}")
    packages = "booktabs, multirow" if group_column else "booktabs"
    preamble = f"% Requires \\usepackage{{{packages}}}"
    if caption is None and label is None:
        return "\n".join([preamble, *tabular]) + "\n"

    wrapped = [preamble, r"\begin{table}[t]", r"  \centering"]
    if caption is not None:
        wrapped.append(rf"  \caption{{{caption}}}")
    if label is not None:
        wrapped.append(rf"  \label{{{label}}}")
    wrapped += [f"  {line}" for line in tabular]
    wrapped.append(r"\end{table}")
    return "\n".join(wrapped) + "\n"
