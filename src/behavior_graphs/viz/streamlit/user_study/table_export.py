"""The study's adjusted estimates as a ``booktabs`` table for the paper.

The same rows the Figures page draws on screen -- what
:func:`~behavior_graphs.viz.streamlit.user_study.claims.adjusted_rating_claims`
and
:func:`~behavior_graphs.viz.streamlit.user_study.claims.adjusted_decision_claim`
return -- so a number in the paper and the number on the page cannot drift
apart. Neither writes a file nor imports Streamlit, so a page offering the
export is a :func:`streamlit.code` call and a download button.

Why not :func:`behavior_graphs.viz.latex_table.booktabs_table`, which is the
repo's table exporter: it renders a frame of *measurements*, and escapes every
text cell on purpose -- ``graph_probe`` has to survive as ``graph\\_probe``
rather than arrive as a subscript. Two of the cells here are raw math by
design: the bracketed interval (``$[-38.4, -14.0]$``) and the arrow after an
outcome's name. Passing them through that function would print the backslashes,
and teaching it to trust a caller's LaTeX would cost the guarantee it exists
for.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from behavior_graphs.viz.latex_table import MISSING, escape

#: The arrow after an outcome's name, by direction — the same note the figures
#: put beside a panel name, in LaTeX's own math rather than matplotlib's.
_BETTER_ARROWS = {"down": r"$\downarrow$", "up": r"$\uparrow$"}

#: What the table says the interval is. **CrI**, not HDI: the paper's readers
#: know a credible interval, and the highest-density choice among the intervals
#: of that credibility is a detail for the caption (which says it) rather than
#: for a column head.
_INTERVAL_HEAD = r"Difference [95\% CrI]"

#: Set inside the float, which is how most IEEE tables are set anyway — and
#: here it is what makes the table fit.
#:
#: Measured with ``\sbox`` under ``IEEEtran`` (conference, 10 pt) against its
#: 252 pt column, on the live store's four rows — the shape this module emits,
#: compiled, not a copy of it (see ``tests/test_survey_table_export.py``, which
#: runs ``pdflatex`` where it is installed):
#:
#: * 295.5 pt — five columns with a spanning ``EMM`` head. **43 pt over.**
#: * 264.2 pt — four columns, one header line. Still 12 pt over.
#: * 243.7 pt — the same, ``\small``. **Fits**, with 8 pt to spare.
#: * 223.2 pt — ``\footnotesize``, if a row is ever added.
#:
#: Two of those 31 pt came from dropping the ``$\hat{r}$`` column, which a fit
#: that failed it could never reach: the model raises instead. The spanning head
#: cost no width at all — it was 8.5 pt of *height*, a header line for one word
#: the caption already says.
DEFAULT_FONT_SIZE = "small"

#: The caption a caller gets unless it passes its own. Long, because everything
#: in it is a thing a reader of the table alone would otherwise get wrong: what
#: the estimand is, that the three rated rows and the decision row are not in
#: the same points, and that the arrows are directions and not results.
DEFAULT_CAPTION = (
    "Adjusted effect of the behavior graph. Each EMM is the model's mean for "
    "that display, averaged over the observed spread of task, stage and prior "
    "experience with the participant intercepts left out; the difference is the "
    "display coefficient with its 95\\% credible interval (highest density). "
    "The three rated outcomes are on the 0--100 scale participants answered on, "
    "so their points are rating points; the success rate regret row is in "
    "points of policy success rate -- the rate the chosen option left on the "
    "table against the best one available. The arrow after an outcome is the "
    "direction that is "
    "better, not a result. Every estimate here converged: the fit raises rather "
    "than report one with $\\hat{r} > 1.01$."
)

#: The default float label, in the ``tab:`` convention the paper's other tables
#: use.
DEFAULT_LABEL = "tab:study-adjusted"


def _number(value: object, precision: int, *, signed: bool = False) -> str:
    """*value* as a math cell, or :data:`MISSING`.

    Math mode so the minus is a minus: LaTeX's text hyphen is shorter than the
    minus in the figure beside it, and a table of negative effects is where
    that shows. *signed* prints ``+`` on a positive, which is what a
    *difference* column wants -- ``+2.5`` says "more confidence with the graph"
    where ``2.5`` leaves the direction to be inferred.
    """
    if value is None:
        return MISSING
    number = float(value)
    if number != number:  # NaN
        return MISSING
    return f"${number:+.{precision}f}$" if signed else f"${number:.{precision}f}$"


def _interval(row: Mapping, precision: int) -> str:
    """The difference and its interval in one cell, or :data:`MISSING`.

    One cell rather than two columns: the point estimate and its interval are
    quoted together in every sentence that uses them, and splitting them is how
    a reader ends up reading an interval against the wrong row.
    """
    mean = row.get("difference")
    lower, upper = row.get("lower"), row.get("upper")
    if mean is None:
        return MISSING
    point = _number(mean, precision, signed=True)
    if lower is None or upper is None:
        return point
    bounds = f"{float(lower):+.{precision}f}, {float(upper):+.{precision}f}"
    return f"{point} $[{bounds}]$"


def _outcome(row: Mapping) -> str:
    """The outcome's name, escaped, with its direction arrow after it."""
    label = escape(str(row.get("label") or row.get("outcome") or ""))
    arrow = _BETTER_ARROWS.get(str(row.get("better") or ""))
    return f"{label} {arrow}" if arrow else label


def adjusted_table_tex(
    rows: Sequence[Mapping],
    *,
    caption: str | None = DEFAULT_CAPTION,
    label: str | None = DEFAULT_LABEL,
    precision: int = 1,
    font_size: str | None = DEFAULT_FONT_SIZE,
) -> str:
    """*rows* as a ``booktabs`` table: the EMMs, the difference, its interval.

    *rows* are :func:`claims.adjusted_rating_claims` rows, optionally with
    :func:`claims.adjusted_decision_claim` appended -- in the order they should
    be printed, which is the order every figure here draws them: the rated
    outcomes, then the decision. ``""`` for no rows, so a caller with nothing
    fitted offers no export rather than an empty float.

    The float is dropped when *caption* and *label* are both ``None``, leaving
    a bare ``tabular`` to place by hand. The preamble comment naming the
    package it needs is emitted either way, as
    :func:`behavior_graphs.viz.latex_table.booktabs_table` does.

    *precision* is one decimal, which is what the estimate supports: the
    Monte-Carlo error on these means runs about 0.1 rating points (the fits
    report ``mcse_mean``), so a second decimal is the sampler's noise, seeded
    and reproducible and not the study's.

    *font_size* is a LaTeX size command without the backslash, applied inside
    the float. It defaults to :data:`DEFAULT_FONT_SIZE` because at full size
    this table does not fit the column it is for — see the measurements there.
    ``None`` leaves the document's size.

    Four columns and one header line, which is the shape those measurements
    argue for: the convergence diagnostic is in the caption rather than in a
    column, since a fit that fails it raises instead of reporting.
    """
    if not rows:
        return ""

    body = [
        [
            _outcome(row),
            _number(row.get("mean_graph"), precision),
            _number(row.get("mean_videos"), precision),
            _interval(row, precision),
        ]
        for row in rows
    ]
    # One header line, and no spanning "EMM" head over the two display columns:
    # it was a whole line of the table for one word the caption already says,
    # and it bought no width -- the columns are set by the numbers under them.
    # The diagnostic is not a column either: the fit raises above r-hat 1.01, so
    # every value that could ever print here is a passing one, and it was 35 pt
    # of a 252 pt column to say so four times.
    headers = ["Outcome", "Graph", "Videos", _INTERVAL_HEAD]
    widths = [max(len(row[index]) for row in [headers, *body]) for index in range(len(headers))]

    def _line(cells: Sequence[str]) -> str:
        padded = [cell.ljust(width) for cell, width in zip(cells, widths, strict=True)]
        return " & ".join(padded).rstrip() + r" \\"

    tabular = [
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        _line(headers),
        r"\midrule",
        *(_line(cells) for cells in body),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    if font_size:
        # Inside the float, so a size command that is a *switch* cannot leak
        # into whatever follows the table.
        tabular.insert(0, "\\" + font_size.lstrip(chr(92)))
    preamble = r"% Requires \usepackage{booktabs}"
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


__all__ = ["DEFAULT_CAPTION", "DEFAULT_FONT_SIZE", "DEFAULT_LABEL", "adjusted_table_tex"]
