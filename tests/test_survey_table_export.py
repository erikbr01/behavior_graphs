"""The study's adjusted table, as LaTeX.

What goes wrong here is silent in a way a figure's is not: a table that
compiles can still print a backslash, quote an interval against the wrong row,
or lose a minus sign to a text hyphen — and the first anyone reads it is in the
PDF. So the tests are about the source, not about how it looks.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from behavior_graphs.viz.streamlit.user_study import table_export

#: The live store's fitted estimates, in the shape `claims.adjusted_*` returns.
ROWS = [
    {
        "outcome": "temporal_demand",
        "label": "Temporal demand",
        "better": "down",
        "mean_graph": 38.7,
        "mean_videos": 64.8,
        "difference": -26.1,
        "lower": -38.4,
        "upper": -14.0,
        "relative": -40.3,
        "r_hat": 1.0,
    },
    {
        "outcome": "confidence",
        "label": "Confidence",
        "better": "up",
        "mean_graph": 61.3,
        "mean_videos": 58.8,
        "difference": 2.5,
        "lower": -9.8,
        "upper": 15.5,
        "relative": 4.2,
        "r_hat": 1.0,
    },
    {
        "outcome": "regret",
        "label": "Decision regret",
        "better": "down",
        "mean_graph": 7.9,
        "mean_videos": 7.0,
        "difference": 0.9,
        "lower": -6.1,
        "upper": 8.5,
        "relative": 12.9,
        "r_hat": 1.0,
    },
]


def _row(tex: str, starts_with: str) -> str:
    (line,) = [line for line in tex.splitlines() if line.strip().startswith(starts_with)]
    return line


def test_a_row_carries_both_marginal_means_the_difference_and_its_interval():
    """The four numbers a sentence quotes, in the order the table heads them."""
    tex = table_export.adjusted_table_tex(ROWS)

    cells = [cell.strip() for cell in _row(tex, "Temporal").split("&")]

    assert cells[0] == r"Temporal demand $\downarrow$"
    assert cells[1] == "$38.7$"
    assert cells[2] == "$64.8$"
    # The estimate and its interval in one cell: they are quoted together in
    # every sentence that uses them, and splitting them is how an interval ends
    # up read against the wrong row.
    assert cells[3].rstrip("\\").strip() == r"$-26.1$ $[-38.4, -14.0]$"


def test_the_difference_is_signed_and_the_marginal_means_are_not():
    """`+2.5` says "more confidence with the graph"; `2.5` leaves the direction
    to be worked out from two other columns. A mean has no direction to sign."""
    tex = table_export.adjusted_table_tex(ROWS)

    cells = [cell.strip() for cell in _row(tex, "Confidence").split("&")]

    assert cells[1] == "$61.3$" and cells[2] == "$58.8$"
    assert cells[3].rstrip("\\").strip() == r"$+2.5$ $[-9.8, +15.5]$"


def test_the_convergence_diagnostic_is_a_caption_sentence_and_not_a_column():
    """A column of it could only ever hold passing values — the fit raises
    rather than report an estimate with r-hat above 1.01 — so it was 35 pt of a
    252 pt column spent saying four times what the caption says once. The
    on-screen table keeps it: a page has the room, and whoever is looking at a
    fresh fit is who wants to see it."""
    tex = table_export.adjusted_table_tex(ROWS)

    assert r"$\hat{r} > 1.01$" in tex  # in the caption
    header = _row(tex, "Outcome")
    assert r"$\hat{r}$" not in header
    assert len(header.split("&")) == 4
    # And no second header line over the two display columns: one word, one
    # whole line of the table, and no width bought by it.
    assert r"\multicolumn" not in tex and r"\cmidrule" not in tex


def test_the_table_is_set_small_inside_its_float():
    r"""At full size it is 264 pt against a 252 pt column, so this is not a
    preference: `\small` is what makes it fit. Inside the float, because a size
    command is a switch and one left outside would shrink the rest of the
    document."""
    tex = table_export.adjusted_table_tex(ROWS)
    lines = [line.strip() for line in tex.splitlines()]

    assert lines.index(r"\small") > lines.index(r"\begin{table}[t]")
    assert lines.index(r"\small") < lines.index(r"\begin{tabular}{@{}lrrl@{}}")
    # And it can be turned off by a caller whose document sets its own size.
    assert r"\small" not in table_export.adjusted_table_tex(ROWS, font_size=None)


def test_every_number_is_set_in_math_so_a_minus_is_a_minus():
    """LaTeX's text hyphen is shorter than the minus in the figure beside this
    table, and a table of negative effects is exactly where that shows."""
    tex = table_export.adjusted_table_tex(ROWS)

    for line in tex.splitlines():
        if not line.strip().startswith(("Temporal", "Confidence", "Decision")):
            continue
        for cell in line.split("&")[1:]:
            body = cell.replace(r"\\", "").strip()
            assert body.startswith("$") and body.endswith("$"), cell


def test_the_arrow_is_the_direction_and_the_word_is_left_to_the_caption():
    """The same convention as the figures: an arrow after the name, and what it
    means said once underneath rather than four times inside."""
    tex = table_export.adjusted_table_tex(ROWS)

    assert r"Temporal demand $\downarrow$" in tex
    assert r"Confidence $\uparrow$" in tex
    assert r"Decision regret $\downarrow$" in tex
    # Not in a cell. The caption says it, which is a different line.
    assert "better" not in _row(tex, "Temporal")


def test_the_interval_is_named_a_credible_interval():
    """It is the 95% HDI of the posterior, which is a credible interval — and
    "CrI" is what a reader of the paper knows. Which of the credible intervals
    it is belongs in the caption, not in a column head."""
    tex = table_export.adjusted_table_tex(ROWS)

    assert r"Difference [95\% CrI]" in tex
    assert "highest density" in tex  # said once, in the caption
    # Escaped, or the header ends the row early and LaTeX reads the rest as a
    # comment.
    assert "95% CrI" not in tex


def test_the_caption_says_the_decision_row_is_not_in_rating_points():
    """Three rows are on the 0–100 scale participants answered on and the
    fourth is policy success rate. Nothing in the columns can say that, and a
    reader who misses it compares +0.9 against +2.5 as if they were the same
    unit."""
    tex = table_export.adjusted_table_tex(ROWS)

    assert "rating points" in tex
    assert "success rate" in tex


def test_an_outcome_name_cannot_smuggle_latex_into_the_table():
    """A label reaches this from a dict of display names, so an underscore in
    one would arrive as a subscript and a percent sign would comment out the
    rest of the row."""
    tex = table_export.adjusted_table_tex(
        [{**ROWS[0], "label": "Mental_demand 50% & up", "better": None}]
    )

    assert r"Mental\_demand 50\% \& up" in tex
    # And no arrow, since that row named no direction.
    assert "downarrow" not in _row(tex, "Mental")


def test_a_missing_estimate_is_a_dash_rather_than_a_zero():
    """A fit that did not report a marginal mean has no mean, which is not the
    same as a mean of nothing. `--` is what the repo's other tables print."""
    tex = table_export.adjusted_table_tex(
        [
            {
                "outcome": "effort",
                "label": "Effort",
                "better": "down",
                "mean_graph": None,
                "mean_videos": None,
                "difference": -22.9,
                "lower": None,
                "upper": None,
                "r_hat": None,
            }
        ]
    )

    cells = [cell.strip() for cell in _row(tex, "Effort").split("&")]

    assert cells[1] == "--" and cells[2] == "--"
    # The point estimate survives on its own; an interval needs both bounds.
    assert cells[3].rstrip("\\").strip() == "$-22.9$"


def test_the_source_names_the_package_it_needs():
    """As `latex_table.booktabs_table` does: the rules are booktabs', and a
    paste into a document without it fails at the first `\\toprule`."""
    tex = table_export.adjusted_table_tex(ROWS)

    assert tex.splitlines()[0] == r"% Requires \usepackage{booktabs}"
    assert r"\toprule" in tex and r"\midrule" in tex and r"\bottomrule" in tex


def test_the_float_is_dropped_when_there_is_no_caption_or_label():
    """A table placed by hand does not want a float around it, and the caller
    that wants one gets `\\caption` and `\\label` filled in."""
    bare = table_export.adjusted_table_tex(ROWS, caption=None, label=None)
    floated = table_export.adjusted_table_tex(ROWS, caption="Mine.", label="tab:mine")

    assert r"\begin{table}" not in bare and r"\begin{tabular}" in bare
    assert r"\begin{table}[t]" in floated
    assert r"  \caption{Mine.}" in floated
    assert r"  \label{tab:mine}" in floated


def test_nothing_fitted_exports_nothing():
    """Rather than a float with a rule and no rows, which reads as a table
    whose numbers went missing."""
    assert table_export.adjusted_table_tex([]) == ""


def test_the_emitted_table_compiles_and_fits_the_column_it_is_for():
    """The one thing about a LaTeX export that cannot be checked by reading it.

    A table that overflows its column is not a compile error and not a warning
    a reader would look for -- it is a paragraph of the paper with a table
    sticking out of it, found in the PDF. This one was 295 pt in a 252 pt
    column when it had five columns and a spanning head.

    Skipped where TeX is not installed, which is most machines that run this
    suite; it is the deployment that has it.
    """
    if not shutil.which("pdflatex"):
        pytest.skip("no pdflatex: the fit is measured where TeX is installed")

    import tempfile
    from pathlib import Path

    tex = table_export.adjusted_table_tex(ROWS)
    # The same table with no float around it, which is what can go in an
    # `\sbox` to be measured: a `\caption` in one is a LaTeX error.
    bare = table_export.adjusted_table_tex(ROWS, caption=None, label=None)
    body = "\n".join(line for line in bare.splitlines() if not line.startswith("%"))
    document = "\n".join(
        [
            r"\documentclass[conference]{IEEEtran}",
            r"\usepackage{booktabs}",
            r"\begin{document}",
            # The float, so a malformed one fails here rather than in whoever
            # pastes it.
            tex,
            # And the same table in a box, to measure.
            r"\sbox0{%",
            body,
            "}",
            # Two lines, because TeX's number scanner eats whatever follows
            # `\the\wd0` while it looks for more digits -- a `\space` between
            # them lands inside the number and comes out as "243.7ptof".
            r"\typeout{MEASURED-WIDTH \the\wd0}",
            r"\typeout{MEASURED-COLUMN \the\columnwidth}",
            r"\end{document}",
        ]
    )

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "table.tex"
        source.write_text(document + "\n")
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", source.name],
            cwd=directory,
            capture_output=True,
            text=True,
        )

    assert result.returncode == 0, result.stdout[-2000:]
    # An overfull box is LaTeX saying the table is wider than what holds it,
    # which is the failure this test exists for.
    assert "Overfull" not in result.stdout
    found = re.search(r"MEASURED-WIDTH ([\d.]+)pt", result.stdout)
    room = re.search(r"MEASURED-COLUMN ([\d.]+)pt", result.stdout)
    assert found and room, result.stdout[-2000:]
    width, column = float(found.group(1)), float(room.group(1))
    assert width <= column, f"{width} pt in a {column} pt column"
