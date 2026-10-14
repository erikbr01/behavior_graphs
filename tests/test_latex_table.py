"""What the LaTeX export has to get right that a screen table does not.

Three obligations, and they are the reason this is not ``DataFrame.to_latex``:
a method called ``graph_probe`` must not arrive in a paper as "graphprobe" with
a subscript; a metric's mean and its spread belong in one cell; and an
undefined metric must read as undefined rather than as ``nan``, ``0``, or a
blank the reader takes for an oversight.

Bolding gets the most attention here because it is the claim the table makes.
Bolding the wrong number -- the largest Brier, the method scored on the most
datasets, a lone row, a seed that got lucky -- is a wrong claim, so each of
those is pinned.
"""

from __future__ import annotations

import pandas as pd
import pytest

from behavior_graphs.viz.latex_table import MISSING, booktabs_table, escape, header_label

NAN = float("nan")


def _frame(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


#: The shape the Failure Detection aggregate has: identity, mean/std pairs, and
#: denominators that are numbers but not scores.
AGGREGATE = _frame(
    method=["graph_probe_path", "safe_mlp", "logistic"],
    mean_roc_auc=[0.8123, 0.7719, 0.6501],
    std_roc_auc=[0.0412, 0.0330, 0.0210],
    mean_brier=[0.19, 0.21, 0.15],
    std_brier=[0.02, 0.03, 0.01],
    auc_wins=[3, 1, 0],
    n_datasets=[6, 6, 6],
)
AGGREGATE_SCORES = ["mean_roc_auc", "std_roc_auc", "mean_brier", "std_brier", "auc_wins"]


def _export(frame=AGGREGATE, scores=None, **kwargs) -> str:
    return booktabs_table(frame, score_columns=scores or AGGREGATE_SCORES, **kwargs)


def _rows(latex: str) -> list[str]:
    """The body rows, between ``\\midrule`` and ``\\bottomrule``."""
    lines = [line.strip() for line in latex.splitlines()]
    return lines[lines.index(r"\midrule") + 1 : lines.index(r"\bottomrule")]


def _header(latex: str) -> list[str]:
    """The header row, between ``\\toprule`` and ``\\midrule``."""
    lines = [line.strip() for line in latex.splitlines()]
    row = lines[lines.index(r"\toprule") + 1]
    return [cell.strip() for cell in row.removesuffix(r"\\").split("&")]


def _cells(latex: str, row: int) -> list[str]:
    return [cell.strip() for cell in _rows(latex)[row].removesuffix(r"\\").split("&")]


class TestStructure:
    def test_it_is_booktabs_and_not_a_grid(self):
        latex = _export()
        assert r"\toprule" in latex and r"\midrule" in latex and r"\bottomrule" in latex
        # booktabs' whole argument: no vertical rules, no \hline between rows.
        assert r"\hline" not in latex
        assert "|" not in latex

    def test_the_column_spec_left_aligns_names_and_right_aligns_numbers(self):
        # One column per emitted column, or LaTeX refuses to compile at all.
        assert r"\begin{tabular}{lrrrr}" in _export()

    def test_one_body_row_per_frame_row_under_a_header(self):
        latex = _export()
        assert len(_rows(latex)) == len(AGGREGATE)
        assert _header(latex)[0] == "Method"

    def test_a_bare_tabular_unless_asked_to_float(self):
        assert r"\begin{table}" not in _export()
        assert r"\begin{table}" in _export(caption="Failure detection.")
        assert r"\begin{table}" in _export(label="tab:fd")

    def test_caption_and_label_land_in_the_float(self):
        latex = _export(caption="Failure detection.", label="tab:fd")
        assert r"\caption{Failure detection.}" in latex
        assert r"\label{tab:fd}" in latex
        assert latex.index(r"\caption") < latex.index(r"\begin{tabular}")

    def test_it_says_which_package_it_needs(self):
        assert _export().startswith(r"% Requires \usepackage{booktabs}")

    def test_an_empty_frame_exports_nothing_rather_than_a_broken_tabular(self):
        assert _export(pd.DataFrame()) == ""


class TestEscaping:
    def test_a_method_name_keeps_its_underscores(self):
        # Unescaped, `graph_probe_path` typesets as "graphprobepath" with two
        # subscripts -- a plausible-looking method name that is not the one run.
        assert r"graph\_probe\_path" in _export()
        assert "graph_probe_path" not in _export()

    def test_headers_are_readable_rather_than_column_names(self):
        assert header_label("mean_roc_auc") == "Mean ROC AUC"
        assert header_label("calibrated_accuracy") == "Calibrated accuracy"
        assert header_label("n_datasets") == "N datasets"
        assert header_label("f1") == "F1"

    def test_every_special_character_is_escaped(self):
        assert escape("a_b & 50% $x$ #1 {c} ~d^e") == (
            r"a\_b \& 50\% \$x\$ \#1 \{c\} \textasciitilde{}d\textasciicircum{}e"
        )

    def test_a_backslash_is_escaped_once_not_twice(self):
        assert escape("a\\b") == r"a\textbackslash{}b"

    def test_a_text_column_is_escaped_too(self):
        frame = _frame(dataset=["set_table"], mean_roc_auc=[0.5], std_roc_auc=[0.1])
        assert r"set\_table" in booktabs_table(frame, score_columns=["mean_roc_auc"])


class TestNumbers:
    def test_precision_applies_to_scores(self):
        assert "0.812" in _export(precision=3)
        assert "0.81" in _export(precision=2)
        assert "0.8123" in _export(precision=4)

    def test_a_count_is_not_given_decimals(self):
        # `n_datasets` is 6, never 6.000, and neither is a whole-numbered score.
        cells = _cells(_export(precision=3), 0)
        assert cells[-1] == "6"
        assert cells[-2] == "3"  # auc_wins, a score but an integral one

    def test_a_float_count_carrying_a_nan_is_still_a_count(self):
        # `n_datasets_calibrated` goes float the moment one method could not be
        # calibrated; 5.000 in a denominator column is noise.
        frame = _frame(method=["a", "b"], mean_roc_auc=[0.5, 0.6], n_datasets_calibrated=[5.0, NAN])
        assert _cells(booktabs_table(frame, score_columns=["mean_roc_auc"]), 0)[-1] == "5"

    def test_a_score_that_is_whole_keeps_its_decimals(self):
        # The counterpart: an AUC of exactly 1.0 on every dataset is a score,
        # and `1` in a column of `0.812`s reads as a different quantity.
        frame = _frame(method=["a", "b"], mean_roc_auc=[1.0, 1.0])
        assert _cells(booktabs_table(frame, score_columns=["mean_roc_auc"]), 0)[-1] == "1.000"


class TestMeanAndSpread:
    def test_a_mean_and_its_std_share_one_cell(self):
        assert _cells(_export(), 0)[1] == r"$0.812 \pm 0.041$"

    def test_the_merged_column_is_headed_with_the_metric(self):
        assert _cells(_export(), 0)[0] == r"graph\_probe\_path"
        assert _header(_export())[1] == "ROC AUC"

    def test_merging_off_keeps_both_columns(self):
        latex = _export(merge_std=False)
        assert r"\pm" not in latex
        assert "Mean ROC AUC" in latex and "Std ROC AUC" in latex

    def test_a_spread_listed_before_its_mean_is_still_merged_once(self):
        # Column order is the caller's, and nothing says a mean comes first.
        # Folding as the loop went emitted the leading `std_x` twice: once as
        # its own column and again inside the merged cell.
        frame = AGGREGATE[["method", "std_roc_auc", "mean_roc_auc"]]
        latex = booktabs_table(frame, score_columns=["mean_roc_auc", "std_roc_auc"])
        assert _header(latex) == ["Method", "ROC AUC"]
        assert latex.count(r"\pm") == len(frame)
        assert _cells(latex, 0) == [r"graph\_probe\_path", r"$0.812 \pm 0.041$"]

    def test_a_std_whose_mean_was_not_selected_is_not_swallowed(self):
        # Dropping it would lose a number the caller asked to see.
        latex = _export(scores=["std_roc_auc"])
        assert "Std ROC AUC" in latex
        assert r"\pm" not in latex

    def test_a_missing_spread_leaves_the_mean_alone(self):
        frame = _frame(method=["a"], mean_roc_auc=[0.5], std_roc_auc=[NAN])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc", "std_roc_auc"])
        assert _cells(latex, 0)[1] == "$0.500$"

    def test_a_missing_mean_makes_the_whole_cell_missing(self):
        # A spread around a centre that does not exist says nothing.
        frame = _frame(method=["a"], mean_roc_auc=[NAN], std_roc_auc=[0.01])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc", "std_roc_auc"])
        assert _cells(latex, 0)[1] == MISSING


class TestMissing:
    def test_an_undefined_metric_reads_as_undefined(self):
        frame = _frame(method=["a", "b"], mean_roc_auc=[0.5, NAN])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc"])
        assert _cells(latex, 1)[1] == MISSING
        assert "nan" not in latex.lower()

    def test_it_is_not_quietly_zero(self):
        frame = _frame(method=["a", "b"], mean_roc_auc=[0.5, NAN])
        assert "0.000" not in booktabs_table(frame, score_columns=["mean_roc_auc"])


class TestBolding:
    def test_off_by_default(self):
        assert r"\mathbf" not in _export()
        assert r"\textbf" not in _export()

    def test_it_bolds_the_best_method_in_each_metric(self):
        latex = _export(bold_best=True)
        assert _cells(latex, 0)[1] == r"$\mathbf{0.812} \pm 0.041$"  # highest AUC
        assert r"\mathbf" not in _cells(latex, 1)[1]
        assert r"\mathbf" not in _cells(latex, 2)[1]

    def test_a_lower_is_better_metric_bolds_its_minimum(self):
        # The whole reason direction is not assumed: bolding the largest Brier
        # score would advertise the worst-calibrated method as the best.
        assert _cells(_export(bold_best=True), 2)[2] == r"$\mathbf{0.150} \pm 0.010$"

    def test_only_the_mean_is_bolded_not_the_spread(self):
        cell = _cells(_export(bold_best=True), 0)[1]
        assert cell.endswith(r"\pm 0.041$")

    def test_a_denominator_is_never_bolded(self):
        # `n_datasets` is a number, but "scored on the most datasets" is not a
        # win, and a run where it varies must not look like one.
        frame = _frame(method=["a", "b"], mean_roc_auc=[0.5, 0.6], n_datasets=[6, 4])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc"], bold_best=True)
        assert _cells(latex, 0)[-1] == "6"
        assert _cells(latex, 1)[1].startswith(r"\textbf")  # the score still wins

    def test_a_spread_is_never_bolded(self):
        # Split into two columns, `std_roc_auc` is bold-eligible only if a
        # spread has a better direction -- and the largest one belongs to the
        # *least* reproducible method, the opposite of a win.
        rows = [_cells(_export(bold_best=True, merge_std=False), row) for row in range(3)]
        assert rows[0][1] == r"\textbf{0.812}"  # the mean still wins
        assert [row[2] for row in rows] == ["0.041", "0.033", "0.021"]

    def test_a_tie_bolds_everyone_rather_than_the_first_row(self):
        frame = _frame(method=["a", "b", "c"], mean_roc_auc=[0.9, 0.9, 0.5])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc"], bold_best=True)
        assert _cells(latex, 0)[1].startswith(r"\textbf")
        assert _cells(latex, 1)[1].startswith(r"\textbf")
        assert not _cells(latex, 2)[1].startswith(r"\textbf")

    def test_a_single_row_wins_nothing(self):
        frame = _frame(method=["a"], mean_roc_auc=[0.9])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc"], bold_best=True)
        assert r"\textbf" not in latex

    def test_a_column_where_every_method_ties_wins_nothing(self):
        frame = _frame(method=["a", "b"], mean_roc_auc=[0.9, 0.9])
        latex = booktabs_table(frame, score_columns=["mean_roc_auc"], bold_best=True)
        assert r"\textbf" not in latex

    def test_a_missing_value_never_wins(self):
        frame = _frame(method=["a", "b", "c"], mean_brier=[NAN, 0.4, 0.9])
        latex = booktabs_table(frame, score_columns=["mean_brier"], bold_best=True)
        assert _cells(latex, 0)[1] == MISSING
        assert _cells(latex, 1)[1].startswith(r"\textbf")
        assert not _cells(latex, 2)[1].startswith(r"\textbf")

    def test_the_only_method_that_scored_wins_nothing(self):
        # One defined value is not a comparison, and bolding it would claim the
        # method beat rivals that never produced a number.
        frame = _frame(method=["a", "b"], mean_brier=[NAN, 0.4])
        latex = booktabs_table(frame, score_columns=["mean_brier"], bold_best=True)
        assert r"\textbf" not in latex

    def test_the_prefixed_spelling_of_a_lower_is_better_metric_still_counts(self):
        frame = _frame(method=["a", "b"], mean_calibrated_brier=[0.1, 0.4])
        latex = booktabs_table(frame, score_columns=["mean_calibrated_brier"], bold_best=True)
        assert _cells(latex, 0)[1].startswith(r"\textbf")

    def test_the_direction_set_is_the_caller_s_to_override(self):
        latex = _export(bold_best=True, lower_is_better=["roc_auc"])
        assert _cells(latex, 2)[1].startswith("$\\mathbf{0.650}")


class TestPasteability:
    """The output has to compile, which mostly means balanced delimiters."""

    @pytest.mark.parametrize("kwargs", [{}, {"bold_best": True}, {"merge_std": False}])
    def test_every_row_has_the_same_number_of_cells_as_the_header(self, kwargs):
        rows = _rows(_export(**kwargs))
        widths = {row.count("&") for row in rows}
        assert len(widths) == 1

    @pytest.mark.parametrize("kwargs", [{}, {"bold_best": True}, {"caption": "x"}])
    def test_math_mode_and_braces_balance(self, kwargs):
        latex = _export(**kwargs)
        assert latex.count("$") % 2 == 0
        assert latex.count("{") == latex.count("}")

    def test_every_row_is_terminated(self):
        assert all(row.endswith(r"\\") for row in _rows(_export()))


class TestGroupedRows:
    """``group_column`` — a table drawn as grouped rows has to export that way."""

    def _frame(self):
        return pd.DataFrame(
            {
                "rubric": ["libero (6)", "libero (6)", "droid (19)", "droid (19)"],
                "method": ["graph", "mlp", "graph", "mlp"],
                "voc": [0.912, 0.940, 0.948, 0.731],
                "mae": [0.082, 0.068, 0.047, 0.173],
            }
        )

    def _tex(self, **kwargs) -> str:
        return booktabs_table(
            self._frame(),
            score_columns=["voc", "mae"],
            merge_std=False,
            group_column="rubric",
            **kwargs,
        )

    def test_one_spanning_cell_per_block_and_a_rule_between(self):
        tex = self._tex()

        assert tex.count(r"\multirow{2}{*}{libero (6)}") == 1
        assert tex.count(r"\multirow{2}{*}{droid (19)}") == 1
        # The label is not repeated on the block's other rows.
        assert tex.count("libero (6)") == 1
        # \toprule ... header \midrule ... block \midrule ... block \bottomrule
        assert tex.count(r"\midrule") == 2

    def test_the_package_it_needs_is_announced(self):
        assert r"% Requires \usepackage{booktabs, multirow}" in self._tex()
        # And not demanded when nothing is grouped.
        plain = booktabs_table(self._frame(), score_columns=["voc"], merge_std=False)
        assert r"% Requires \usepackage{booktabs}" in plain

    def test_the_winner_is_per_block_not_per_column(self):
        """A column spans every group, so its single best cell would be
        comparing libero against droid — a difference in task difficulty."""
        tex = self._tex(bold_best=True)

        # Best voc in each block: 0.940 (libero) and 0.948 (droid).
        assert r"\textbf{0.940}" in tex
        assert r"\textbf{0.948}" in tex
        # 0.912 is second in its block and must not be bolded even though it
        # beats droid's mlp row.
        assert r"\textbf{0.912}" not in tex
        assert r"\textbf{0.731}" not in tex

    def test_a_lower_is_better_column_is_not_ranked_upside_down(self):
        tex = self._tex(bold_best=True)

        # Lowest MAE per block: 0.068 and 0.047.
        assert r"\textbf{0.068}" in tex
        assert r"\textbf{0.047}" in tex
        assert r"\textbf{0.082}" not in tex
        assert r"\textbf{0.173}" not in tex

    def test_the_metric_direction_ignores_case(self):
        """A caller may rename a column to the acronym a table should SET —
        `MAE` — and a case-sensitive lookup then bolds the worst cell."""
        frame = self._frame().rename(columns={"voc": "VOC", "mae": "MAE"})

        tex = booktabs_table(
            frame,
            score_columns=["VOC", "MAE"],
            merge_std=False,
            group_column="rubric",
            bold_best=True,
        )

        assert r"\textbf{0.068}" in tex and r"\textbf{0.047}" in tex
        assert r"\textbf{0.082}" not in tex and r"\textbf{0.173}" not in tex

    def test_non_adjacent_rows_of_one_group_are_separate_blocks(self):
        """`\\multirow` spans consecutive rows only, so runs are what can be
        merged — a groupby would claim a span LaTeX cannot typeset."""
        frame = pd.DataFrame(
            {
                "rubric": ["a", "b", "a"],
                "method": ["x", "y", "z"],
                "voc": [0.1, 0.2, 0.3],
            }
        )

        tex = booktabs_table(frame, score_columns=["voc"], merge_std=False, group_column="rubric")

        # Three runs of one row each: no \multirow at all, and a rule between.
        assert r"\multirow" not in tex
        assert tex.count("a ") >= 1
        assert tex.count(r"\midrule") == 3


class TestCompact:
    """``compact`` — for a table that has to fit a single column.

    Measured under IEEEtran on the progress-estimation rubric table: 323 pt
    as-is against a 252 pt column, 265 pt compact. Every knob here is width
    with no information removed, which is why they are one flag.
    """

    def _frame(self):
        return pd.DataFrame(
            {
                "method": ["graph", "mlp"],
                "n_datasets": [19, 19],
                "voc": [0.948, 0.731],
                "kendall": [0.930, 0.564],
                "auroc": [0.904, 0.782],
                "mae": [0.047, -0.173],
            }
        )

    def _tex(self, **kwargs) -> str:
        return booktabs_table(
            self._frame(),
            score_columns=["voc", "kendall", "auroc", "mae"],
            merge_std=False,
            compact=True,
            **kwargs,
        )

    def test_the_outer_padding_is_trimmed(self):
        # `n_datasets` is numeric, so it aligns right: l r r r r r.
        assert r"\begin{tabular}{@{}lrrrrr@{}}" in self._tex()
        # And not when compact is off.
        plain = booktabs_table(self._frame(), score_columns=["voc"], merge_std=False)
        assert r"\begin{tabular}{lrrrrr}" in plain

    def test_the_column_padding_is_narrowed_inside_the_table_only(self):
        tex = self._tex(caption="c", label="l")

        assert r"\setlength{\tabcolsep}{4pt}" in tex
        # Inside the float, so it cannot leak into the document's other tables.
        assert tex.index(r"\begin{table}") < tex.index(r"\setlength")
        assert tex.index(r"\setlength") < tex.index(r"\end{table}")

    def test_compact_keeps_the_leading_zero(self):
        """It is the weakest of the width knobs -- 4 pt of an IEEE column with
        three metric columns -- and a number without its zero is harder to
        read, so it is opt-in rather than part of compact."""
        assert "0.948" in self._tex()

    def test_the_leading_zero_goes_from_scores_but_not_from_counts(self):
        tex = self._tex(strip_leading_zero=True)

        assert ".948" in tex and "0.948" not in tex
        # A negative keeps its sign and loses its zero, or the column would
        # look misaligned rather than compact.
        assert "-.173" in tex and "-0.173" not in tex
        # `n_datasets` is a count, not a score: it keeps its digits.
        assert "19" in tex

    def test_the_wide_headers_are_set_short(self):
        tex = self._tex()

        assert r"$\tau$" in tex and "Kendall" not in tex
        assert "AUC" in tex and "AUROC" not in tex
        # VOC stays: it is already short, and it is what the work being
        # compared against calls it.
        assert "VOC" in tex

    def test_the_headers_can_be_kept_in_full(self):
        """Worth ~20 pt of an IEEE column, so it is a knob: a table set
        `footnotesize` fits with the words spelled out, one set `small` does
        not."""
        tex = self._tex(abbreviate_headers=False)

        assert "Kendall" in tex and "AUROC" in tex
        assert r"$\tau$" not in tex and ">AUC<" not in tex
        # Still compact in every other respect.
        assert r"\setlength{\tabcolsep}{4pt}" in tex
        assert "@{}" in tex

    def test_compact_does_not_change_which_cell_wins(self):
        """It is a width knob. If it moved the bolding it would be a claim."""
        plain = booktabs_table(
            self._frame(),
            score_columns=["voc", "kendall", "auroc", "mae"],
            merge_std=False,
            bold_best=True,
        )
        tight = self._tex(bold_best=True, strip_leading_zero=True)

        # graph wins voc/kendall/auroc; mae is lower-is-better, so mlp's -0.173.
        assert r"\textbf{0.948}" in plain and r"\textbf{.948}" in tight
        assert r"\textbf{-0.173}" in plain and r"\textbf{-.173}" in tight


class TestFontSize:
    """``font_size`` — the widest-reaching knob, and the one that must not leak.

    A size command is a switch: left outside the float it would shrink the rest
    of the document. Worth ~23 pt of a 252 pt IEEE column at ``small``, measured
    on the progress-estimation rubric table.
    """

    def _frame(self):
        return pd.DataFrame({"method": ["graph", "mlp"], "voc": [0.948, 0.731]})

    def test_the_size_command_goes_inside_the_float(self):
        tex = booktabs_table(
            self._frame(),
            score_columns=["voc"],
            merge_std=False,
            caption="c",
            label="l",
            font_size="small",
        )

        assert r"\small" in tex
        assert tex.index(r"\begin{table}") < tex.index(r"\small")
        assert tex.index(r"\small") < tex.index(r"\begin{tabular}")
        assert tex.index(r"\small") < tex.index(r"\end{table}")

    def test_a_leading_backslash_is_tolerated(self):
        """`\\small` and `small` are the same request; a doubled backslash is a
        LaTeX error rather than a smaller table."""
        tex = booktabs_table(
            self._frame(), score_columns=["voc"], merge_std=False, font_size=r"\footnotesize"
        )

        assert r"\footnotesize" in tex
        assert r"\\footnotesize" not in tex

    def test_no_size_command_unless_asked(self):
        tex = booktabs_table(self._frame(), score_columns=["voc"], merge_std=False)

        assert r"\small" not in tex and r"\footnotesize" not in tex


# ── Metrics with no better direction ─────────────────────────────────────────


class TestNotRankable:
    """A measurement can be a number and still have no best value.

    Density is predicted segments over ground-truth segments: 1.0 is the target
    and both 0.3 and 3.0 are failures, a collapse and an over-segmentation.
    Neither ``LOWER_IS_BETTER`` nor its complement expresses that, so bolding it
    either way makes a claim the metric does not support -- and bolding the
    largest reads as "this row segmented best" when it means the opposite. That
    went out in a table before it was caught.
    """

    def test_density_is_never_bolded(self):
        frame = pd.DataFrame(
            {
                "method": ["a", "b", "c"],
                "f1": [0.30, 0.40, 0.20],
                "density": [3.00, 1.02, 0.31],
            }
        )
        tex = booktabs_table(frame, score_columns=("f1", "density"), bold_best=True)

        # The best F1 is still bolded, so this is not "bolding turned off".
        assert r"\textbf{0.400}" in tex
        # No density is, neither the largest, the smallest, nor the nearest 1.
        assert r"\textbf{3.000}" not in tex
        assert r"\textbf{0.310}" not in tex
        assert r"\textbf{1.020}" not in tex

    def test_density_still_gets_score_precision(self):
        """Not rankable is not the same as not a measurement."""
        frame = pd.DataFrame({"method": ["a"], "density": [1.0]})
        tex = booktabs_table(frame, score_columns=("density",), precision=3)

        assert "1.000" in tex

    def test_a_prefixed_density_column_is_also_unrankable(self):
        """``mean_density`` is density, reached through ``base_metric``."""
        frame = pd.DataFrame({"method": ["a", "b"], "mean_density": [2.5, 1.0]})
        tex = booktabs_table(frame, score_columns=("mean_density",), bold_best=True)

        assert r"\textbf" not in tex

    def test_counts_are_not_rankable_either(self):
        """Bolding the row that predicted the most segments means nothing."""
        frame = pd.DataFrame({"method": ["a", "b"], "f1": [0.1, 0.2], "n_pred": [400, 100]})
        tex = booktabs_table(frame, score_columns=("f1", "n_pred"), bold_best=True)

        assert r"\textbf{0.200}" in tex
        assert r"\textbf{400" not in tex
