"""The interactive figures: what each one draws, and what it must not.

Drawn from small fixtures rather than from the shipped export wherever the claim
is about the chart, so a test that says "no bar where a policy was not an
option" fails when that rule breaks and not when the study's data changes.
``tests/test_paper_results.py`` is where the real numbers are checked.
"""

from __future__ import annotations

import pandas as pd
import pytest

from behavior_graphs.viz.plotly import results as charts


def _task() -> dict:
    return {
        "label": "Set table",
        "arms": [
            {
                "dataset": "set_table_baseline",
                "label": "baseline",
                "successes": 30,
                "trials": 60,
                "letters": "b",
                "baseline": True,
            },
            {
                "dataset": "set_table_plusleft",
                "label": "plus milk left",
                "successes": 48,
                "trials": 60,
                "letters": "a",
                "baseline": False,
            },
        ],
        "pvalues": [[None, 0.01], [0.01, None]],
    }


def _choices() -> list[dict]:
    return [
        {
            "task": "set_table",
            "task_label": "Set the table",
            "display": "graph",
            "dataset": "set_table_plusleft",
            "choice_label": "Milk left",
            "picks": 3,
            "answers": 4,
        },
        {
            "task": "set_table",
            "task_label": "Set the table",
            "display": "videos",
            "dataset": "set_table_plusleft",
            "choice_label": "Milk left",
            "picks": 1,
            "answers": 5,
        },
    ]


def _ratings() -> list[dict]:
    rows = []
    for index, (graph, videos) in enumerate([(20, 80), (20, 80), (40, 60)], start=1):
        for display, score in (("graph", graph), ("videos", videos)):
            rows.append(
                {
                    "participant": f"P{index:02d}",
                    "task": "set_table",
                    "stage": 1,
                    "display": display,
                    "outcome": "effort",
                    "score": float(score),
                }
            )
    return rows


_OUTCOMES = [{"key": "effort", "label": "Effort", "better": "down"}]


# ── The palette ──────────────────────────────────────────────────────────────


def test_the_displays_are_coloured_as_the_paper_figures_colour_them():
    """Two copies of one palette, held together here: a reader who meets the
    orange box in the paper and the blue one on the website for the same
    display has been told the displays are three things."""
    from behavior_graphs.viz.streamlit.user_study import figure_export

    assert charts.DISPLAY_COLORS == figure_export.PHASE_COLORS
    assert charts.DISPLAY_ORDER == tuple(figure_export.PHASE_ORDER)
    assert charts.DISPLAY_LABELS == figure_export.PHASE_TICKS


# ── Data collection ──────────────────────────────────────────────────────────


def test_each_policy_gets_a_violin_and_a_median_with_its_interval():
    figure = charts.success_rate_figure([_task()])
    violins = [trace for trace in figure.data if trace.type == "violin"]
    markers = [trace for trace in figure.data if trace.type == "scatter"]
    assert len(violins) == 2
    assert len(markers) == 2
    # The whisker is the posterior interval, so it has to be asymmetric: an
    # interval drawn symmetrically around the median is wrong at every count
    # away from 50%.
    assert all(marker.error_y.symmetric is False for marker in markers)


def test_a_violin_is_a_posterior_over_the_arms_own_counts():
    """48 of 60 under a uniform prior has its mass near 80%, and a violin drawn
    from the wrong arm's counts would sit somewhere else entirely."""
    figure = charts.success_rate_figure([_task()])
    violin = next(
        trace for trace in figure.data if trace.type == "violin" and trace.name == "plus milk left"
    )
    assert min(violin.y) > 55
    assert max(violin.y) < 95
    assert float(sum(violin.y) / len(violin.y)) == pytest.approx(100 * 49 / 62, abs=0.5)


def test_every_letter_is_annotated_above_its_policy():
    figure = charts.success_rate_figure([_task()])
    annotated = {
        annotation.text: annotation.x
        for annotation in figure.layout.annotations
        if annotation.text.startswith("<b>")
    }
    assert annotated == {"<b>b</b>": "baseline", "<b>a</b>": "plus milk left"}


def test_the_baseline_gets_no_choice_bar():
    """Nobody was offered the option of collecting nothing, so a zero there
    would say a policy went unchosen when it was never on the screen."""
    figure = charts.success_rate_figure([_task()], _choices())
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert len(bars) == 2  # one per display
    for bar in bars:
        assert tuple(bar.x) == ("baseline", "plus milk left")
        assert bar.y[0] is None


def test_a_choice_bar_is_the_share_of_that_displays_answers():
    figure = charts.success_rate_figure([_task()], _choices())
    graph = next(trace for trace in figure.data if trace.type == "bar" and trace.name == "Graph")
    videos = next(trace for trace in figure.data if trace.type == "bar" and trace.name == "Videos")
    assert graph.y[1] == pytest.approx(75.0)  # 3 of 4
    assert videos.y[1] == pytest.approx(20.0)  # 1 of 5
    # The counts travel with the bar, because a share of four is a sentence a
    # reader has to be able to check.
    assert tuple(graph.customdata[1]) == (3, 4)


def test_the_choices_join_on_the_policy_not_on_the_tasks_name():
    """The two exports name a task differently ("Set table" against "Set the
    table"), and an earlier join on the name silently drew no bars at all."""
    choices = [{**row, "task_label": "something else", "task": "other"} for row in _choices()]
    figure = charts.success_rate_figure([_task()], choices)
    assert [trace for trace in figure.data if trace.type == "bar"]


def test_no_choices_is_a_figure_with_no_second_row():
    figure = charts.success_rate_figure([_task()], [])
    assert not [trace for trace in figure.data if trace.type == "bar"]
    assert "yaxis2" not in figure.layout


def test_the_policy_table_carries_the_counts_behind_every_rate():
    frame = charts.policy_table(_task())
    assert list(frame["Policy"]) == ["baseline", "plus milk left"]
    assert list(frame["Successes"]) == [30, 48]
    assert frame["Success rate (%)"].tolist() == [50.0, 80.0]
    assert list(frame["Letters"]) == ["b", "a"]


# ── The user study ───────────────────────────────────────────────────────────


def _regret_rows() -> list[dict]:
    return [
        {
            "task_label": "Set the table",
            "display": "graph",
            "choice_label": "Milk left",
            "regret": 0.0,
        },
        {
            "task_label": "Cook the meal",
            "display": "videos",
            "choice_label": "Lid",
            "regret": 5.0,
        },
    ]


def _answers_figure():
    return charts.study_answers_figure(
        {"graph": 16, "videos": 2}, _ratings(), _OUTCOMES, _regret_rows()
    )


def test_the_study_figure_carries_the_panels_the_paper_carries():
    """One figure with the paper's own panels, in its order, so a reader with
    the pdf open is looking at the same thing rather than a rearrangement."""
    titles = [a.text for a in _answers_figure().layout.annotations]
    assert titles == ["(a) Chosen display", "(b) Effort \u2193", "(e) Success rate regret \u2193"]


def test_each_panel_keeps_its_own_axis():
    """A count, a rating and a success rate are three units, and one axis
    across them would be three claims at once."""
    layout = _answers_figure().layout
    assert layout.yaxis.title.text == "Participants"
    assert layout.yaxis2.title.text == "Rating"
    assert layout.yaxis3.title.text == "Success rate points"
    assert tuple(layout.yaxis2.range) == (-8, 108)


def test_the_preference_panel_names_both_displays_even_at_zero():
    """A missing bar reads as missing data rather than as a zero."""
    figure = charts.study_answers_figure(
        {"graph": 16, "videos": 0}, _ratings(), _OUTCOMES, _regret_rows()
    )
    bars = [trace for trace in figure.data if trace.type == "bar"]
    assert [trace.x[0] for trace in bars] == ["Graph", "Videos"]
    assert [trace.y[0] for trace in bars] == [16, 0]
    # And the count is written on it, so the denominator is never away from it.
    assert [trace.text[0] for trace in bars] == ["16", "0"]


def test_a_rating_panel_draws_every_answer_not_only_the_box():
    """The answers are on a six-point scale: a box over 18 ratings that take
    six values hides how many sit on one of them."""
    boxes = [trace for trace in _answers_figure().data if trace.type == "box"]
    assert len(boxes) == 4  # one rated outcome and the regret panel, two displays each
    for box in boxes:
        assert box.boxpoints == "all"


def test_a_rated_outcome_panel_says_which_direction_is_better():
    up = charts.study_answers_figure(
        {"graph": 1, "videos": 1},
        _ratings(),
        [{"key": "effort", "label": "Confidence", "better": "up"}],
        _regret_rows(),
    )
    assert "Confidence \u2191" in [a.text for a in up.layout.annotations][1]


def test_the_regret_panel_carries_the_task_and_the_option_on_every_dot():
    figure = _answers_figure()
    regret = [trace for trace in figure.data if trace.type == "box"][-2:]
    assert tuple(regret[0].text) == ("Set the table: Milk left",)
    assert tuple(regret[1].text) == ("Cook the meal: Lid",)


def _effects() -> list[dict]:
    return [
        {
            "outcome": "temporal_demand",
            "label": "Temporal demand",
            "better": "down",
            "mean_graph": 38.7,
            "mean_videos": 64.8,
            "difference": -26.1,
            "lower": -38.4,
            "upper": -14.0,
        },
        {
            "outcome": "regret",
            "label": "Success rate regret",
            "better": "down",
            "mean_graph": 7.9,
            "mean_videos": 7.0,
            "difference": 0.9,
            "lower": -6.1,
            "upper": 8.5,
        },
    ]


def test_the_effect_table_is_the_estimates_and_the_intervals_of_the_difference():
    """The unit is in the caption rather than in a column of two repeated
    strings, and the interval header names which number it is around."""
    frame = charts.effect_table(_effects())
    assert list(frame.columns) == [
        "Outcome",
        "Graph",
        "Videos",
        "Difference",
        "95% interval of difference",
    ]
    assert list(frame["Outcome"]) == ["Temporal demand \u2193", "Success rate regret \u2193"]
    assert list(frame["95% interval of difference"]) == ["-38.4 to -14.0", "-6.1 to +8.5"]


# ── Progress estimation ──────────────────────────────────────────────────────

_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

_METRICS = [
    {"key": "voc", "label": "VOC", "higher_better": True},
    {"key": "mae", "label": "MAE", "higher_better": False},
]
_METHODS = [
    {"key": "graph_stage", "label": "Behavior graph", "ours": True},
    {"key": "mlp", "label": "MLP on policy embeddings", "ours": False},
    {"key": "robometer_4b", "label": "ROBOMETER-4B", "ours": False},
]


def _rollout(idx: int = 7, success: bool = False) -> dict:
    return {
        "idx": idx,
        "success": success,
        "curves": {
            "graph_stage": list(_GRID),
            "mlp": [min(1.0, v + 0.2) for v in _GRID],
            "robometer_4b": [min(1.0, v + 0.3) for v in _GRID],
        },
    }


def _curve_entry() -> dict:
    return {
        "dataset": "cooking_baseline",
        "domain": "Real-world",
        "episodes": 60,
        "successes": 14,
        "rollouts": [_rollout(0, False), _rollout(5, True)],
    }


def test_the_chart_draws_one_rollout_and_no_aggregate():
    """A progress estimate is per rollout and the estimators are
    deterministic, so a median across rollouts and a band around it are
    quantities the method never produces."""
    figure = charts.progress_curve_figure(_rollout(), _GRID, _METHODS)
    assert not [trace for trace in figure.data if trace.fill == "toself"]
    lines = [trace for trace in figure.data if trace.mode == "lines"]
    # Three methods and the reference diagonal, nothing else.
    assert len(lines) == 4
    assert lines[-1].name == "Elapsed fraction"


def test_the_x_axis_is_the_fraction_the_video_cursor_reads():
    """`video_curve_player` drives the cursor from currentTime/duration, so a
    chart under a clip has to be on [0, 1] of the rollout."""
    figure = charts.progress_curve_figure(_rollout(), _GRID, _METHODS)
    assert tuple(figure.layout.xaxis.range) == (0, 1)
    assert figure.layout.xaxis.title.text == "Fraction of the rollout"


def test_the_three_baselines_are_told_apart_by_dash_as_well_as_by_lightness():
    """Three greys on one chart are three greys."""
    figure = charts.progress_curve_figure(_rollout(), _GRID, _METHODS)
    lines = {
        trace.name: (trace.line.color, trace.line.dash)
        for trace in figure.data
        if trace.mode == "lines" and trace.name != "Elapsed fraction"
    }
    baselines = [v for name, v in lines.items() if name != "Behavior graph"]
    assert len({dash for _, dash in baselines}) == len(baselines)
    assert len({colour for colour, _ in baselines}) == len(baselines)


def test_no_baseline_borrows_the_reference_lines_dash():
    figure = charts.progress_curve_figure(_rollout(), _GRID, _METHODS)
    reference = next(trace for trace in figure.data if trace.name == "Elapsed fraction")
    assert reference.line.dash == "dot"
    assert "dot" not in [
        trace.line.dash for trace in figure.data if trace.mode == "lines" and trace is not reference
    ]


def test_a_method_with_no_curve_for_this_rollout_is_absent_not_flat():
    rollout = _rollout()
    del rollout["curves"]["mlp"]
    figure = charts.progress_curve_figure(rollout, _GRID, _METHODS)
    assert "MLP on policy embeddings" not in [trace.name for trace in figure.data]


def test_a_rollout_is_named_by_its_index_and_outcome():
    """Both, because the outcome is what the diagonal means on that rollout."""
    assert charts.rollout_label(_rollout(3, True)) == "Rollout 003 · success"
    assert charts.rollout_label(_rollout(12, False)) == "Rollout 012 · failure"


def test_a_collection_is_found_by_name():
    curves = {"grid": _GRID, "datasets": [_curve_entry()]}
    assert charts.curve_entry(curves, "cooking_baseline")["successes"] == 14
    assert charts.curve_entry(curves, "not_here") is None


def test_collections_are_listed_in_the_tables_own_order_of_sources():
    """A dropdown of 43 names is easier to find a thing in when it runs in the
    same order as the table under it, rather than alphabetically."""
    curves = {
        "grid": _GRID,
        "datasets": [
            {"dataset": "cooking", "domain": "Real-world", "series": []},
            {"dataset": "aaa_mimic", "domain": "MimicGen", "series": []},
            {"dataset": "libero_goal", "domain": "LIBERO", "series": []},
        ],
    }
    domains = [{"label": "LIBERO"}, {"label": "Real-world"}, {"label": "MimicGen"}]
    assert [e["dataset"] for e in charts.curve_datasets(curves, domains)] == [
        "libero_goal",
        "cooking",
        "aaa_mimic",
    ]


def test_the_table_the_paper_carries_is_still_buildable():
    """The page no longer draws it -- twenty rows of four metrics was a wall
    rather than a result -- but it is the paper's table and the builder stays
    tested, since the prose beside the curves quotes its average row.

    This test had no body at all: a docstring asserting that something stays
    tested, and nothing testing it. It could not fail, and it read as coverage
    the suite did not have.
    """
    means = [
        {"domain": "LIBERO", "method": "graph_stage", "datasets": 6, "voc": 0.91, "mae": 0.08},
        {"domain": "LIBERO", "method": "mlp", "datasets": 6, "voc": 0.86, "mae": 0.10},
        {"domain": "Real-world", "method": "graph_stage", "datasets": 19, "voc": 0.95, "mae": 0.06},
        {"domain": "Real-world", "method": "mlp", "datasets": 19, "voc": 0.74, "mae": 0.15},
    ]
    frame = charts.progress_table(means, _METRICS, _METHODS)

    # One row per (domain, method), each source carrying its dataset count --
    # the count is what stops a domain mean resting on fewer datasets than it
    # appears to.
    assert list(frame["Source"]) == [
        "LIBERO (6)",
        "LIBERO (6)",
        "Real-world (19)",
        "Real-world (19)",
    ]
    assert list(frame["VOC \u2191"]) == [0.91, 0.86, 0.95, 0.74]
    # Direction is per metric: highest VOC and LOWEST MAE win their block.
    assert charts.best_cells(frame, _METRICS) == {
        (0, "VOC \u2191"): True,
        (0, "MAE \u2193"): True,
        (2, "VOC \u2191"): True,
        (2, "MAE \u2193"): True,
    }


def test_the_table_bolds_the_best_cell_in_each_block_by_its_metrics_direction():
    means = [
        {"domain": "LIBERO", "method": "graph_stage", "datasets": 1, "voc": 0.9, "mae": 0.08},
        {"domain": "LIBERO", "method": "mlp", "datasets": 1, "voc": 0.8, "mae": 0.10},
    ]
    frame = charts.progress_table(means, _METRICS, _METHODS)
    assert list(frame["Source"]) == ["LIBERO (1)", "LIBERO (1)"]
    best = charts.best_cells(frame, _METRICS)
    assert best == {(0, "VOC ↑"): True, (0, "MAE ↓"): True}


def test_the_table_leaves_a_metric_no_dataset_defined_empty():
    means = [{"domain": "LIBERO", "method": "graph_stage", "datasets": 1, "voc": 0.9, "mae": None}]
    frame = charts.progress_table(means, _METRICS, _METHODS)
    assert pd.isna(frame.loc[0, "MAE ↓"])
    assert charts.best_cells(frame, _METRICS) == {(0, "VOC ↑"): True}
