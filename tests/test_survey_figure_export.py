"""The paper figure the analytics page exports.

What is worth testing here is not how it looks — that is a judgement made by
looking at it — but what goes into it and what comes out. Two things go wrong
silently: a rating that names no display gets pooled into a graph-versus-videos
box (comparing a condition with a modality), and a PDF gets written with Type 3
fonts, which is what an IEEE submission is bounced for and which is invisible
until the submission is bounced.
"""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from behavior_graphs.viz.streamlit.user_study import figure_export  # noqa: E402

# pytest puts this file's directory on sys.path (tests/ is not a package).
from test_survey_analytics import (  # noqa: E402
    CONTRASTIVE,
    ORDER_EFFECT,
    PRODUCTION,
    PRODUCTION_ALLOCATING,
    THREE_STAGE,
)

#: A three-stage response that also rated confidence, which the shared
#: fixtures do not: it is the one dimension of the three that improves *upward*,
#: so it is the one a single shared arrow would get backwards.
RATED_CONFIDENCE = {
    **THREE_STAGE,
    "participant_id": "p-confident",
    "nasa_tlx": {
        task: {"temporal_demand": 40, "effort": 40, "confidence": 70}
        for task in ("cooking", "set_table", "prepare_tray")
    },
}

#: Two scored choices, one per display, on assigned stages.
DECISIONS = [
    {"phase": "graph", "task": "t", "task_label": "Task", "regret": 2.0, "assignment": "assigned"},
    {"phase": "videos", "task": "t", "task_label": "Task", "regret": 9.0, "assignment": "assigned"},
]


def test_only_the_freely_chosen_stage_counts_as_a_preference():
    """The three-stage layout assigns two displays and lets the participant
    pick the third; counting the assigned ones would report the draw."""
    counts = figure_export.preference_counts([THREE_STAGE])

    # prepare_tray is the chosen stage, and it was read as videos.
    assert counts == {"graph": 0, "videos": 1}


def test_a_display_nobody_picked_is_a_zero_and_not_a_missing_bar():
    counts = figure_export.preference_counts([THREE_STAGE])

    assert set(counts) == set(figure_export.PHASE_ORDER)


def test_a_layout_with_no_free_choice_offers_no_preference():
    assert figure_export.preference_counts([PRODUCTION, CONTRASTIVE, ORDER_EFFECT]) == {
        "graph": 0,
        "videos": 0,
    }


def test_ratings_are_split_by_the_display_they_were_given_for():
    scores = figure_export.rating_scores([THREE_STAGE])

    # cooking was read as the graph, the other two as videos.
    assert scores["temporal_demand"] == {"graph": [30.0], "videos": [50.0, 40.0]}


def test_a_session_rated_once_contributes_no_box():
    """The production layout rates the whole session and never says which
    display it saw, so its 40 is not a graph rating or a videos one."""
    scores = figure_export.rating_scores([PRODUCTION])

    assert scores["effort"] == {"graph": [], "videos": []}


def test_every_layout_that_rated_per_display_is_read():
    scores = figure_export.rating_scores([CONTRASTIVE, THREE_STAGE])

    assert sorted(scores["temporal_demand"]["graph"]) == [30.0, 40.0]
    assert sorted(scores["temporal_demand"]["videos"]) == [40.0, 40.0, 50.0]


def test_only_the_dimensions_the_figure_draws_are_collected():
    scores = figure_export.rating_scores([PRODUCTION, CONTRASTIVE, THREE_STAGE])

    assert set(scores) == set(figure_export.PAPER_DIMENSIONS)
    # The production layout rates mental demand and the study does not ask it
    # at all — asking for it is how the figure lost a panel to an empty axis.
    assert "mental_demand" not in scores


def test_a_panel_is_drawn_per_answer_that_was_actually_given():
    """The fixtures rate two of the three dimensions, so a four-panel figure
    over them would carry one panel of nothing."""
    fig = figure_export.study_figure([THREE_STAGE])

    try:
        labels = [ax.get_xlabel() for ax in fig.axes]
        titles = [ax.get_title() for ax in fig.axes]
        y_labels = [ax.get_ylabel() for ax in fig.axes]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert labels == [
        "(a) Chosen display",
        "(b) Temporal demand $\\downarrow$",
        "(c) Effort $\\downarrow$",
    ]
    # Under the axes, where a subcaption goes and where the LaTeX caption is
    # read from — not as titles at the top of the figure.
    assert titles == ["", "", ""]
    # The rating axis is named and nothing more. A rotated label's length runs
    # along the vertical axis, so every word in it costs the figure height — the
    # scale runs 0 to 100 on the ticks, and which way it runs is the caption's
    # to say, where it costs nothing.
    assert y_labels[1] == "Rating"
    assert y_labels[2] == ""  # one axis for the three, labelled once on the left
    # Said once, on the leftmost box panel.
    assert y_labels[2] == ""


def test_the_panel_letters_follow_the_panels_that_survived():
    """(a) is whatever is leftmost, so a caption's (b) points at the second
    panel drawn rather than at the second panel imagined."""
    fig = figure_export.study_figure([CONTRASTIVE])

    try:
        labels = [ax.get_xlabel() for ax in fig.axes]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    # No free choice in the contrastive layout, so temporal demand leads.
    assert labels == [
        "(a) Temporal demand $\\downarrow$",
        "(b) Effort $\\downarrow$",
    ]


def test_every_panel_that_draws_a_metric_says_which_way_it_improves():
    """For a reader who does not already know that less perceived effort is
    the better display and more confidence is.

    Under the panel name, and said per panel even though the three rating
    panels share one y axis — precisely because they do not share a direction.
    One arrow on the shared axis would be right for two of them and backwards
    for the third.
    """
    fig = figure_export.study_figure([RATED_CONFIDENCE], DECISIONS)

    try:
        labels = [ax.get_xlabel() for ax in fig.axes]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert labels == [
        # No note: it counts participants, and neither direction of a count is
        # better than the other.
        "(a) Chosen display",
        "(b) Temporal demand $\\downarrow$",
        "(c) Effort $\\downarrow$",
        "(d) Confidence $\\uparrow$",
        # Named after its unit rather than after what it is a decision about:
        # the bar is success-rate points left on the table, which is the thing
        # a reader has to hold and cannot get from "decision".
        "(e) Success rate regret $\\downarrow$",
    ]


def test_the_arrow_is_mathtext_and_not_a_unicode_arrow():
    """U+2193 is not carried by every face this figure may resolve to, and a
    missing glyph is drawn as a box — in the PDF, silently. Mathtext is set in
    STIX, which has both arrows; it is why the difference panel's minus is
    ``$-$`` too."""
    fig = figure_export.study_figure([RATED_CONFIDENCE], DECISIONS)

    try:
        labels = " ".join(ax.get_xlabel() for ax in fig.axes)
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert "$\\downarrow$" in labels and "$\\uparrow$" in labels
    assert "\u2193" not in labels and "\u2191" not in labels


def test_the_panels_that_hold_every_dimension_at_once_say_it_per_tick():
    """The difference panel draws all three outcomes on one axis, so there is
    no panel name to hang a direction on: below zero is the better display for
    two of its three ticks and the worse one for confidence."""
    fig = figure_export.adjusted_study_figure([THREE_STAGE], _effects())

    try:
        ticks = [label.get_text() for label in fig.axes[1].get_xticklabels()]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert ticks == [
        "Temporal demand $\\downarrow$",
        "Effort $\\downarrow$",
        "Confidence $\\uparrow$",
    ]


def test_the_difference_panels_unit_fits_the_panel_it_is_drawn_in():
    """One line of "Graph $-$ Videos (points)" is 79 pt of rotated text in a
    76.5 pt panel, so it was drawn clipped in every export of this figure until
    it was measured. Two lines, like the regret panels' label — see
    `FIGURE_HEIGHT_IN` for the budget and how to re-measure it."""
    fig = figure_export.adjusted_study_figure([THREE_STAGE], _effects())

    try:
        label = fig.axes[1].get_ylabel()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert "Graph $-$ Videos" in label
    assert "(points)" in label
    assert label.count("\n") == 1


def test_responses_that_answered_none_of_it_export_nothing():
    """Rather than a figure of four empty axes, which reads as a broken plot."""
    assert figure_export.study_figure([PRODUCTION, PRODUCTION_ALLOCATING]) is None
    assert figure_export.study_figure_files([PRODUCTION_ALLOCATING]) is None


def test_the_export_is_a_pdf_that_embeds_its_fonts_as_truetype():
    """Type 3 is matplotlib's fallback when the resolved face is not a
    TrueType one, and it is what IEEE PDF eXpress rejects."""
    files = figure_export.study_figure_files([THREE_STAGE, CONTRASTIVE])

    assert files is not None
    assert files["pdf"].startswith(b"%PDF")
    assert files["png"].startswith(b"\x89PNG")
    assert b"/FontFile2" in files["pdf"]
    assert b"/Type3" not in files["pdf"]


def test_the_same_responses_export_the_same_bytes():
    """No creation date in the PDF: a figure that changes every time it is
    exported cannot be diffed against the one in the paper."""
    first = figure_export.study_figure_files([THREE_STAGE])
    second = figure_export.study_figure_files([THREE_STAGE])

    assert first is not None and second is not None
    assert first["pdf"] == second["pdf"]


def test_exporting_leaves_no_open_figures_behind():
    """The page rebuilds this on every change of the data; leaking one figure
    per rebuild is how a long-running Streamlit process runs out of them."""
    import matplotlib.pyplot as plt

    plt.close("all")
    figure_export.study_figure_files([THREE_STAGE])

    assert plt.get_fignums() == []


def test_drawing_the_figure_does_not_restyle_the_process():
    """The style is applied through an rc_context because this module is
    imported into an app that draws other matplotlib figures."""
    import matplotlib as mpl

    before = mpl.rcParams["font.family"], mpl.rcParams["pdf.fonttype"]
    figure_export.study_figure_files([THREE_STAGE])

    assert (mpl.rcParams["font.family"], mpl.rcParams["pdf.fonttype"]) == before


def test_tied_answers_are_spread_rather_than_drawn_on_top_of_each_other():
    """The scale has six points, so a strip over a box is mostly ties, and a
    point hidden under another point is a participant the figure does not
    show."""
    offsets = figure_export._fanned([40.0, 40.0, 40.0, 80.0], spread=0.06)

    assert len(set(offsets[:3])) == 3
    # Symmetric about the box, and a value held by one person stays centred.
    assert sum(offsets[:3]) == pytest.approx(0.0)
    assert offsets[3] == 0.0


def test_the_rating_panels_share_one_y_axis():
    """A reader compares a box in one panel against a box in another, so the
    three are one 0–100 scale drawn once — linked, not merely set to matching
    limits, so a later change to the scale cannot reach one and miss its
    neighbours. The preference panel counts participants and stays out of it.
    """
    rated_all_three = dict(
        THREE_STAGE,
        nasa_tlx={
            task: {"temporal_demand": 40, "effort": 60, "confidence": 80}
            for task in THREE_STAGE["task_order"]
        },
    )
    fig = figure_export.study_figure([rated_all_three])

    try:
        preference, first, second, third = fig.axes
        shared = first.get_shared_y_axes()
        assert shared.joined(first, second) and shared.joined(first, third)
        assert not shared.joined(first, preference)
        # Drawn once, on the left: the panels beside it carry neither the
        # numbers nor the spine, and read against its gridlines.
        fig.canvas.draw()
        assert [tick.get_text() for tick in first.get_yticklabels()] == [
            "0",
            "20",
            "40",
            "60",
            "80",
            "100",
        ]
        assert [tick.get_text() for tick in second.get_yticklabels()] == []
        assert [tick.get_text() for tick in third.get_yticklabels()] == []
        assert first.spines["left"].get_visible()
        assert not second.spines["left"].get_visible()
        assert not third.spines["left"].get_visible()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


# ── The adjusted figure ───────────────────────────────────────────────────────
# Its panels are model output, so these tests hand it the summaries a fit would
# return rather than fitting anything: what can go wrong here is a panel that
# quietly stops being drawn, not an interval that comes out wrong.


def _effects() -> list[dict]:
    return [
        {
            "outcome": outcome,
            "mean": 5.0,
            "lower": -2.0,
            "upper": 12.0,
            "marginal_means": {
                modality: {"mean": 50.0, "lower": 40.0, "upper": 60.0}
                for modality in ("graph", "videos")
            },
        }
        for outcome in ("temporal_demand", "effort", "confidence")
    ]


#: Less regret with the graph: 8 success-rate points less left on the table,
#: with an interval that still touches zero. NEGATIVE deliberately -- regret
#: improves downward, so a positive fixture here would assert the opposite of
#: what it reads as, and this number has already been flipped once by a
#: reframing that did not reach the tests.
_REGRET_EFFECT = {"mean": -8.0, "lower": -16.0, "upper": 1.0, "n": 42}


def _titles(fig) -> list[str]:
    # The panel name is the x label, not the title: see `_label`.
    return [ax.get_xlabel() for ax in fig.axes]


def test_the_adjusted_figure_carries_a_decision_panel_where_the_choices_are_scored():
    """The decision is the one outcome of the study that is not self-reported,
    so where it can be scored it belongs beside the ratings it explains."""
    fig = figure_export.adjusted_study_figure(
        [THREE_STAGE], _effects(), regret_effect=_REGRET_EFFECT
    )

    try:
        assert len(fig.axes) == 3
        assert _titles(fig) == [
            "(a) Chosen display",
            "(b) Graph $-$ Videos",
            # The name every figure and table gives this quantity, from
            # `STUDY_PANEL_TITLES` — see the raw figure's panel (e).
            "(c) Success rate regret $\\downarrow$",
        ]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_a_deployment_without_the_retrained_rollouts_still_gets_the_other_panels():
    """Scoring a choice needs every option's retrained policy on disk. Where
    they are not mounted the regret panel cannot be drawn — and the figure has
    to lose that panel only, not the preference panel beside it."""
    fig = figure_export.adjusted_study_figure([THREE_STAGE], _effects(), regret_effect=None)

    try:
        assert _titles(fig) == ["(a) Chosen display", "(b) Graph $-$ Videos"]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_ratings_view_leaves_the_decision_effect_out():
    """Its middle panel is already two series wide; a contrast panel beside two
    absolute-rating series reads as a third series rather than a difference."""
    fig = figure_export.adjusted_study_figure(
        [THREE_STAGE], _effects(), regret_effect=_REGRET_EFFECT, mode="ratings"
    )

    try:
        assert _titles(fig) == ["(a) Chosen display", "(b) Adjusted ratings"]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_decision_panel_draws_the_interval_it_was_given():
    fig = figure_export.adjusted_study_figure(
        [THREE_STAGE], _effects(), regret_effect=_REGRET_EFFECT
    )

    try:
        decision_panel = fig.axes[2]
        lower, upper = decision_panel.get_ylim()
        # The whole interval fits, with the no-effect line kept in view: an
        # interval clipped at the axes edge reads as a shorter one.
        assert lower < _REGRET_EFFECT["lower"] and upper > max(_REGRET_EFFECT["upper"], 0.0)
        assert [tick.get_text() for tick in decision_panel.get_xticklabels()] == ["Data collection"]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_the_difference_figure_is_one_tick_per_outcome_with_the_decision_last():
    """The four estimates the paper's sentences quote, and nothing else. The
    decision goes last because it is the outcome the ratings are the subjective
    half of, which is the order every other figure here draws.

    Ticks on x and points on y, like every one of those figures: the y axis is
    the score everywhere else on the page, and one panel that transposed it
    would make a reader learn the same quantity twice. The two long names wrap
    rather than abbreviate -- four ticks share about 200 pt here.
    """
    fig = figure_export.adjusted_differences_figure(_effects(), regret_effect=_REGRET_EFFECT)

    try:
        (ax,) = fig.axes
        ticks = [label.get_text() for label in ax.get_xticklabels()]
        # A hairline between the third tick and the fourth, because the unit
        # changes there: rating points to the left of it, success-rate points
        # to the right.
        assert [
            list(line.get_xdata())
            for line in ax.lines
            if len(line.get_xdata()) == 2 and len(set(line.get_xdata())) == 1
        ] == [[2.5, 2.5]]
        assert "Graph $-$ Videos" in ax.get_ylabel()
        assert "(points)" in ax.get_ylabel()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert ticks == [
        "Temporal\ndemand $\\downarrow$",
        "Effort $\\downarrow$",
        "Confidence $\\uparrow$",
        "Decision\nregret $\\downarrow$",
    ]


def test_the_difference_figures_arrows_point_up_or_down():
    """The same arrows as the panels of levels, because the axis is the same
    axis: down for the two where less is the better display and for regret, up
    for confidence. Never a sideways arrow -- this figure's better direction is
    up or down, like everything else here."""
    fig = figure_export.adjusted_differences_figure(_effects(), regret_effect=_REGRET_EFFECT)

    try:
        (ax,) = fig.axes
        ticks = [label.get_text() for label in ax.get_xticklabels()]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert len([tick for tick in ticks if "downarrow" in tick]) == 3
    upward = [tick for tick in ticks if "uparrow" in tick]
    assert len(upward) == 1 and upward[0].startswith("Confidence")
    assert not any("leftarrow" in tick or "rightarrow" in tick for tick in ticks)
    # And the word is not in the figure: the caption says once what the arrow
    # means, where four panels saying "better" is four times the ink.
    assert not any("better" in tick for tick in ticks)


def test_the_difference_figure_keeps_zero_in_view():
    """Every mark is read against zero — an interval that clears it is a
    difference the model separates from none — so an axis scaled to the marks
    alone, which for these three would start at -38, would leave the reference
    off the figure."""
    fig = figure_export.adjusted_differences_figure(_effects())

    try:
        (ax,) = fig.axes
        low, high = ax.get_ylim()
        # A rule spans the panel in two points; the marker line and the error
        # bars' caps carry one point per tick.
        horizontals = [list(line.get_ydata()) for line in ax.lines if len(line.get_ydata()) == 2]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert low < 0.0 < high
    # And drawn, not merely in range.
    assert horizontals == [[0.0, 0.0]]


def test_the_difference_figure_fits_one_column():
    """It exists to sit beside the paragraph that quotes it, which the
    five-panel figures cannot. Its height is theirs — it is the same kind of
    panel and the figures are laid out against each other — so what makes it
    compact is the width and the one panel: a quarter of the page area."""
    fig = figure_export.adjusted_differences_figure(_effects(), regret_effect=_REGRET_EFFECT)

    try:
        width, height = fig.get_size_inches()
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert width == pytest.approx(figure_export.SINGLE_COLUMN_WIDTH_IN)
    assert height == pytest.approx(figure_export.FIGURE_HEIGHT_IN)
    assert width * height < 0.5 * figure_export.DOUBLE_COLUMN_WIDTH_IN * (
        figure_export.FIGURE_HEIGHT_IN
    )


def test_a_deployment_that_cannot_score_a_choice_gets_the_rating_rows():
    """Scoring a choice needs every option's retrained policy on disk. Without
    them the figure loses the decision tick and keeps the three it can draw --
    and with it the hairline, which marks a change of unit that is no longer in
    the figure."""
    fig = figure_export.adjusted_differences_figure(_effects(), regret_effect=None)

    try:
        (ax,) = fig.axes
        ticks = [label.get_text() for label in ax.get_xticklabels()]
        # Two-point lines are the rules: the zero one, and the hairline before
        # the decision tick if there is one. The hairline is the vertical of
        # the two.
        verticals = [
            list(line.get_xdata())
            for line in ax.lines
            if len(line.get_xdata()) == 2 and len(set(line.get_xdata())) == 1
        ]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)

    assert len(ticks) == 3
    assert not any(tick.startswith("Decision") for tick in ticks)
    assert verticals == []


def test_nothing_to_contrast_draws_nothing():
    """As the other figures here do, rather than one empty axis."""
    assert figure_export.adjusted_differences_figure([]) is None
    assert figure_export.adjusted_differences_figure_files([]) is None


def test_the_difference_figure_exports_a_pdf_with_its_fonts_embedded():
    files = figure_export.adjusted_differences_figure_files(
        _effects(), regret_effect=_REGRET_EFFECT
    )

    assert files is not None
    assert files["pdf"].startswith(b"%PDF")
    assert files["png"].startswith(b"\x89PNG")


def test_an_unknown_mode_is_refused_before_anything_is_drawn():
    """The mode comes off a radio today, but the raise is what keeps a typo in
    a script from silently getting the default figure."""
    with pytest.raises(ValueError, match="Unknown adjusted-study figure mode"):
        figure_export.adjusted_study_figure([THREE_STAGE], _effects(), mode="rating")


def test_responses_that_fit_no_model_draw_nothing_rather_than_empty_axes():
    assert figure_export.adjusted_study_figure([THREE_STAGE], []) is None
    assert figure_export.adjusted_study_figure_files([THREE_STAGE], []) is None


def test_the_decision_panel_leaves_out_the_stage_they_chose():
    """`randomized_only` has to reach the decision panel too, not just the
    boxes. The third stage's display was the participant's own pick and almost
    everyone picked the graph, so a choice made in it belongs to no randomized
    comparison -- and on the live store pooling those in reversed which display
    led to the better decisions."""
    rows = [
        {"phase": "graph", "task": "t", "regret": 2.0, "assignment": "assigned"},
        {"phase": "graph", "task": "t", "regret": 30.0, "assignment": "chosen"},
        {"phase": "videos", "task": "t", "regret": 10.0, "assignment": "assigned"},
    ]

    restricted = figure_export.study_figure([THREE_STAGE], rows, randomized_only=True)
    pooled = figure_export.study_figure([THREE_STAGE], rows)
    try:
        # One dot per randomized answer, and nothing from the chosen stage.
        assert _strip_values(restricted.axes[-1]) == [[2.0], [10.0]]
        assert [label.get_text() for label in restricted.axes[-1].get_xticklabels()] == [
            "Graph\nn=1",
            "Videos\nn=1",
        ]
        # Unrestricted, the chosen stage's 30 joins the graph's pile and drags
        # it above the videos it is being compared against -- the reversal this
        # flag exists to prevent.
        assert _strip_values(pooled.axes[-1]) == [[2.0, 30.0], [10.0]]
    finally:
        import matplotlib.pyplot as plt

        plt.close(restricted)
        plt.close(pooled)


def _strip_values(ax) -> list[list[float]]:
    """The dots a box panel carries — one per answer, per display, in order.

    Scatter collections only: an error bar drawn with ``fmt="none"`` is a
    ``LineCollection`` that answers ``get_offsets()`` with a placeholder, so a
    mean panel otherwise reports a phantom dot at zero.
    """
    from matplotlib.collections import PathCollection

    return [
        sorted(round(float(y), 2) for _x, y in collection.get_offsets())
        for collection in ax.collections
        if isinstance(collection, PathCollection)
    ]


# ── The choices, as bars for the policy comparison ───────────────────────────


def _share_rows(**picks: dict[str, int]) -> list[dict]:
    """``choice_share_rows``-shaped rows: ``phase={dataset: picks}``."""
    rows = []
    for phase, counts in picks.items():
        answers = sum(counts.values())
        rows += [
            {
                "task": "set_table",
                "task_label": "Set the table",
                "baseline": "set_table_baseline_pretrained_dit",
                "phase": phase,
                "dataset": dataset,
                "choice_label": dataset,
                "picks": count,
                "answers": answers,
                "share": count / answers if answers else 0.0,
            }
            for dataset, count in counts.items()
        ]
    return rows


SET_TABLE_GROUP = (
    "set_table_baseline_pretrained_dit",
    "set_table_plusleft_pretrained_dit",
    "set_table_plusmiddle_pretrained_dit",
    "set_table_plusright_pretrained_dit",
)


def test_the_bars_are_keyed_by_the_label_the_figure_draws():
    """Not by the dataset name: the figure has thrown those away by the time it
    draws, and a key that does not match a violin's label is a bar the figure
    silently reports as "not offered"."""
    series = figure_export.choice_series(
        _share_rows(
            graph={
                "set_table_plusleft_pretrained_dit": 3,
                "set_table_plusmiddle_pretrained_dit": 1,
                "set_table_plusright_pretrained_dit": 0,
            }
        ),
        SET_TABLE_GROUP,
    )

    (graph,) = series
    assert graph.label == "Graph"
    assert graph.color == figure_export.PHASE_COLORS["graph"]
    assert graph.picks == {"plus milk left": 3, "plus milk middle": 1, "plus milk right": 0}
    assert graph.total == 4
    # The baseline is not a key at all, so the figure draws a dash for it.
    assert "baseline" not in graph.picks


def test_a_policy_outside_the_group_is_not_drawn_in_its_panel():
    """The group is the source of truth for what the panel holds — an arm of
    another task landing in it would be a bar under somebody else's violin."""
    rows = _share_rows(graph={"set_table_plusleft_pretrained_dit": 2, "prepare_tray_plustop": 5})

    (graph,) = figure_export.choice_series(rows, SET_TABLE_GROUP)

    assert graph.picks == {"plus milk left": 2}


def test_the_displays_come_out_graph_first():
    """The paper meets the graph on the left of every panel and on the top row
    of every strip; the order is :data:`regret.PHASE_ORDER`'s."""
    rows = _share_rows(
        videos={"set_table_plusleft_pretrained_dit": 1},
        graph={"set_table_plusright_pretrained_dit": 2},
    )

    assert [series.label for series in figure_export.choice_series(rows, SET_TABLE_GROUP)] == [
        "Graph",
        "Videos",
    ]


def test_a_display_that_answered_nothing_about_this_group_gets_no_series():
    rows = _share_rows(graph={"prepare_tray_plustop": 4})

    assert figure_export.choice_series(rows, SET_TABLE_GROUP) == ()


def test_the_denominator_is_the_displays_own_answers():
    """Read off the rows, not summed from the picks the panel happens to draw:
    a group holding only some of the task's arms still shows shares of every
    answer that display gave, so they do not add up to 100%."""
    rows = _share_rows(
        graph={
            "set_table_plusleft_pretrained_dit": 2,
            "set_table_plusmiddle_pretrained_dit": 3,
            "set_table_plusright_pretrained_dit": 0,
        }
    )

    (graph,) = figure_export.choice_series(
        rows, ("set_table_baseline_pretrained_dit", "set_table_plusleft_pretrained_dit")
    )

    assert graph.total == 5
    assert graph.picks == {"plus milk left": 2}
    assert graph.share("plus milk left") == pytest.approx(0.4)


def test_the_labels_can_be_overridden_the_way_the_page_overrides_them():
    """The comparison page lets its labels be edited, and a figure drawn with
    edited labels has to take its bars' keys from the same place."""
    rows = _share_rows(graph={"set_table_plusleft_pretrained_dit": 1})

    (graph,) = figure_export.choice_series(
        rows, SET_TABLE_GROUP, {"set_table_plusleft_pretrained_dit": "+ milk L"}
    )

    assert graph.picks == {"+ milk L": 1}


# ── Which mark the regret panel wears ────────────────────────────────────────


def _regret_panel(fig):
    """The regret panel of a study figure — the last one drawn."""
    return fig.axes[-1]


def test_the_box_is_the_default_mark():
    """The answers themselves: regret takes one of three values per task, so a
    mean with a standard error around it implies a spread between the piles
    that no participant could have answered."""
    import matplotlib.pyplot as plt

    fig = figure_export.study_figure([RATED_CONFIDENCE], DECISIONS)
    try:
        panel = _regret_panel(fig)
        # A dot per answer, and boxes rather than bars: a boxplot's patch is a
        # `PathPatch`, a bar chart's is a `Rectangle`.
        assert _strip_values(panel) == [[2.0], [9.0]]
        assert {type(patch).__name__ for patch in panel.patches} == {"PathPatch"}
        assert panel.get_ylabel().startswith("Regret")
    finally:
        plt.close(fig)


def test_the_mean_bar_is_still_available():
    """Kept as a choice, not as the default: it is the mark an audience reads
    fastest and the number the paper's sentences quote."""
    import matplotlib.pyplot as plt

    fig = figure_export.study_figure(
        [RATED_CONFIDENCE], DECISIONS, regret_mark=figure_export.MEAN_MARK
    )
    try:
        panel = _regret_panel(fig)
        # One bar per display, at the mean of its answers, and no dot strip.
        assert {type(patch).__name__ for patch in panel.patches} == {"Rectangle"}
        assert [round(bar.get_height(), 2) for bar in panel.patches] == [2.0, 9.0]
        assert _strip_values(panel) == []
        # And it says it is a mean, where the box panel says only "Regret".
        assert panel.get_ylabel().startswith("Mean regret")
    finally:
        plt.close(fig)


def test_the_two_marks_are_two_files():
    """A paper may want to compare them, so neither export can overwrite the
    other's bytes."""
    box = figure_export.study_figure_files([RATED_CONFIDENCE], DECISIONS)
    mean = figure_export.study_figure_files(
        [RATED_CONFIDENCE], DECISIONS, regret_mark=figure_export.MEAN_MARK
    )

    assert box is not None and mean is not None
    assert box["pdf"] != mean["pdf"]


def test_only_the_pooled_panel_of_the_regret_figure_takes_the_mark():
    """A task's four answers are not a mean of anything, so the per-task panels
    are distributions whatever the pooled one is."""
    import matplotlib.pyplot as plt

    # A task `regret.by_task` knows: it keeps the study's own three and drops
    # anything else, so a made-up key draws no per-task panel at all.
    rows = [
        {"phase": "graph", "task": "set_table", "task_label": "Set the table", "regret": 2.0},
        {"phase": "graph", "task": "set_table", "task_label": "Set the table", "regret": 8.0},
        {"phase": "videos", "task": "set_table", "task_label": "Set the table", "regret": 9.0},
    ]

    fig = figure_export.regret_figure(rows, "both", pooled_mark=figure_export.MEAN_MARK)
    try:
        pooled, per_task = fig.axes[0], fig.axes[1]
        # Bars in the pooled panel, dots in the task panel.
        assert [round(bar.get_height(), 2) for bar in pooled.patches] == [5.0, 9.0]
        assert _strip_values(pooled) == []
        assert _strip_values(per_task) == [[2.0, 8.0], [9.0]]
        # The mean keeps its own axis: it is a different quantity from the
        # boxes beside it, and a bar as tall as a box would invite the
        # comparison it cannot support.
        assert pooled.get_ylabel().startswith("Mean regret")
        assert per_task.get_ylabel().startswith("Regret")
        assert pooled.get_ylim() != per_task.get_ylim()
    finally:
        plt.close(fig)


def test_the_pooled_box_shares_the_task_panels_axis():
    """Drawn as a distribution it *is* their answers in one pile, so the panels
    have to be on one scale for the boxes to be comparable."""
    import matplotlib.pyplot as plt

    rows = [
        {"phase": "graph", "task": "set_table", "task_label": "Set the table", "regret": 2.0},
        {"phase": "videos", "task": "set_table", "task_label": "Set the table", "regret": 9.0},
    ]

    fig = figure_export.regret_figure(rows, "both")
    try:
        pooled, per_task = fig.axes[0], fig.axes[1]
        assert pooled.get_ylim() == per_task.get_ylim()
        # Labelled once, on the left.
        assert pooled.get_ylabel().startswith("Regret")
        assert per_task.get_ylabel() == ""
    finally:
        plt.close(fig)


def test_an_unknown_mark_is_a_bug_and_not_a_silent_default():
    with pytest.raises(ValueError, match="regret mark"):
        figure_export.study_figure([RATED_CONFIDENCE], DECISIONS, regret_mark="whiskers")
