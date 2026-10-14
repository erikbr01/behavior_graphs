"""The export panel: it offers the graph above it, and it offers it as a file.

Driven through ``AppTest`` on a small script rather than on the demo page: the
page needs a swept run on disk and half a minute to build one, and what is being
tested here is the panel's own contract — that the options reach the drawing,
that the buttons are wired to the figure they preview, and that a graph with
nothing left in it says so instead of offering an empty PDF.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz import graph_figure, paper  # noqa: E402
from behavior_graphs.viz.streamlit.components import graph_figure_export as gfe  # noqa: E402

#: A script that puts the panel on a page with a two-behavior graph. Inline so
#: the graph it draws is visible right here next to what is asserted about it.
SCRIPT = """
from behavior_graphs.core.graph import (
    FAILURE_NODE_ID, START_NODE_ID, SUCCESS_NODE_ID, BehaviorGraph, BehaviorNode,
)
from behavior_graphs.viz.streamlit.components.graph_figure_export import (
    graph_figure_export_panel,
)

EXCLUDE = frozenset({int(x) for x in "{excluded}".split(",") if x})

nodes = {
    START_NODE_ID: BehaviorNode(START_NODE_ID, "START", 0, 10),
    SUCCESS_NODE_ID: BehaviorNode(SUCCESS_NODE_ID, "SUCCESS", 0, 6),
    FAILURE_NODE_ID: BehaviorNode(FAILURE_NODE_ID, "FAILURE", 0, 4),
    0: BehaviorNode(0, "Behavior 0", 100, 10),
    1: BehaviorNode(1, "Behavior 1", 80, 7),
}
counts = {
    START_NODE_ID: {0: 10},
    0: {1: 7, FAILURE_NODE_ID: 3},
    1: {SUCCESS_NODE_ID: 6, FAILURE_NODE_ID: 1},
}
probs = {s: {t: c / sum(d.values()) for t, c in d.items()} for s, d in counts.items()}
graph = BehaviorGraph(
    nodes=nodes, transition_counts=counts, transition_probs=probs, num_episodes=10
)
pos = {
    START_NODE_ID: (0.0, 1.0),
    0: (1.0, 1.0),
    1: (2.0, 1.0),
    SUCCESS_NODE_ID: (3.0, 0.0),
    FAILURE_NODE_ID: (3.0, 2.0),
}
# One written-out name among the short ones: the wrap width has nothing to do
# on a graph of "B0" and "FAILURE", since a name breaks on spaces only.
labels = {0: "Slides the tray out from under the shelf"}
graph_figure_export_panel(
    graph, key="t", pos=pos, excluded=EXCLUDE, prob_floor=0.0, labels=labels
)
"""


def _run(excluded: str = "") -> AppTest:
    import streamlit as st

    st.cache_data.clear()
    at = AppTest.from_string(SCRIPT.replace("{excluded}", excluded), default_timeout=120).run()
    assert not at.exception, at.exception
    return at


def _download(at: AppTest, needle: str):
    return next(b for b in at.get("download_button") if needle in b.label)


def test_it_offers_the_graph_as_a_pdf_and_a_png():
    at = _run()

    assert "PDF" in _download(at, "PDF").label
    assert "PNG" in _download(at, "PNG").label
    # The preview is the PNG of the same figure, so what is approved is what is
    # downloaded — one image, not a separate rendering.
    assert len(list(at.get("imgs"))) == 1


def test_the_options_are_the_figures_and_not_the_screens():
    """Width, height, labels: what only a figure has to answer. Everything about
    *which* graph is drawn comes from the caller, so it is not offered here."""
    at = _run()

    labels = {w.label for w in (*at.radio, *at.slider, *at.checkbox)}
    assert labels == {
        "Width (in)",
        "Height (in)",
        "Node size",
        "Label size (pt)",
        "Wrap labels at (pt)",
        "Labels",
        "Arrowheads",
        "Probabilities on edges",
        "Outcome shares",
        "Full node names",
        "Greyscale",
    }


def test_the_width_changes_the_exported_file():
    """Not just the preview. A download button's payload is not readable through
    ``AppTest``, but its URL is a hash of the content — so a URL that moves is
    the file having changed, which is the thing worth asserting."""
    at = _run()
    before = _download(at, "PDF").proto.url

    width = next(s for s in at.slider if s.label == "Width (in)")
    assert width.value == gfe.DEFAULT_WIDTH_IN

    assert _download(width.set_value(paper.SINGLE_COLUMN_WIDTH_IN).run(), "PDF").proto.url != before


def test_the_slider_can_land_on_a_column_width_exactly():
    """The width is free — a graph that reads at four inches should be four
    inches — but the two sizes that have to be *exact* are the column widths: a
    figure placed at ``\\textwidth`` that is not one of them gets rescaled by the
    LaTeX, and its 8 pt type stops being 8 pt on the page. So both have to be a
    whole number of steps from the minimum rather than merely inside the range.
    """
    for column_width in (paper.SINGLE_COLUMN_WIDTH_IN, paper.DOUBLE_COLUMN_WIDTH_IN):
        assert gfe.WIDTH_MIN_IN <= column_width <= gfe.WIDTH_MAX_IN
        steps = (column_width - gfe.WIDTH_MIN_IN) / gfe.WIDTH_STEP_IN
        assert steps == pytest.approx(round(steps)), f"{column_width} in is not on the slider"


def test_a_height_chosen_by_hand_survives_a_nudge_of_the_width():
    """The two sliders should not fight each other: the auto height barely
    depends on the width, so resetting it on every step of a continuous width
    would just take the choice away."""
    at = _run()
    at = next(s for s in at.slider if s.label == "Height (in)").set_value(4.0).run()

    at = next(s for s in at.slider if s.label == "Width (in)").set_value(5.0).run()

    assert next(s for s in at.slider if s.label == "Height (in)").value == 4.0


def test_nothing_left_to_draw_says_so_instead_of_exporting_a_blank_page():
    at = _run(excluded="-2,-4,-5,0,1")

    assert any("Nothing to draw" in element.value for element in at.info)
    assert not list(at.get("download_button"))


def test_the_label_default_follows_the_graph():
    """Three roles here, so their names are worth the ink and the radio starts
    on "everything" — the count is the rule, not the kind of graph."""
    at = _run()

    assert next(r for r in at.radio if r.label == "Labels").value == "all"


def test_a_preset_sets_every_control_it_names():
    """A preset is a whole shape, not a width. Setting the width alone is the
    trap: a graph exported at a third of the page keeps 8 pt labels and an
    inch-and-a-half wrap width, so its names run across their neighbours and its
    nodes touch."""
    at = _run()
    third = gfe.PRESETS["third"]

    at = next(s for s in at.selectbox if s.label == "Size preset").set_value("third").run()

    assert not at.exception, at.exception
    assert next(s for s in at.slider if s.label == "Width (in)").value == third.width_in
    assert next(s for s in at.slider if s.label == "Height (in)").value == third.height_in
    assert next(s for s in at.slider if s.label == "Node size").value == third.node_scale
    assert next(s for s in at.slider if s.label == "Label size (pt)").value == third.label_font_pt
    assert (
        next(s for s in at.slider if s.label == "Wrap labels at (pt)").value == third.label_wrap_pt
    )


def test_every_preset_lands_on_its_own_sliders_steps():
    """Or choosing a preset and then nudging one control would move the rest:
    Streamlit snaps a value off the step grid to the nearest step, silently."""
    grids = {
        "width_in": (gfe.WIDTH_MIN_IN, gfe.WIDTH_STEP_IN),
        "height_in": (1.0, 0.1),
        "node_scale": (0.5, 0.1),
        "label_font_pt": (5.0, 0.5),
        "label_wrap_pt": (24.0, 6.0),
    }
    for name, preset in gfe.PRESETS.items():
        for field, (start, step) in grids.items():
            steps = (getattr(preset, field) - start) / step
            assert steps == pytest.approx(round(steps)), f"{name}.{field} is off its step grid"


def test_a_preset_is_a_starting_point_and_not_a_lock():
    """The controls stay editable, because the presets are the layouts that come
    up often and not the only ones a paper has."""
    at = _run()
    at = next(s for s in at.selectbox if s.label == "Size preset").set_value("third").run()

    nudged = next(s for s in at.slider if s.label == "Label size (pt)").set_value(6.5).run()

    assert next(s for s in nudged.slider if s.label == "Label size (pt)").value == 6.5
    # And nothing else moved with it.
    assert (
        next(s for s in nudged.slider if s.label == "Width (in)").value
        == gfe.PRESETS["third"].width_in
    )


def test_the_wrap_width_reaches_the_drawing():
    """The lever for a figure of written-out names: a name is several times
    wider than the node it belongs to, and where it breaks is what decides
    whether it sits over that node or across four of them."""
    at = _run()
    before = _download(at, "PDF").proto.url

    wrap = next(s for s in at.slider if s.label == "Wrap labels at (pt)")
    assert wrap.value == graph_figure.LABEL_WRAP_PT

    # The graph above carries one written-out name, which is the only kind this
    # setting can act on — a name breaks on spaces, not between characters.
    assert _download(wrap.set_value(24.0).run(), "PDF").proto.url != before


def test_the_outcome_shares_are_on_unless_they_are_turned_off():
    """``P(reach)`` under each outcome is the number the picture cannot give, so
    it is not something a reader of the page has to know to ask for."""
    at = _run()
    shares = next(c for c in at.checkbox if c.label == "Outcome shares")
    before = _download(at, "PDF").proto.url

    assert shares.value is True
    assert _download(shares.set_value(False).run(), "PDF").proto.url != before
