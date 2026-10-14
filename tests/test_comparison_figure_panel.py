"""The comparison export panel: it offers the chart above it, as a file.

Driven through ``AppTest`` on a small script rather than on the Policy
Comparison page, which needs rollout datasets on disk and a run of pairwise
exact tests to get as far as a chart. What is tested here is the panel's own
contract — that its options reach the drawing, and that it asks only the
questions a figure has.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz import comparison_figure, paper  # noqa: E402
from behavior_graphs.viz.streamlit.components import comparison_figure_export as cfe  # noqa: E402

#: A script that puts the panel under a three-policy comparison. Inline so the
#: comparison it draws is visible next to what is asserted about it.
SCRIPT = """
from behavior_graphs.stats.policy_comparison import PolicyOutcomes, compare_policies
from behavior_graphs.viz.streamlit.components.comparison_figure_export import (
    comparison_figure_export_panel,
)

def group(*counts):
    return compare_policies([PolicyOutcomes(n, s, 60) for n, s in counts])

GROUPS = int("{groups}")
results = [
    group(("set_table_baseline_pretrained_dit", 41), ("set_table_plusleft_pretrained_dit", 32)),
    group(("cooking_baseline_dit_200", 18), ("cooking_baseline_lid_dit", 9)),
][:GROUPS]
titles = ["Set table", "Cooking"][:GROUPS]
comparison_figure_export_panel(results, titles, key="t")
"""


def _run(groups: int = 1) -> AppTest:
    import streamlit as st

    st.cache_data.clear()
    at = AppTest.from_string(SCRIPT.replace("{groups}", str(groups)), default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def _download(at: AppTest, needle: str):
    return next(b for b in at.get("download_button") if needle in b.label)


def test_it_offers_the_comparison_as_a_pdf_and_a_png():
    at = _run()

    assert "PDF" in _download(at, "PDF").label
    assert "PNG" in _download(at, "PNG").label
    # One preview, which is the PNG of the same figure — so what is approved on
    # the page is what is downloaded.
    assert len(list(at.get("imgs"))) == 1


def test_the_options_are_the_figures_and_not_the_statistics():
    """Nothing here can change a p-value, a letter or an interval: those come in
    already computed, and a control for any of them on an export panel would be
    a second, quieter place to run the analysis."""
    at = _run()

    assert {w.label for w in (*at.slider, *at.checkbox, *at.radio)} == {
        "Width (in)",
        "Height (in)",
        "Letters above the violins",
        "Rollout counts",
    }


def test_one_group_is_not_asked_which_layout_it_wants():
    """A row of one panel is not a layout choice, and the panel labels it "(a)"
    — which is wrong for a figure standing on its own."""
    at = _run(groups=1)

    assert not [r for r in at.radio if r.label == "Layout"]
    assert not [c for c in at.checkbox if c.label == "Shared rate axis"]


def test_several_groups_can_go_out_as_one_row_or_as_separate_figures():
    """Both, because a paper wants each at a different moment: side by side to
    be read against each other, separate when they are discussed a page apart.
    """
    at = _run(groups=2)

    layout = next(r for r in at.radio if r.label == "Layout")
    assert layout.value == "row"
    # A row is one file, and the shared axis is what makes it worth having.
    assert len(list(at.get("download_button"))) == 2
    assert next(c for c in at.checkbox if c.label == "Shared rate axis").value is True

    separate = layout.set_value("each").run()

    # One PDF and one PNG per group, each named after the group.
    assert len(list(separate.get("download_button"))) == 4


def test_the_width_changes_the_exported_file():
    """A download button's payload is not readable through ``AppTest``, but its
    URL is a hash of the content — so a URL that moves is the file changing."""
    at = _run()
    before = _download(at, "PDF").proto.url

    width = next(s for s in at.slider if s.label == "Width (in)")
    assert width.value == cfe.DEFAULT_WIDTH_IN

    assert _download(width.set_value(paper.DOUBLE_COLUMN_WIDTH_IN).run(), "PDF").proto.url != before


def test_the_slider_can_land_on_a_column_width_exactly():
    """The two sizes that have to be exact: a figure placed at ``\\textwidth``
    that is not one of them gets rescaled by the LaTeX, and its 8 pt type stops
    being 8 pt on the page."""
    for column_width in (paper.SINGLE_COLUMN_WIDTH_IN, paper.DOUBLE_COLUMN_WIDTH_IN):
        assert cfe.WIDTH_MIN_IN <= column_width <= cfe.WIDTH_MAX_IN
        steps = (column_width - cfe.WIDTH_MIN_IN) / cfe.WIDTH_STEP_IN
        assert steps == pytest.approx(round(steps)), f"{column_width} in is not on the slider"


def test_it_starts_where_the_drawing_starts():
    """The panel's defaults are the module's, so the figure a reader gets
    without touching anything is the one the drawing was tuned as."""
    assert cfe.DEFAULT_WIDTH_IN == comparison_figure.DEFAULT_WIDTH_IN
    assert cfe.DEFAULT_HEIGHT_IN == comparison_figure.DEFAULT_HEIGHT_IN


def test_the_letters_can_be_turned_off_from_the_panel():
    at = _run()
    before = _download(at, "PDF").proto.url
    letters = next(c for c in at.checkbox if c.label == "Letters above the violins")

    assert letters.value is True
    assert _download(letters.set_value(False).run(), "PDF").proto.url != before


# ── The study's choices under the violins ────────────────────────────────────

#: The same panel with the study's data-collection answers passed in: two
#: displays over the two policies of each group, one of them never picked.
CHOICES_SCRIPT = """
from behavior_graphs.stats.policy_comparison import PolicyOutcomes, compare_policies
from behavior_graphs.viz.comparison_figure import ChoiceSeries
from behavior_graphs.viz.streamlit.components.comparison_figure_export import (
    comparison_figure_export_panel,
)

def group(*counts):
    return compare_policies([PolicyOutcomes(n, s, 60) for n, s in counts])

GROUPS = int("{groups}")
results = [
    group(("set_table_baseline_pretrained_dit", 41), ("set_table_plusleft_pretrained_dit", 32)),
    group(("cooking_baseline_dit_200", 18), ("cooking_baseline_lid_dit", 9)),
][:GROUPS]
titles = ["Set table", "Cooking"][:GROUPS]
choices = [
    (
        ChoiceSeries("Graph", "#ff7f0e", {"set_table_plusleft_pretrained_dit": 4}),
        ChoiceSeries("Videos", "#1f77b4", {"set_table_plusleft_pretrained_dit": 3}),
    ),
    (ChoiceSeries("Graph", "#ff7f0e", {"cooking_baseline_lid_dit": 2}),),
][:GROUPS]
comparison_figure_export_panel(results, titles, choices=choices, key="t")
"""


def _run_choices(groups: int = 1) -> AppTest:
    import streamlit as st

    st.cache_data.clear()
    at = AppTest.from_string(
        CHOICES_SCRIPT.replace("{groups}", str(groups)), default_timeout=180
    ).run()
    assert not at.exception, at.exception
    return at


def test_the_choices_are_offered_only_when_there_are_any():
    """A checkbox for a strip of bars nobody has answers for would be a control
    that does nothing, on a panel whose other controls all change the file."""
    assert not [c for c in _run().checkbox if "choices" in c.label.lower()]

    offered = next(c for c in _run_choices().checkbox if "Study choices" in c.label)
    assert offered.value is True


def test_the_bars_change_the_exported_file():
    at = _run_choices()
    with_bars = _download(at, "PDF").proto.url

    without = next(c for c in at.checkbox if "Study choices" in c.label).set_value(False).run()

    assert _download(without, "PDF").proto.url != with_bars


#: The panel again, with the download component spied on: ``AppTest`` can see a
#: download button's URL but not the name the file is offered under, and the
#: name is the whole point of the test below.
NAMES_SCRIPT = CHOICES_SCRIPT.replace(
    'comparison_figure_export_panel(results, titles, choices=choices, key="t")',
    """
import streamlit as st
import behavior_graphs.viz.streamlit.components.comparison_figure_export as cfe

_real = cfe.paper_figure_downloads
_names = []


def _spy(files, *, filename, **kwargs):
    _names.append(filename)
    return _real(files, filename=filename, **kwargs)


cfe.paper_figure_downloads = _spy
try:
    comparison_figure_export_panel(results, titles, choices=choices, key="t")
finally:
    cfe.paper_figure_downloads = _real
st.session_state["stems"] = _names
""",
)


def test_the_two_versions_are_two_files():
    """A paper carries both — the violins for the results section and the
    violins over the choices for the study's — so one download must not
    overwrite the other."""
    import streamlit as st

    st.cache_data.clear()
    at = AppTest.from_string(NAMES_SCRIPT.replace("{groups}", "1"), default_timeout=180).run()
    assert not at.exception, at.exception

    # One group, so the figure is named after it rather than lettered.
    assert at.session_state["stems"] == ["policy_comparison_Set-table_violins_choices"]

    without = next(c for c in at.checkbox if "Study choices" in c.label).set_value(False).run()

    assert without.session_state["stems"] == ["policy_comparison_Set-table_violins"]


def test_a_figure_with_bars_starts_tall_enough_for_them():
    """The rows are height the violins would otherwise have had, so the default
    is the layout's plus what they take — at the row layout's own 1.2 in, two
    rows leave the rate axis half an inch."""
    at = _run_choices()
    height = next(s for s in at.slider if s.label == "Height (in)")

    assert height.value > comparison_figure.DEFAULT_HEIGHT_IN
