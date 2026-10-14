"""Tests for the NASA-TLX block's rating scale.

The instrument's own scale is a 20-interval visual-analog line, which the study
renders as six points 0-100. What matters here is that an unanswered dimension
stays unanswered: the slider this replaced always reported its handle position,
so "never touched it" and "chose the middle" were the same value in the data.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.user_study.nasa_tlx import (  # noqa: E402
    RATING_POINTS,
    RATING_STEP,
)

_SCRIPT = """
import streamlit as st
from behavior_graphs.viz.streamlit.user_study.nasa_tlx import render_nasa_tlx

st.session_state["out"] = render_nasa_tlx(key_prefix="t", items=["temporal_demand", "effort"])
"""


def _run() -> AppTest:
    return AppTest.from_string(_SCRIPT, default_timeout=30).run()


def test_the_scale_is_zero_to_a_hundred_in_twenties():
    assert RATING_STEP == 20
    assert RATING_POINTS == (0, 20, 40, 60, 80, 100)


def test_each_dimension_offers_exactly_those_points():
    at = _run()

    assert not at.exception
    assert not at.slider, "a slider cannot be left unanswered"
    rated = [r for r in at.radio if r.label in ("Time Taken", "Effort")]
    assert len(rated) == 2
    for radio in rated:
        assert [int(str(option)) for option in radio.options] == list(RATING_POINTS), radio.label


def test_an_untouched_dimension_is_none_not_a_middle_value():
    """What the block that embeds this gates its Continue button on -- see
    ``tests/test_task_questions.py``, which renders that gate."""
    at = _run()

    assert at.session_state["out"] == {"temporal_demand": None, "effort": None}


def test_an_answer_comes_back_as_the_number():
    at = _run()
    at.radio(key="t_effort").set_value(80).run()

    assert at.session_state["out"]["effort"] == 80


def test_the_points_are_bare_numbers_and_the_anchors_sit_beside_them():
    """The anchors are their own text either side of the row, not part of an
    option label — a point is a number, and "0 · Very Low" made it a sentence."""
    at = _run()

    assert [str(option) for option in at.radio(key="t_effort").options] == [
        "0",
        "20",
        "40",
        "60",
        "80",
        "100",
    ]
    anchors = " | ".join(m.value for m in at.markdown)
    assert "Very Low" in anchors and "Very High" in anchors
    assert "Lots of Time" in anchors and "Very Little" in anchors


def test_the_time_item_asks_about_duration_not_pace():
    """Nothing in this study sets a pace, so time *pressure* had nothing to measure."""
    from behavior_graphs.viz.streamlit.user_study.nasa_tlx import _ITEM_DESCRIPTIONS

    asked = _ITEM_DESCRIPTIONS["temporal_demand"]
    assert "time pressure" not in asked
    assert "How much time did answering take" in asked


def test_the_effort_item_asks_about_answering_the_questions():
    """The canonical item asks about work "to accomplish your level of
    performance", which a participant judging a policy does not have."""
    from behavior_graphs.viz.streamlit.user_study.nasa_tlx import _ITEM_DESCRIPTIONS

    assert "How much effort was it to answer the questions" in _ITEM_DESCRIPTIONS["effort"]


def test_the_time_anchors_are_amounts_of_time():
    from behavior_graphs.viz.streamlit.user_study.nasa_tlx import _NASA_TLX_ITEMS

    low, high = next((lo, hi) for key, _, lo, hi in _NASA_TLX_ITEMS if key == "temporal_demand")
    assert (low, high) == ("Very Little", "Lots of Time")


def test_the_question_is_the_heading_and_is_not_repeated():
    """It was a short title with the question restated under it as a caption —
    "Time Taken", then "How much time did answering take?" — the same thing
    twice, the second time in smaller type."""
    from behavior_graphs.viz.streamlit.user_study.nasa_tlx import _ITEM_DESCRIPTIONS

    at = _run()

    said = [markdown.value for markdown in at.markdown]
    assert f"**{_ITEM_DESCRIPTIONS['temporal_demand']}**" in said
    # Not drawn as a heading of its own any more.
    assert "**Time Taken**" not in said
    assert not [c for c in at.caption if "How much time" in c.value]


def test_the_scale_keeps_its_accessible_name():
    """The heading is a sibling of the radio, not its label, so without this the
    scale is announced as nothing at all."""
    at = _run()

    assert {radio.label for radio in at.radio} == {"Time Taken", "Effort"}
