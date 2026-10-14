"""Tests for the study's exit page: who answered, and anything they want to add.

The page exists so the two things that belong at the end are asked at the end —
prior experience (asked up front, it tells a participant what the study thinks
the answer depends on) and open remarks (asked before the evidence, there is
nothing to remark on yet). It is shared by every layout that ends this way, so
it is tested through the shared module and, for the step it sits at, through the
contrastive layout that first had it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.user_studies.exit_questions import (  # noqa: E402
    EXPERIENCE_OPTIONS,
)

# Step indices the published three-stage layout uses; the exit page is drawn
# from the shared helper, so the test drives that directly.
_STEP_PHASE2 = 1
_STEP_EXIT = 2
_STEP_SUBMIT = 3

_SCRIPT = f"""
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.exit_questions import render_exit_questions

render_exit_questions("t", "t_step", {_STEP_EXIT})
"""


def _run() -> AppTest:
    return AppTest.from_string(_SCRIPT, default_timeout=30).run()


def test_the_exit_page_sits_between_the_evidence_and_the_submission():
    assert _STEP_PHASE2 < _STEP_EXIT < _STEP_SUBMIT


def test_it_asks_experience_on_a_scale_and_remarks_in_prose():
    at = _run()

    assert not at.exception
    (scale,) = at.radio
    assert list(scale.options) == list(EXPERIENCE_OPTIONS)
    assert scale.value is None, "nothing pre-selected"
    assert [t.label for t in at.text_area] == ["Remarks"]


def test_continue_waits_for_the_scale_but_not_the_remarks():
    at = _run()
    assert at.button[0].disabled

    at = at.radio(key="t_experience").set_value("Some").run()
    assert not at.button[0].disabled, "remarks are optional"


_READBACK = """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.exit_questions import exit_answers

st.session_state["seen"] = exit_answers("t")
"""


def test_the_answers_survive_the_page_they_were_answered_on():
    """Read back from a run that has only the non-widget keys.

    The radio and the text area are widgets, and Streamlit discards a widget's
    value once the widget stops being rendered — so the submit page, which draws
    neither, saw an empty session state and recorded ``null`` for the experience
    and ``""`` for the remarks however the participant had answered. The page
    therefore copies both onto a plain key, and the submission reads that copy;
    dropping the widget keys here is what the real page transition does.
    """
    at = AppTest.from_string(_SCRIPT, default_timeout=30).run()
    at = at.radio(key="t_experience").set_value("Some").run()
    at = at.text_area(key="t_remarks").set_value("the graph was quicker").run()

    after = AppTest.from_string(_READBACK, default_timeout=30)
    for key, value in at.session_state.filtered_state.items():
        if key in ("t_experience", "t_remarks"):
            continue
        after.session_state[key] = value
    after = after.run()

    assert not after.exception
    assert after.session_state["seen"] == ("Some", "the graph was quicker")


def test_an_unanswered_page_reads_back_as_unanswered():
    """Not as a crash, and not as a blank that looks like a real answer."""
    after = AppTest.from_string(_READBACK, default_timeout=30).run()

    assert after.session_state["seen"] == (None, "")


def test_the_submit_page_reads_the_answers_through_that_copy():
    """A layout reading the widget keys directly is the bug above, again."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import (
        render as three_stage,
    )

    for module in (three_stage,):
        # Module-wide rather than per function: the three-stage layout builds one
        # response dict for its checkpoints and its submission, so the read lives
        # in that builder rather than on the submit page itself.
        src = inspect.getsource(module)
        assert "exit_answers(pfx)" in src, module.__name__
        assert 'st.session_state.get(f"{pfx}_experience")' not in src, module.__name__
        assert 'st.session_state.get(f"{pfx}_remarks"' not in src, module.__name__
