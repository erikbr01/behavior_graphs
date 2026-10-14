"""The last page before submitting: who answered, and anything they want to add.

Shared by every variant that ends this way, because both questions belong at the
end for reasons about the study rather than about the layout — see
:func:`render_exit_questions`. A variant passes the index of its own exit step;
the two questions, their wording and their gating are the same everywhere.
"""

from __future__ import annotations

import streamlit as st

from behavior_graphs.viz.streamlit.user_study.analytics import EXPERIENCE_SCALE
from behavior_graphs.viz.streamlit.user_study.survey_steps import advance_step

#: How much the participant has done this before. Asked once, at the end: before
#: the evidence it would tell them what the study thinks matters, and it is the
#: one answer that plausibly explains a difference between two participants
#: reading the same graph. The scale itself lives with the analytics, which has
#: to plot these answers in this order.
EXPERIENCE_OPTIONS = EXPERIENCE_SCALE


def _answers_key(pfx: str) -> str:
    return f"{pfx}_exit_answers"


def exit_answers(pfx: str) -> tuple[str | None, str]:
    """``(experience, remarks)`` as answered, for the submit page to record.

    Read from the copy :func:`render_exit_questions` keeps rather than from the
    radio and text area themselves. Streamlit discards a widget's value once the
    widget stops being rendered, and the submit page does not render these two —
    so the run that builds the response found both keys already gone and recorded
    ``null`` for the experience and ``""`` for the remarks however the
    participant had answered. Reproduced by walking the deployed contrastive
    layout from its closing page to its submit page (see
    ``tests/test_exit_questions.py``).
    """
    stored = st.session_state.get(_answers_key(pfx))
    if not isinstance(stored, dict):
        # Still on the page (or a session that predates the copy): the widgets
        # themselves are the answer.
        return st.session_state.get(f"{pfx}_experience"), str(
            st.session_state.get(f"{pfx}_remarks", "") or ""
        )
    return stored.get("experience"), str(stored.get("remarks", "") or "")


def render_exit_questions(
    pfx: str, step_key: str, step_idx: int, *, show_continue: bool = True
) -> bool:
    """Last page before submitting: experience, and anything they want to add.

    After the evidence rather than before it: asked up front, "how much
    experience do you have collecting robot data" tells a participant what the
    study thinks the answer depends on, which is exactly what a study comparing
    two displays should not hand them.

    Returns whether the required question is answered. With *show_continue*
    off the caller draws its own way forward — the three-stage layout puts
    Submit on this page rather than on one of its own — and needs to know
    whether to enable it.
    """
    st.header("Closing questions")
    st.markdown("**How much experience do you have with collecting data for robot learning?**")
    experience = st.radio(
        "Experience with robot data collection",
        options=EXPERIENCE_OPTIONS,
        index=None,
        key=f"{pfx}_experience",
        label_visibility="collapsed",
    )
    # No caption prompting for what to write: it invited feedback on the study
    # itself ("anything you would want changed"), which is not what this box is
    # collecting. The warning below already says the remarks are optional.
    st.markdown("**Anything else you would like to tell us?**")
    st.text_area(
        "Remarks",
        height=140,
        key=f"{pfx}_remarks",
        label_visibility="collapsed",
    )
    # Copied onto plain keys on every run of this page, which is the only place
    # these widgets exist. See :func:`exit_answers` for why the widget keys cannot
    # be read from the submit page.
    st.session_state[_answers_key(pfx)] = {
        "experience": experience,
        "remarks": st.session_state.get(f"{pfx}_remarks", ""),
    }
    if show_continue and st.button(
        "Continue →",
        type="primary",
        disabled=experience is None,
        key=f"{pfx}_exit_continue",
    ):
        advance_step(step_idx, step_key)
    if experience is None:
        st.warning("Please answer the experience question. The remarks are optional.")
    return experience is not None
