"""The page a participant sees when they come back to a study link.

Streamlit has no memory of them: their previous connection is gone and with it
everything in ``st.session_state``. What is left is the session id in their URL
and the record it addresses (see :mod:`behavior_graphs.viz.streamlit.user_studies
.session`), so this asks the one question worth asking — carry on, or start
again — instead of silently doing either.

Silently resuming would drop somebody who wanted a fresh start into the middle
of a half-finished study with no way back; silently restarting would throw away
work they had just spent twenty minutes on. Neither is recoverable by the
participant, so the choice is theirs.
"""

from __future__ import annotations

import streamlit as st

RESUME = "resume"
RESTART = "restart"

#: Who a participant writes to when the study has gone wrong for them. One
#: constant rather than the address inline, so the pages that need to name
#: somebody cannot end up naming different people.
CONTACT_EMAIL = "erbauer@stanford.edu"


def render_resume_gate(*, pfx: str, progress: str) -> str | None:
    """Offer to continue the found session or start over.

    *progress* says where they got to, in the layout's own words ("Task 2
    (rollout videos)"), because "resume your session" alone does not tell
    someone whether resuming is worth it.

    Returns :data:`RESUME`, :data:`RESTART`, or ``None`` while they decide.
    """
    st.header("Welcome back")
    st.markdown(
        f"""We found an earlier session from this link, and it is not finished.

**Where you got to:** {progress}

You can carry on from there, or start the study again from the beginning. If you
start again, your earlier answers are kept but set aside, and you will see the
tasks in a new order."""
    )
    resume_column, restart_column = st.columns(2)
    with resume_column:
        if st.button(
            "Resume where I left off", type="primary", key=f"{pfx}_resume", width="stretch"
        ):
            return RESUME
    with restart_column:
        if st.button("Start over", key=f"{pfx}_restart", width="stretch"):
            return RESTART
    return None


def render_already_submitted(*, pfx: str) -> bool:
    """The other thing a returning link can mean: they already finished.

    Kept separate from the gate above because there is nothing to decide about
    *their own* attempt: offering "resume" over a submitted response is how one
    participant's second pass overwrites their first.

    There is one thing to decide, though, and it needs a way out. This page is
    reached from the browser's cookie as well as from the URL, and the cookie is
    named for the study and the participant id, which a bare recruitment link
    does not carry. Everyone on one browser therefore shares a single anonymous
    bucket for seven days, and without the button below the next person at a
    shared machine is told they have already finished a study they have never
    seen, with no ``?s=`` to edit and no way forward. Returns whether they asked
    to start fresh.
    """
    st.header("You have already completed this study")
    st.markdown(
        f"""Your answers were recorded, and there is nothing more to do. Thank you.

If you believe this is a mistake, please get in touch at {CONTACT_EMAIL} rather
than starting again, so that your original answers are not replaced."""
    )
    st.caption(
        "Using a shared computer? If somebody else completed the study on this "
        "browser and you have not taken it yet, start your own below. Your "
        "answers are recorded separately and nothing already submitted is "
        "affected."
    )
    return bool(st.button("I am someone else — start the study", key=f"{pfx}_not_me"))
