"""What a recruitment link lands on once the study has closed.

The study's own pages are untouched and still render — in the lab app, where a
researcher can walk the layout for the record, and in
``tests/test_study_app_route.py``, which opens the gate below explicitly. What
changes is the **participant route**: ``apps/study/Home.py`` serves this notice
instead, before it resolves a variant, before it asks for a password, and before
it marks the session — so a visit to an old link cannot start an attempt, cannot
write a record, and cannot be told apart afterwards from a participant who
quit, because there is nothing to tell apart.

**Closed is the default.** Not a flag a deployment has to remember to set: the
image on the participant URL is built from this repo and restarted by hand, so
a closure that needed configuration would still be open on any container nobody
got to. :data:`ACCEPTING_ENV` is the way back, and it has to be set on purpose.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Where a participant's question goes. Written into the notice as a ``mailto:``
#: link whose *text is the address*, so a reader whose mail client does not open
#: it can still read and select it -- which is why it is not also repeated in
#: plain text beside itself, as it was at first.
CONTACT_EMAIL = "erbauer@stanford.edu"

#: The variable that reopens the participant route. Named for what it does to
#: the data rather than for the page it swaps: what is off is the *taking of
#: responses*, and the notice is the consequence.
ACCEPTING_ENV = "STUDY_ACCEPTING_RESPONSES"

#: What counts as "yes" in :data:`ACCEPTING_ENV`. A closed study is the safe
#: reading of anything else — ``"0"``, ``"false"``, ``""``, a typo — because the
#: cost of the two mistakes is not the same: a notice shown to a participant who
#: could have answered is a lost response, and a form shown after closing is a
#: response collected without anyone expecting it.
_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})


def accepting_responses(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the participant route should serve the study at all.

    *environ* defaults to ``os.environ``; it is a parameter so a test can ask
    what a given deployment would do without editing the process's own
    environment.
    """
    source = os.environ if environ is None else environ
    return str(source.get(ACCEPTING_ENV, "")).strip().casefold() in _TRUTHY


def render_closed_notice() -> None:
    """The page an old recruitment link now shows.

    Three things, in the order a participant needs them: that there is nothing
    to fill in, that the raffle still reaches them without any action on their
    part, and who to ask. Nothing else — no study branding, no progress bar, no
    button, and no note about what happens to the responses: a page that looks
    like the study's first page invites a reader to look for the way in, and a
    participant reading this has no stake in how the record is kept.
    """
    import streamlit as st

    st.title("This study has closed")
    st.markdown(
        "Thank you for your interest. **The study is no longer accepting "
        "responses**, so there is nothing to fill in here."
    )
    st.markdown(
        "- **The raffle will be drawn soon.** If you chose to enter it, you will "
        "be contacted by email should you win something. Nothing further is "
        "needed from you.\n"
        f"- **Questions?** Write to [{CONTACT_EMAIL}](mailto:{CONTACT_EMAIL})."
    )


__all__ = ["ACCEPTING_ENV", "CONTACT_EMAIL", "accepting_responses", "render_closed_notice"]
