"""Server-side study sessions, addressed by an id in the participant's URL.

Streamlit keeps a participant's answers in ``st.session_state``, which lives and
dies with one websocket: a refresh, a slept laptop, a dropped connection or a
redeploy starts a new one. For a study that takes half an hour and only wrote
its answers at the very end, that meant the participant's whole session was
lost — and with it, everything they had already told us.

So the answers live server-side instead, in the response store, as one record
per *attempt*:

  * On a participant's first visit a session id is minted, the record is written
    before they have clicked anything, and the id is put into the URL as ``?s=``.
    The browser then carries the identity, so the server does not have to.
  * Every change to the answers is checkpointed onto that record. A study
    abandoned at stage two still leaves stage one's answers, marked as
    unfinished rather than mixed in with the completed ones.
  * Coming back to the URL without a live Streamlit session finds the record and
    offers to resume it (:func:`resume_choice`) — or to start over, which mints a
    new id and leaves the old record alone, marked abandoned. Study data is never
    deleted, only superseded.

What this module deliberately does not do is decide what a payload contains:
each study layout builds its own response dict and hands it over. The one field
this module owns is ``session_id``, and the two it maintains are ``status`` and
``updated_at``.
"""

from __future__ import annotations

import logging
import secrets
from datetime import datetime

import streamlit as st

from behavior_graphs.viz.streamlit.user_study.response_store import get_store

_log = logging.getLogger(__name__)

#: Query parameter carrying the session id. Short, because participants see it.
SESSION_PARAM = "s"

#: Query parameters a participant id can arrive under. ``PROLIFIC_PID`` is what
#: Prolific appends; the rest are for hand-built or piloted links.
PID_PARAMS = ("pid", "PROLIFIC_PID", "prolific_pid", "participant", "worker_id")

#: The other two fields Prolific appends, kept so a response can be reconciled
#: with a Prolific submission without asking the participant for anything.
PROLIFIC_PARAMS = {"study_id": ("STUDY_ID", "study_id"), "session_id": ("SESSION_ID",)}

IN_PROGRESS = "in_progress"
SUBMITTED = "submitted"
ABANDONED = "abandoned"

#: Session-state flag set by the participant-facing app, and nothing else.
_PARTICIPANT_ROUTE_KEY = "study_participant_route"

#: Where a record came from. A layout that writes on the first page load writes
#: one for a researcher opening it in the lab app too, and those look exactly
#: like a participant who quit on the introduction: they landed in the funnel and
#: in the completed-of-started metric. Recorded rather than filtered on the
#: participant id, because a pilot run by hand through the lab app has one.
PARTICIPANT = "participant"
LAB = "lab"


def mark_participant_route() -> None:
    """Called by ``apps/study/Home.py``: this session is a real participant's.

    Set by the entry point rather than sniffed from the environment down here,
    so the renderers stay callable from either app without reading env vars.
    """
    st.session_state[_PARTICIPANT_ROUTE_KEY] = True


def participant_route() -> bool:
    """Whether this session is being served by the participant-facing app."""
    return bool(st.session_state.get(_PARTICIPANT_ROUTE_KEY))


def origin() -> str:
    """:data:`PARTICIPANT` or :data:`LAB`, for the record being written."""
    return PARTICIPANT if participant_route() else LAB


def _new_session_id() -> str:
    """A short, unguessable id.

    Unguessable because the URL is the only credential: anyone holding it can
    resume that attempt. 16 hex characters is 64 bits, which is not enumerable
    by anyone who would bother.
    """
    return secrets.token_hex(8)


def _query_first(names: tuple[str, ...]) -> str:
    for name in names:
        value = str(st.query_params.get(name, "") or "").strip()
        if value:
            return value
    return ""


def participant_id() -> str:
    """The participant id from the URL, or ``""`` when the link carried none.

    Not a text box: a sidebar field defaulting to "anonymous" is a field nobody
    fills in, and a study of hundreds of people then cannot tell a repeat
    submission from a new participant or match a response to a payment.
    """
    return _query_first(PID_PARAMS)[:64]


def prolific_fields() -> dict[str, str]:
    """Prolific's study and session ids, when the link came from Prolific."""
    found = {}
    for field, names in PROLIFIC_PARAMS.items():
        value = _query_first(names)
        if value:
            found[field] = value[:64]
    return found


def url_session_id() -> str:
    """The session id in the URL, or ``""``."""
    return str(st.query_params.get(SESSION_PARAM, "") or "").strip()[:64]


def start_session(pfx: str) -> str:
    """Mint this participant's session, publish it to the URL, and return its id.

    Idempotent per Streamlit session: the id is kept in ``st.session_state`` and
    the URL is only written once. Writing a query parameter triggers a rerun, so
    a mint that ran on every rerun would loop.
    """
    key = f"{pfx}_session_id"
    existing = st.session_state.get(key)
    if existing:
        return str(existing)
    session_id = _new_session_id()
    st.session_state[key] = session_id
    st.query_params[SESSION_PARAM] = session_id
    return session_id


def adopt_session(pfx: str, session_id: str) -> None:
    """Make *session_id* this Streamlit session's, in state and in the URL."""
    st.session_state[f"{pfx}_session_id"] = session_id
    st.query_params[SESSION_PARAM] = session_id


def load(session_id: str) -> dict | None:
    """The stored record for *session_id*, or ``None``.

    Never raises: a store that is unreachable must not stop a participant from
    starting the study, it only means there is nothing to resume.
    """
    if not session_id:
        return None
    try:
        store = get_store()
        getter = getattr(store, "get", None)
        if getter is not None:
            return getter(session_id)
        return next(
            (
                record
                for record in reversed(store.list_all())
                if str(record.get("session_id", "")) == session_id
            ),
            None,
        )
    except Exception as error:  # unreadable store, bad credentials, ...
        _log.warning("could not read study session %s: %s", session_id, error)
        return None


def build_record(session_id: str, payload: dict, status: str) -> dict:
    """The stored shape of an attempt: a layout's payload plus what this module owns.

    Shared by :func:`checkpoint`, :func:`submit` and the off-box mirror in
    :mod:`behavior_graphs.viz.streamlit.user_studies.backup`, so a copy in a
    bucket is the same record as the copy on disk rather than one assembled a
    second way.
    """
    return {
        **payload,
        "session_id": session_id,
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def checkpoint(pfx: str, session_id: str, payload: dict, *, status: str = IN_PROGRESS) -> bool:
    """Write *payload* as the current state of *session_id*, if it has changed.

    Called on every rerun, so it compares against what it last wrote and does
    nothing when a rerun changed no answer — a participant reading a graph
    generates plenty of reruns that are not worth a write.

    Best-effort by design: a checkpoint that fails is logged and the page carries
    on, because losing a checkpoint costs one interaction's worth of data while
    taking the page down costs the whole session. The submission itself does not
    go through here (see :func:`submit`).
    """
    record = build_record(session_id, payload, status)
    # `updated_at` is excluded from the comparison: it changes on every rerun by
    # definition, and including it would make every rerun a write.
    fingerprint = repr(sorted((k, repr(v)) for k, v in record.items() if k != "updated_at"))
    state_key = f"{pfx}_checkpoint_fingerprint"
    if st.session_state.get(state_key) == fingerprint:
        return False

    # A finished response is never written back to an unfinished one. The other
    # guard is `st.session_state[f"{pfx}_submitted"]`, which is per websocket,
    # and the app hands out a second one on the same id by design: the
    # introduction tells a participant they can open the link again, and doing
    # that in a *second tab* leaves the first live. Whichever tab writes last
    # wins, so a submit in one and any answer change in the other replaced a
    # completed response with an in-progress one, which the analytics then hides
    # because unfinished sessions are filtered out by default.
    #
    # Below the fingerprint check, not above it: this reads the whole store
    # (every revision of every participant, re-parsed; two network round trips
    # when the store is a bucket) and `checkpoint` is called on every rerun,
    # including every radio click and graph interaction. Only a rerun that
    # actually changed an answer reaches here, which is the only time the guard
    # can matter.
    if status == IN_PROGRESS:
        stored = load(session_id)
        if stored is not None and str(stored.get("status")) == SUBMITTED:
            _log.info("not checkpointing over submitted session %s", session_id)
            return False

    try:
        get_store().save(record)
    except Exception as error:
        _log.warning("could not checkpoint study session %s: %s", session_id, error)
        return False
    st.session_state[state_key] = fingerprint
    return True


def submit(pfx: str, session_id: str, payload: dict) -> str:
    """Write the finished response. Unlike a checkpoint, failure is the caller's.

    Returns the store's id for the record. Raising is deliberate: the
    participant is pressing Submit and is owed the truth about whether it
    worked.
    """
    record = build_record(session_id, payload, SUBMITTED)
    response_id = get_store().save(record)
    st.session_state[f"{pfx}_checkpoint_fingerprint"] = None
    return response_id


def abandon(session_id: str, record: dict) -> None:
    """Mark a record superseded because its participant chose to start over.

    The record is kept: a session someone walked away from is still evidence
    about where they walked away, and the point of checkpointing was to have it.
    """
    try:
        get_store().save({**record, "status": ABANDONED, "session_id": session_id})
    except Exception as error:
        _log.warning("could not mark study session %s abandoned: %s", session_id, error)


def is_resumable(record: dict | None) -> bool:
    """Whether *record* is an attempt that can still be continued."""
    return bool(record) and str(record.get("status", IN_PROGRESS)) == IN_PROGRESS
