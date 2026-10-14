"""Remembering a study session in the browser, so a closed tab is recoverable.

A session id lives in the participant's URL as ``?s=`` (see
:mod:`behavior_graphs.viz.streamlit.user_studies.session`), which is enough for a
refresh, a slept laptop or a redeploy: the address bar still holds it. It is not
enough for a *closed tab*. Someone who closes the window and clicks the
recruitment link again arrives at the bare link, and the server cannot tell them
from a new participant — so half an hour of answers sits in the store under an
id nobody is holding. The URL only works as a durable handle for a participant
who thought to copy it, which is not a thing to design a half-hour study around.

So the browser keeps a copy, in a cookie: it comes back on the next visit and
:func:`remembered_session` reads it straight off the connection. A recalled id
then goes through exactly the same resume gate as one that arrived in the URL —
this module decides nothing about what to do with it.

Why a cookie and not ``localStorage``, which is the obvious choice: storage is
readable only from JavaScript, and the only way to get a value from there to
Streamlit is to put it in the URL and reload. A component cannot do that.
Streamlit sandboxes component iframes with

    allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox
    allow-same-origin allow-scripts allow-downloads

and no ``allow-top-navigation``, so ``window.parent.location.replace`` is
blocked — found by driving the deployed app in a real browser, where the
redirect silently did nothing at all. ``allow-same-origin`` *is* there, so the
same iframe can reach ``window.parent.document`` and set a cookie on it, which
is what the writer below does. The cookie then rides the next request without
any of it.

Nothing here is a security boundary. The cookie holds the same id the URL
carries in plain sight, and anyone at the keyboard could read either.
"""

from __future__ import annotations

import json
import logging
import re

import streamlit as st
import streamlit.components.v1 as components

_log = logging.getLogger(__name__)

#: Namespace for the cookie names, so the study cannot collide with anything
#: else served from the same origin.
COOKIE_PREFIX = "pg_study"

#: How long a browser keeps the id. Long enough for a participant who comes back
#: the next morning, short enough that a shared machine does not carry one
#: person's session into somebody else's week.
MAX_AGE_SECONDS = 7 * 24 * 60 * 60

#: Cookie names allow far less than the ids that go into them.
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def cookie_name(variant_id: str, participant: str) -> str:
    """The cookie this participant's session id is kept in.

    Named by participant as well as by variant, because a shared machine is a
    real thing: a lab laptop that ran participant A and then participant B would
    otherwise hand B the id of A's half-finished attempt, and B would be offered
    somebody else's answers to resume. A link with no participant id (a pilot, or
    a hand-built URL) falls in the one anonymous bucket, which is the best that
    can be done without one.

    Both parts are scrubbed to what a cookie name may hold, since the
    participant id comes off the query string and is nobody's to trust.
    """
    variant = _UNSAFE.sub("_", variant_id)[:48]
    who = _UNSAFE.sub("_", participant)[:64]
    return f"{COOKIE_PREFIX}_{variant}_{who}"


def remembered_session(variant_id: str, participant: str) -> str:
    """The session id this browser last carried, or ``""``.

    Never raises: a browser that sent no cookies, or a Streamlit that cannot
    report them, means there is nothing to recall — not an error in front of
    somebody trying to start a study.
    """
    try:
        cookies = st.context.cookies or {}
        return str(cookies.get(cookie_name(variant_id, participant), "") or "").strip()[:64]
    except Exception as error:  # no request context, older Streamlit, ...
        _log.debug("could not read the study session cookie: %s", error)
        return ""


def remember_script(name: str, session_id: str, max_age: int = MAX_AGE_SECONDS) -> str:
    """The script that writes the cookie onto the parent document.

    ``Secure`` only over HTTPS: the deployment sits behind a TLS edge, but a
    developer on ``http://localhost`` would otherwise watch the browser drop the
    cookie on the floor with nothing to say why. ``SameSite=Lax`` because a
    recruitment link is a top-level navigation, which Lax allows — ``Strict``
    would withhold the cookie on exactly the arrival this exists for.

    A string rather than inline below so it can be tested: ``AppTest`` renders a
    component as an opaque element with no readable content, so the only way to
    check what the browser is told is to check what it is handed.
    """
    return f"""<script>
      try {{
        const outer = window.parent;
        const secure = outer.location.protocol === "https:" ? "; Secure" : "";
        outer.document.cookie = {json.dumps(name)} + "="
          + encodeURIComponent({json.dumps(session_id)})
          + "; path=/; max-age={int(max_age)}; SameSite=Lax" + secure;
      }} catch (e) {{}}
    </script>"""


def remember(variant_id: str, participant: str, session_id: str) -> None:
    """Keep *session_id* in the browser, for a visit that arrives without it.

    Written on every run rather than once: setting a cookie is idempotent, a
    rerun that re-emits it costs a zero-height frame, and the alternative —
    writing it once and trusting it to stay — is wrong twice over. A fragment
    rerun sweeps away app-body elements it did not itself re-emit (the same
    reason ``apps/study/Home.py`` puts its CSS in ``<head>``), and "start over"
    mints a second id that has to replace the first.
    """
    if not session_id:
        return
    components.html(remember_script(cookie_name(variant_id, participant), session_id), height=0)
