"""Standalone user-study app — the one URL a participant is sent.

Run with:
    streamlit run apps/study/Home.py

**The study is closed.** This app serves
:func:`behavior_graphs.viz.streamlit.user_studies.closed.render_closed_notice` and
stops, unless ``STUDY_ACCEPTING_RESPONSES`` says otherwise — see that module for
why closed is the default rather than a flag a deployment has to remember. The
study's own layouts are unchanged and still render in the lab app.

Environment variables:
    STUDY_ACCEPTING_RESPONSES
                             ``1`` to serve the study rather than the closure
                             notice. Anything else -- unset included -- keeps
                             the participant route closed.
    STUDY_VARIANT            Study variant to serve. Defaults to three_stage_v1,
                             the design the paper reports; other variants live
                             in configs/viz/study_variants/.
    STUDY_VARIANT_GROUP      Which of the variant's groups to serve, when it has
                             more than one. Defaults to its first.
    APP_PASSWORD_SHA256      SHA-256 hex of the access password (omit for open access)
    SURVEY_GCS_BUCKET        GCS bucket name for persisting responses (omit to save locally)
    STUDY_RESULTS_DIR        Local response dir when not using GCS (default: ./study_results)
    BEHAVIOR_GRAPHS_RUN_DIR    Run dir fallback when a session yaml has no run_dir
    STUDY_SESSIONS_DIR       Study config dir (default: ./configs/viz/study)

Two things separate this from the researcher-facing demo app, which can also
render every variant: the variant is pinned by the environment rather than picked
from a dropdown a participant could change mid-study, and the sidebar is hidden,
because a participant has no business seeing which condition they are in — or the
other pages of the lab app.

The group assignment (in the legacy A/B mode) is random and stable per session
(stored in st.session_state). Participants see no indication of which group they
are in.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Streamlit does not source shell ``.env`` files when launched with
# ``streamlit run``. Load the local study configuration before any session
# YAML is resolved; existing environment variables remain authoritative.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _REPO_ROOT / ".env"
if _ENV_FILE.is_file():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _name, _value = _line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(_name.strip(), _value.strip().strip('"').strip("'"))

# Make sure the repo's src tree is importable when behavior_graphs isn't
# pip-installed (e.g. running straight from a checkout).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

st.set_page_config(
    page_title="User Study",
    layout="wide",
    page_icon="🤖",
    # Belt to `hide_sidebar`'s braces: if its rule ever fails to apply, a
    # collapsed sidebar is a better failure than an open one.
    initial_sidebar_state="collapsed",
)

# ── Hide the sidebar ─────────────────────────────────────────────────────────
# A participant's window should hold the study and nothing else: the sidebar
# carries the study configuration, and in the variant-driven mode below the
# participant id and session id too.

from behavior_graphs.viz.streamlit.chrome import hide_sidebar  # noqa: E402

hide_sidebar()

# ── Is the study taking responses? ───────────────────────────────────────────
# First, and before everything: an old recruitment link must not be able to
# start an attempt. Ahead of `password_gate`, because a closed study has no
# reason to ask a participant for a password before telling them it is closed;
# ahead of the variant resolution, so a deployment pinned to a study that no
# longer runs still says the same thing; and ahead of `mark_participant_route`,
# so the visit writes nothing at all.
#
# A warm-up hit (`STUDY_WARMUP`) gets the notice too, which is right: there are
# no sessions left to warm.

from behavior_graphs.viz.streamlit.user_studies.closed import (  # noqa: E402
    accepting_responses,
    render_closed_notice,
)

if not accepting_responses():
    render_closed_notice()
    st.stop()

from behavior_graphs.viz.streamlit.auth import password_gate  # noqa: E402

if not password_gate("Access password"):
    st.stop()

# ── Whose session this is ────────────────────────────────────────────────────
# Anyone arriving here came from a recruitment link. The lab app renders the
# same layouts, and a layout that writes a record on the first page load writes
# one for a researcher checking the wording too — indistinguishable, afterwards,
# from a participant who quit on the introduction. Marked at the entry point, so
# the renderers below need no way of asking which app is running them.

from behavior_graphs.viz.streamlit.user_studies.session import (  # noqa: E402
    mark_participant_route,
)

mark_participant_route()

# ── Serve one variant, pinned by the environment ─────────────────────────────

# Defaults to the design the paper reports. Set STUDY_VARIANT to serve another
# variant from configs/viz/study_variants/.
_VARIANT_ID = os.environ.get("STUDY_VARIANT", "").strip() or "three_stage_v1"

from behavior_graphs.viz.streamlit.user_studies.registry import get_variant  # noqa: E402

_variant = get_variant(_VARIANT_ID)
if _variant is None:
    st.error(
        f"STUDY_VARIANT={_VARIANT_ID!r} is not a study variant with a renderer. "
        "See configs/viz/study_variants/ and user_studies/registry.py."
    )
    st.stop()
if not _variant.groups:
    st.error(f"Study variant {_VARIANT_ID!r} defines no groups.")
    st.stop()
_group_id = os.environ.get("STUDY_VARIANT_GROUP", "").strip()
_group = _variant.group(_group_id) if _group_id else _variant.groups[0]
if _group is None:
    _names = ", ".join(g.id for g in _variant.groups)
    st.error(f"STUDY_VARIANT_GROUP={_group_id!r} is not one of: {_names}.")
    st.stop()
# A warm-up, not a visit. Checked here because the renderer below is the
# only thing that writes a record, and a warm-up that left one would put an
# abandoned attempt in the funnel after every restart. See
# user_studies/warmup.py for what it costs a participant when nobody does
# this first.
from behavior_graphs.viz.streamlit.user_studies.warmup import (  # noqa: E402
    WARMUP_MARKER,
    WARMUP_PARAM,
    warm,
)

if str(st.query_params.get(WARMUP_PARAM, "")).strip():
    _warmed = warm(_variant)
    st.write(f"{WARMUP_MARKER}: {len(_warmed)} sessions")
    st.stop()

_variant.render(_variant, _group)
st.stop()
