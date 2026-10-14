"""Combined demo app: Graph demo + sweep analysis + user study variants.

Run with:
    streamlit run apps/demo/Home.py

Environment variables:
    BEHAVIOR_GRAPHS_RUN_DIR    Pin one pipeline run directory.
    BEHAVIOR_GRAPHS_RUNS_ROOT  Colon-separated roots to scan for run dirs
                             (default: ./runs:./multirun).
    APP_PASSWORD_SHA256      SHA-256 hex of the access password (omit = open).
    STUDY_RESULTS_DIR        Where study responses are written (default
                             ./study_results).

Uses st.navigation to decouple sidebar display label from the URL slug,
so URLs stay lowercase while labels stay nicely cased.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Make sure the repo's src tree is importable when behavior_graphs isn't
# pip-installed (e.g. running straight from a checkout).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from behavior_graphs.viz.streamlit.auth import password_gate  # noqa: E402
from behavior_graphs.viz.streamlit.data import clear_run_query_params  # noqa: E402

if not password_gate(title="Policy Graphs demo"):
    st.stop()


_HERE = Path(__file__).resolve().parent

pages = [
    st.Page(
        _HERE / "_pages" / "graph_demo.py", title="Graph Demo", url_path="graph_demo", default=True
    ),
    st.Page(
        _HERE / "_pages" / "sweep_analysis.py", title="Sweep Analysis", url_path="sweep_analysis"
    ),
    st.Page(
        _HERE / "_pages" / "ablation_study.py",
        title="Ablation Study",
        url_path="ablation_study",
    ),
    st.Page(
        _HERE / "_pages" / "progress_estimation.py",
        title="Progress Estimation",
        url_path="progress_estimation",
    ),
    st.Page(
        _HERE / "_pages" / "policy_comparison.py",
        title="Policy Comparison",
        url_path="policy_comparison",
    ),
    st.Page(
        _HERE / "_pages" / "figures.py",
        title="Figures",
        url_path="figures",
    ),
    st.Page(
        _HERE / "_pages" / "survey_analytics.py",
        title="Survey Analytics",
        url_path="survey_analytics",
    ),
    st.Page(_HERE / "_pages" / "user_study.py", title="User Study", url_path="user_study"),
]

pg = st.navigation(pages)
pg.run()

# Placed after pg.run() (which executes the selected page's full script
# inline) so this lands at the bottom of every page's sidebar without each
# page file needing its own copy. Clears st.cache_data/st.cache_resource
# (run index, loaded run artifacts, ...) for when on-disk sweeps/runs change
# underneath a still-running session, and resets session_state (sidebar
# selectors, appearance toggles, ...) back to fresh defaults --
# preserving "_authed" so this doesn't also force a re-login.
st.sidebar.divider()
if st.sidebar.button("Clear cache & reset session", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    # Query params survive session_state.clear() below (they live on the URL,
    # not in session_state) -- without dropping them too, the rerun's
    # seed_selectors_from_query_params would just re-seed the exact same
    # task/K/etc. right back from the still-present URL, silently no-opping
    # the "reset" for anyone who arrived via a query-param deep link.
    clear_run_query_params()
    authed = st.session_state.get("_authed")
    st.session_state.clear()
    if authed:
        st.session_state["_authed"] = True
    st.rerun()
