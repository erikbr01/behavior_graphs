"""User Study — every study setup/version, side by side.

Includes both the original Group A / Group B study (still what `study.*`
deploys, wired in unchanged as the `production_group_ab` variant) and any
new designs drafted since. Every setup is driven by
`behavior_graphs.viz.streamlit.user_studies` (see that package's docstring for
how to add a new one); this page just offers "Setup" and "Group / page"
selectors and delegates rendering to the chosen variant.
"""

from __future__ import annotations

import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.registry import list_variants

st.set_page_config(page_title="User Study", layout="wide")

variants = list_variants()

if not variants:
    st.error("No study variants configured under configs/viz/study_variants/.")
    st.stop()

st.sidebar.header("User Study")

variant_ids = [v.id for v in variants]
variant_titles = {v.id: v.title for v in variants}
query_variant = st.query_params.get("study_version")
if "study_variant" not in st.session_state and query_variant in variant_ids:
    st.session_state["study_variant"] = query_variant
variant_choice = st.sidebar.selectbox(
    "Study version",
    options=variant_ids,
    format_func=lambda vid: variant_titles.get(vid, vid),
    key="study_variant",
)
variant = next(v for v in variants if v.id == variant_choice)
st.query_params["study_version"] = variant_choice

if variant.description:
    st.sidebar.caption(variant.description)

group_ids = [g.id for g in variant.groups]
group_labels = {g.id: g.label for g in variant.groups}
group_key = f"study_group_{variant.id}"
query_group = st.query_params.get("group")
if group_key not in st.session_state and query_group in group_ids:
    st.session_state[group_key] = query_group
if st.session_state.get(group_key) not in group_ids:
    st.session_state.pop(group_key, None)
group_choice = st.sidebar.selectbox(
    "Group / page",
    options=group_ids,
    format_func=lambda gid: group_labels.get(gid, gid),
    key=group_key,
)
group = variant.group(group_choice)
st.query_params["group"] = group_choice

st.sidebar.divider()

variant.render(variant, group)
