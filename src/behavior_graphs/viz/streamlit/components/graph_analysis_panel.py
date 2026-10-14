"""Streamlit component for rendering a graph analysis report."""

from __future__ import annotations

import streamlit as st


def render_graph_analysis_panel(
    analysis: dict,
    node_vlm_labels: dict[int, dict] | None = None,
) -> None:
    """Render the graph analysis expander panel.

    Args:
        analysis: Dict loaded from graph_analysis.json.
        node_vlm_labels: Optional node label dict (node_id -> {label, ...}) used
            to annotate similar-pair headings with human-readable names.
    """
    model = analysis.get("model", "VLM")
    ts = analysis.get("timestamp", "")
    label = f"Graph Analysis — {model}"
    if ts:
        label += f" · {ts[:10]}"

    with st.expander(f"🔬 {label}", expanded=False):
        summary = analysis.get("task_summary", "")
        if summary:
            st.subheader("Task Summary")
            st.write(summary)

        success_modes = analysis.get("success_modes", [])
        failure_modes = analysis.get("failure_modes", [])
        if success_modes or failure_modes:
            col_s, col_f = st.columns(2)
            with col_s:
                st.subheader("Success modes")
                for m in success_modes:
                    path_str = m.get("path_str", "")
                    prob = m.get("probability")
                    desc = m.get("description", "")
                    prob_str = f" · {prob:.0%}" if prob is not None else ""
                    st.markdown(f"**{path_str}**{prob_str}")
                    if desc:
                        st.caption(desc)
            with col_f:
                st.subheader("Failure modes")
                for fi, m in enumerate(failure_modes):
                    desc = m.get("description", "")
                    key_node_ids = m.get("key_node_ids", [])
                    prob = m.get("probability")
                    nodes_str = ", ".join(f"B{n}" for n in key_node_ids) if key_node_ids else ""
                    prob_str = f" · {prob:.0%}" if prob is not None else ""
                    dominant = " ★ dominant" if fi == 0 else ""
                    header = (
                        f"**{nodes_str}**{prob_str}{dominant}"
                        if nodes_str
                        else f"**Failure mode**{prob_str}{dominant}"
                    )
                    st.markdown(header)
                    if desc:
                        st.caption(desc)

        pairs = analysis.get("similar_pairs", [])
        if pairs:
            st.subheader("Similar cluster pairs")
            for p in pairs:
                na, nb = p.get("node_a"), p.get("node_b")
                dist = p.get("distinction", "")
                lbl_a = (
                    (node_vlm_labels or {}).get(na, {}).get("label", f"B{na}")
                    if na is not None
                    else "?"
                )
                lbl_b = (
                    (node_vlm_labels or {}).get(nb, {}).get("label", f"B{nb}")
                    if nb is not None
                    else "?"
                )
                st.markdown(f"**B{na} ({lbl_a}) ≈ B{nb} ({lbl_b})**")
                if dist:
                    st.caption(dist)

        overall = analysis.get("overall_assessment", "")
        if overall:
            st.subheader("Overall Assessment")
            st.info(overall)
