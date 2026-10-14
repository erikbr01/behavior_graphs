"""The study's graph key, on its own.

A figure drawn by the same component as the real graph, with callout lines
naming what the drawing encodes. Only a browser can say whether the callouts
landed on what they point at and whether the text fits the canvas — ``AppTest``
does not execute custom components at all.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="graph intro smoke test", layout="wide")

from behavior_graphs.viz.streamlit.user_study.graph_intro import (  # noqa: E402
    render_graph_intro,
)

render_graph_intro()
