"""Where a graph's selection videos go: beside the canvas, or below it.

Beside the canvas — the default — keeps the clicked node's clips and the graph on
screen together, so you can click and watch without scrolling; that is the loop
the graph views exist for. Below gives the graph the full content width instead,
which a wide, deep trajectory tree can need before its sibling nodes and labels
become legible. :func:`graph_panel_slots` turns the answer into two containers to
render into.

Who makes the choice differs by app. The demo is a tool its operator drives, so
it gets a live sidebar control (:func:`panel_layout_selector`). A study run is
not: the layout is part of what a participant is shown, so it comes from the
session's ``graph:`` config (:func:`panel_layout_from_config`) where the
experimenter sets it and a participant cannot reach it mid-session.
"""

from __future__ import annotations

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

BELOW = "below"
RIGHT = "right"

#: What every caller falls back to, and what the sidebar radio starts on.
DEFAULT_LAYOUT = RIGHT

#: The key a study session's ``graph:`` block sets.
_CONFIG_KEY = "video_panel"

# Ordered with the default first — st.radio starts on its first option.
_LABELS = {RIGHT: "Right of graph", BELOW: "Below graph"}


def panel_layout_selector(
    *,
    key: str = "graph_panel_layout",
    default: str | None = None,
    container: DeltaGenerator | None = None,
) -> str:
    """Sidebar radio choosing where selection videos render.

    Remembers the answer outside the widget's own key, so a page that only offers
    the control on some views (the demo: the plotly views have no panel to place)
    can read the last choice back with :func:`remembered_panel_layout` instead of
    silently reverting to the default. Streamlit drops a widget's key on any run
    that does not render it, which is what made that revert happen.

    ``default`` is what the control opens on before anyone touches it, for a page
    that has an opinion of its own — the demo takes it from
    ``configs/viz/graph_demo.yaml``. Pass the same value to
    :func:`remembered_panel_layout` on the runs where the control isn't drawn, or
    a view without a panel will read back :data:`DEFAULT_LAYOUT` instead.
    """
    # Seeding the widget's key before it is created sets what it opens on, so a
    # radio coming back after a run without it resumes the remembered choice
    # rather than the default.
    if key not in st.session_state:
        st.session_state[key] = remembered_panel_layout(key=key, default=default)
    radio = container.radio if container is not None else st.sidebar.radio
    layout = radio(
        "Video panel",
        options=list(_LABELS),
        format_func=_LABELS.__getitem__,
        key=key,
        help="Where clips for the clicked node, edge, or path appear.",
    )
    st.session_state[_memory_key(key)] = layout
    return layout


def remembered_panel_layout(*, key: str = "graph_panel_layout", default: str | None = None) -> str:
    """The last layout :func:`panel_layout_selector` returned for ``key``.

    For the runs where the control isn't rendered at all — the choice still
    applies, it just has nowhere to show itself. ``default`` (falling back to
    :data:`DEFAULT_LAYOUT`) covers the runs before there is anything to remember;
    an unknown value is ignored rather than propagated, so a typo in a config
    cannot put a page into a layout no slot-maker understands.
    """
    fallback = default if default in _LABELS else DEFAULT_LAYOUT
    return st.session_state.get(_memory_key(key), fallback)


def _memory_key(key: str) -> str:
    return f"{key}__remembered"


def panel_layout_from_config(graph_params: dict | None) -> str:
    """The layout a study session's ``graph:`` config asks for.

    Unknown values fall back to :data:`DEFAULT_LAYOUT` with a visible warning rather
    than silently: a typo in a session yaml should be fixed before the
    participant arrives, not discovered afterwards in the responses.
    """
    raw = (graph_params or {}).get(_CONFIG_KEY, DEFAULT_LAYOUT)
    layout = str(raw).strip().lower()
    if layout in _LABELS:
        return layout
    st.warning(
        f"Unknown `graph.{_CONFIG_KEY}: {raw}` in the session config — using {DEFAULT_LAYOUT!r}."
    )
    return DEFAULT_LAYOUT


def graph_panel_slots(layout: str) -> tuple[DeltaGenerator, DeltaGenerator]:
    """``(graph slot, panel slot)`` containers for ``layout``.

    Both slots are created up front so the caller renders into them in either
    layout without branching: side by side for ``RIGHT``, stacked in order for
    ``BELOW``.
    """
    if layout == RIGHT:
        graph_col, panel_col = st.columns([2, 1], gap="medium")
        return graph_col, panel_col
    return st.container(), st.container()


def is_narrow(layout: str) -> bool:
    """Whether ``layout`` puts the panel in a column rather than a full row.

    Panels ask this to decide whether their own contents stack. It used to be
    inferred from ``videos_per_page(layout) > 1``, which tied a stacking decision
    to a clip count that is only tuned for how clips read.
    """
    return layout == RIGHT


def empty_state_direction(layout: str) -> str:
    """Where the graph is, from the panel's point of view — "above"/"to the left"."""
    return "above" if layout == BELOW else "to the left"


def videos_per_page(layout: str) -> int:
    """How many clips fit across a panel in ``layout``.

    A panel beside the graph is a third of the width, so the three-across row
    that fits below the graph becomes one clip at a readable size.
    """
    return 1 if layout == RIGHT else 3
