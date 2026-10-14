"""The data-collection card grid, as one custom component.

Each option is a clip with a control under it, and picking one has to feel
immediate. Built out of Streamlit widgets it does not: a card is a single-option
``st.radio``, so mutual exclusion is Python's job, and the card you just left
stays lit until the answer comes back over the websocket. Measured against the
task-load scale on the same page, which is one widget over six options:

    one widget, many options      14, 14, 17 ms
    three one-option widgets     316, 193, 474 ms

Rebuilding the grid as a single ``st.radio`` is that fast, and puts the options
in a list under the row instead of under the clip each one names — which is the
thing the cards exist for.

So the grid is a component instead, the same shape as ``_subtask_editor`` and
``_graph_component``: one hand-written ``index.html``, no build step. Inside its
frame the cards are one native radio group, so the browser does the deselect on
the click, and the pick is posted back to Python afterwards. Both halves of what
the two Streamlit-widget layouts each had to give up.

The clips are passed as URLs from :mod:`behavior_graphs.viz.media`, not as paths.
``st.video(path)`` hands Streamlit's media manager the path, which reads and
hashes the whole file to derive a URL on *every* rerun — 7.5MB per clip here.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import streamlit.components.v1 as components

from behavior_graphs.viz.media import POSTER_WIDTH, poster_src, proxy_height, video_src

_COMPONENT_DIR = Path(__file__).parent / "_collection_cards"
_cards = components.declare_component("pg_collection_cards", path=str(_COMPONENT_DIR))


def card_views(options: Sequence[tuple[str, Path]], *, key: str) -> list[dict[str, str]]:
    """``[{option, src, poster}]`` for the options whose clip can be served.

    An option with no servable clip is dropped rather than drawn with an empty
    ``<video>``: the card is the clip, and a card with nothing in it is a choice
    a participant cannot make on the evidence the question is about.
    """
    views: list[dict[str, str]] = []
    for index, (option, clip) in enumerate(options):
        src = video_src(clip, coordinates=f"cards:{key}:{index}", height=proxy_height())
        if src is None:
            continue
        poster = poster_src(clip, coordinates=f"cards-poster:{key}:{index}", width=POSTER_WIDTH)
        views.append({"option": option, "src": src, "poster": poster or ""})
    return views


def render_collection_cards(
    views: Sequence[dict[str, str]], *, key: str, selected: str | None
) -> str | None:
    """Draw the grid and return the option chosen, or *selected* unchanged.

    The component reports a pick by setting its value, which reruns the app, so
    the answer is here on the run after the click — the same contract as any
    widget. It reports nothing on a rerun it did not cause, and then this
    returns what was already chosen.
    """
    if not views:
        return selected
    picked = _cards(
        views=[dict(view) for view in views],
        selected=selected,
        key=f"{key}_cards",
        default=None,
    )
    return str(picked) if picked is not None else selected
