"""The panel that offers the graph on screen as a paper figure.

Sits under the graph, wherever a graph is drawn with ``paper_export=True`` — one
panel for both the Markov view and the trajectory tree, since both draw through
:func:`~behavior_graphs.viz.streamlit.user_study.graph_explorer.render_graph_full_width`
and hand it the same things a figure needs.

It offers the *same* graph, not a second opinion about it: the arguments the
component was given are the arguments the figure is drawn from, so pruning, the
layout, the colour encoding and the rare-edge dashing all carry over. What the
controls here choose is what only a figure has to answer — which column it goes
in, how tall it is allowed to be, and whether it has to survive being printed in
greyscale. The drawing itself is :mod:`behavior_graphs.viz.graph_figure`.

The existing SVG export next door is not replaced by this and should not be: it
is a snapshot of the live canvas, so it keeps any node the reader dragged, which
a Python-side redraw cannot know about.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import NamedTuple

import streamlit as st

from behavior_graphs.core.graph import BehaviorGraph
from behavior_graphs.viz import graph_figure, paper
from behavior_graphs.viz.streamlit.components.paper_export import paper_figure_downloads

#: The width slider, in inches. A range rather than the template's two column
#: widths, because a figure does not have to be a column wide — a graph that
#: reads at four inches should be four inches, and the space next to it is
#: worth more than the space under it.
#:
#: The step is what keeps the important part honest: both column widths are a
#: whole number of steps from the minimum, so the two sizes that need to be
#: exact — the ones a figure placed at ``\textwidth`` has to hit or be rescaled
#: by the LaTeX — are reachable on the slider rather than approximable. See
#: :mod:`behavior_graphs.viz.paper`, and the test that pins it.
WIDTH_MIN_IN = 2.0
WIDTH_MAX_IN = 8.0
WIDTH_STEP_IN = 0.02

#: Where the size controls start. Not the template's double column and not what
#: the layout would ask for: a shape settled by exporting these graphs and
#: looking at them. Narrower than a column and taller than it is wide, because
#: the layered layout answers a narrow figure by folding its fan downward, and a
#: portrait graph beside half a page of text spends the page better than a wide
#: band with the corners empty.
#:
#: :func:`~behavior_graphs.viz.graph_figure.default_height_in` still decides for a
#: caller who asks for no height at all — a script writing a figure straight out
#: of a run, where there is nobody to look at it.
DEFAULT_WIDTH_IN = 4.4
DEFAULT_HEIGHT_IN = 5.4


class SizePreset(NamedTuple):
    """A whole shape, not just a width.

    Setting the width alone is the trap this exists to close: a graph exported
    at a third of the page keeps 8 pt labels and a wrap width of an inch and a
    half, so its names run across their neighbours and its nodes touch. The size
    a figure is placed at implies a type size and a node size too, and those are
    what a preset carries.
    """

    label: str
    width_in: float
    height_in: float
    node_scale: float
    label_font_pt: float
    label_wrap_pt: float


#: The layouts a graph actually gets placed in, each as the shape that reads at
#: that size — settled by exporting these graphs and looking at them, not
#: derived. Every width is a whole number of :data:`WIDTH_STEP_IN` steps from
#: :data:`WIDTH_MIN_IN`, and every other value lands on its own slider's step,
#: so choosing a preset and then nudging one control does not move the rest.
#:
#: 2.30 rather than 2.29 for the third-of-a-page one: the exact third of a 7.16
#: in text width is not on the width slider, and 0.4% of scale is invisible
#: where a whole point of type size is not.
PRESETS: dict[str, SizePreset] = {
    "tuned": SizePreset("As tuned — 4.4 × 5.4 in", 4.4, 5.4, 1.0, 8.0, 102.0),
    "double": SizePreset("Double column, wide band — 7.16 × 2.6 in", 7.16, 2.6, 1.0, 8.0, 102.0),
    "page": SizePreset("Double column, tall — 7.16 × 5.4 in", 7.16, 5.4, 1.0, 8.0, 102.0),
    "single": SizePreset("Single column — 3.5 in wide", 3.5, 3.0, 0.8, 8.0, 66.0),
    "third": SizePreset("Third of a double column — 2.30 in wide", 2.30, 3.2, 0.6, 7.0, 42.0),
    "half": SizePreset("Half of a double column — 3.5 × 4.4 in", 3.5, 4.4, 0.9, 8.0, 66.0),
}

#: Which control each field of a preset writes, as a suffix on the panel's key.
_PRESET_KEYS = {
    "width_in": "fig_width",
    "height_in": "fig_height",
    "node_scale": "fig_nodes",
    "label_font_pt": "fig_label_pt",
    "label_wrap_pt": "fig_label_wrap",
}


def _apply_preset(key: str) -> None:
    """Write the chosen preset into the controls below it.

    A callback rather than inline, because Streamlit will not let a script set a
    widget's state after that widget has been created in the same run — the
    assignment has to happen before the rerun that draws them.
    """
    preset = PRESETS.get(st.session_state.get(f"{key}_fig_preset", ""))
    if preset is None:
        return
    for field, suffix in _PRESET_KEYS.items():
        st.session_state[f"{key}_{suffix}"] = getattr(preset, field)


#: How the label modes read on the radio.
_LABEL_MODE_NAMES = {
    "all": "Everything",
    "behaviors": "Behaviors only",
    "none": "None",
}


def _fingerprint(payload: Mapping[str, object]) -> str:
    """A cache key for one drawing.

    Over everything the figure is drawn from, because everything is a slider
    away from changing: the node set and their episode counts, every edge's
    probability, the layout, the colour override and each option below. Rounded
    before hashing so a recomputed layout that differs in the last bit of a
    float does not look like a different graph.
    """
    graph: BehaviorGraph = payload["graph"]  # type: ignore[assignment]
    pos: Mapping[int, tuple[float, float]] = payload["pos"] or {}  # type: ignore[assignment]
    colors: Mapping[int, str] = payload["colors"] or {}  # type: ignore[assignment]
    labels: Mapping[int, str] = payload["labels"] or {}  # type: ignore[assignment]
    symbols: Mapping[int, str] = payload["symbols"] or {}  # type: ignore[assignment]
    state = {
        "nodes": [
            [nid, node.name, node.cluster_id, node.num_episodes]
            for nid, node in sorted(graph.nodes.items())
        ],
        "edges": [
            [src, tgt, round(prob, 6), graph.transition_counts.get(src, {}).get(tgt, 0)]
            for src, targets in sorted(graph.transition_probs.items())
            for tgt, prob in sorted(targets.items())
        ],
        "pos": [[nid, round(x, 6), round(y, 6)] for nid, (x, y) in sorted(pos.items())],
        "colors": sorted(colors.items()),
        "labels": sorted(labels.items()),
        "symbols": sorted(symbols.items()),
        "excluded": sorted(payload["excluded"]),  # type: ignore[call-overload]
        "options": {
            k: v
            for k, v in payload.items()
            if k not in ("graph", "pos", "colors", "symbols", "labels", "excluded")
        },
    }
    return hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]


@st.cache_data(show_spinner=False, max_entries=8)
def _files(token: str, _payload: dict) -> dict[str, bytes] | None:
    """The rendered figure, keyed on *token*.

    Cached because the panel is rebuilt on every rerun of the page — a node
    click, a slider anywhere — while the figure changes only when the graph or
    one of the options does. ``_payload`` is underscore-prefixed so Streamlit
    leaves it out of the hash: it holds a graph, which is not hashable, and
    *token* already stands for it.
    """
    return graph_figure.behavior_graph_figure_files(**_payload)


def graph_figure_export_panel(
    graph: BehaviorGraph,
    *,
    key: str,
    pos: Mapping[int, tuple[float, float]] | None = None,
    excluded: frozenset[int] = frozenset(),
    colors: Mapping[int, str] | None = None,
    symbols: Mapping[int, str] | None = None,
    labels: Mapping[int, str] | None = None,
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    prob_floor: float = 0.01,
    rare_edge_max_count: int = 0,
    arrows: bool = True,
    filename: str = "behavior_graph",
    expanded: bool = False,
    preset: str = "tuned",
) -> None:
    """Draw the export panel for the graph described by these arguments.

    Pass exactly what was passed to the component, so the figure is the drawing
    above it. *arrows* defaults from the page's edge style for the same reason.

    *preset* is which of :data:`PRESETS` the controls start on, for a caller
    that knows the shape its graphs want — a page of trees labelled with
    sentences needs the width of the page, where a Markov graph of a dozen
    nodes does not.
    """
    # Seeded before the controls exist, because Streamlit will not let a script
    # set a widget's state once that widget has been created in the same run.
    # Only on the first render: after that the reader's own choices are in there.
    # Every preset-controlled slider below therefore reads its value out of
    # session state and passes no `value=` — the two together are what Streamlit
    # warns about, and the warning is right that one of them is being ignored.
    if f"{key}_fig_preset" not in st.session_state:
        st.session_state[f"{key}_fig_preset"] = preset if preset in PRESETS else "tuned"
        _apply_preset(key)

    # Collapsed by default: on a page whose subject is the graph it sits under, the
    # export is secondary and a panel of controls in the way of it is worse than
    # a click. A page whose subject *is* the export opens it (`expanded=True`) —
    # there, a collapsed expander and no chart beside it reads as a section that
    # failed to draw.
    with st.expander("⬇ Export as a paper figure (PDF)", expanded=expanded):
        st.caption(
            "The graph above, redrawn for print: the paper's serif at the size "
            "it will be printed, embedded TrueType outlines, and no timestamp — "
            "so re-exporting an unchanged graph gives an unchanged file. Nodes "
            "moved by hand are not included; the SVG export keeps those."
        )
        st.selectbox(
            "Size preset",
            list(PRESETS),
            format_func=lambda name: PRESETS[name].label,
            key=f"{key}_fig_preset",
            on_change=_apply_preset,
            args=(key,),
            help=(
                "The shape a figure needs at the size it is *placed* at — width, "
                "height, node size, type size and wrap width together. Picking "
                "one sets the controls below, which stay editable. Export at the "
                "width the figure will occupy on the page: a graph exported at "
                "7.16 in and placed in a third of a page has its 8 pt type "
                "scaled to 2.7 pt by the LaTeX, which is what makes a "
                "three-across figure unreadable."
            ),
        )
        left, right = st.columns(2)
        width_in = left.slider(
            "Width (in)",
            min_value=WIDTH_MIN_IN,
            max_value=WIDTH_MAX_IN,
            step=WIDTH_STEP_IN,
            key=f"{key}_fig_width",
            help=(
                f"The size the figure is *placed* at, so its 8 pt type is 8 pt "
                f"on the page. {paper.DOUBLE_COLUMN_WIDTH_IN} is this template's "
                f"double column and {paper.SINGLE_COLUMN_WIDTH_IN} its single "
                "column — hit either exactly and the figure needs no `width=` "
                "in the LaTeX. Any other value is fine as long as you place it "
                "at its natural size; forcing it to a column width afterwards "
                "is what rescales the type."
            ),
        )
        # Keyed on nothing but the panel, so a height chosen by hand survives a
        # nudge of the width: the two are independent, and losing one on every
        # step of the other would make the sliders fight.
        kept = graph_figure.drawn_nodes(graph, pos or {}, excluded)
        height_in = right.slider(
            "Height (in)",
            min_value=1.0,
            max_value=9.0,
            step=0.1,
            key=f"{key}_fig_height",
            help=(
                "Shorter is where a figure stops wasting the page; taller gives "
                "a deep graph or a crowded level room to spread into."
            ),
        )
        node_scale = left.slider(
            "Node size",
            min_value=0.5,
            max_value=2.5,
            step=0.1,
            key=f"{key}_fig_nodes",
            help="Radius still tracks episode count; this scales all of them.",
        )
        label_font_pt = left.slider(
            "Label size (pt)",
            min_value=5.0,
            max_value=9.0,
            step=0.5,
            key=f"{key}_fig_label_pt",
            help=(
                "Points on the printed page. The lever for a figure whose names "
                "have nowhere left to go — below about 6 pt they stop being "
                "readable in print, so hiding some is the better trade."
            ),
        )
        label_wrap_pt = left.slider(
            "Wrap labels at (pt)",
            min_value=24.0,
            max_value=180.0,
            step=6.0,
            key=f"{key}_fig_label_wrap",
            help=(
                "How wide a name may run before it breaks onto another line. "
                "Narrower spends the figure's vertical room, which a graph laid "
                "out in levels has to spare, to buy the horizontal room it does "
                "not. At the top of the range nothing short of a sentence "
                "wraps, which is the same as switching it off."
            ),
        )
        show_arrows = right.checkbox(
            "Arrowheads", value=arrows, key=f"{key}_fig_arrows", help="Direction of travel."
        )
        edge_probabilities = right.checkbox(
            "Probabilities on edges",
            value=False,
            key=f"{key}_fig_probs",
            help="Legible on a small graph; on a big one the numbers collide.",
        )
        outcome_shares = right.checkbox(
            "Outcome shares",
            value=True,
            key=f"{key}_fig_shares",
            help=(
                "The share of episodes that reached each outcome, under its "
                "name — as on screen. Marker size alone does not make small "
                "differences between two outcomes readable. Survives “Behaviors "
                "only”, where the glyph says which outcome but not how many."
            ),
        )
        label_mode = right.radio(
            "Labels",
            graph_figure.LABEL_MODES,
            # Starts where this graph wants it: naming three outcomes is
            # information, naming thirty is noise. A choice made here sticks,
            # since the key does not change with the graph.
            index=graph_figure.LABEL_MODES.index(
                graph_figure.suggested_label_mode(graph, kept, symbols)
            ),
            format_func=_LABEL_MODE_NAMES.__getitem__,
            key=f"{key}_fig_labels",
            help=(
                "A tree ends every path in an outcome, so naming those writes "
                "FAILURE under thirty crosses that already say it. “Behaviors "
                "only” leaves the roles to their markers."
            ),
        )
        full_names = left.checkbox(
            "Full node names",
            value=False,
            key=f"{key}_fig_names",
            help=(
                "“Behavior 2” rather than “B2” — and any name written for a "
                "node, which is worth more to a reader than its number. Long "
                "ones break onto a second line rather than running across the "
                "graph."
            ),
        )
        greyscale = left.checkbox(
            "Greyscale",
            value=False,
            key=f"{key}_fig_grey",
            help=(
                "Drops the colour encoding; the roles stay legible by the glyph "
                "inside the node (▶ start, ✓ success, ✕ failure, ■ end)."
            ),
        )

        payload = {
            "graph": graph,
            "pos": dict(pos) if pos is not None else None,
            "excluded": excluded,
            "colors": dict(colors) if colors is not None else None,
            "symbols": dict(symbols) if symbols is not None else None,
            "labels": dict(labels) if labels is not None else None,
            "min_edge_prob": min_edge_prob,
            "min_edge_count": min_edge_count,
            "prob_floor": prob_floor,
            "rare_edge_max_count": rare_edge_max_count,
            "width_in": width_in,
            "height_in": height_in,
            "arrows": show_arrows,
            "edge_probabilities": edge_probabilities,
            "shorten_labels": not full_names,
            "label_font_pt": label_font_pt,
            "label_wrap_pt": label_wrap_pt,
            "outcome_shares": outcome_shares,
            "label_mode": label_mode,
            "node_scale": node_scale,
            "greyscale": greyscale,
        }
        files = _files(_fingerprint(payload), payload)
        if files is None:
            st.info(
                "Nothing to draw: every node is pruned out of this graph. "
                "Loosen the simplification settings in the sidebar."
            )
            return

        paper_figure_downloads(
            files,
            key=f"{key}_fig",
            filename=filename,
            width_in=width_in,
            height_in=height_in,
        )
