from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)
from behavior_graphs.viz.labels import short_node_label
from behavior_graphs.viz.plotly.clusters import CLUSTER_COLORS

_SPECIAL_IDS = frozenset({START_NODE_ID, END_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID})

# Fold-badge clicks are sent as [_FOLD_TOGGLE_OFFSET - node.id, seq] over the
# same list channel as normal node clicks (see render_graph_component) so
# they're always well below any real node id (synthetic ids start at 100_000;
# the special ids above and RARE_COLLAPSE_CLUSTER_ID are small negatives) or
# the -1 deselect sentinel, and unambiguously decodable back to node.id via
# `_FOLD_TOGGLE_OFFSET - encoded`.
_FOLD_TOGGLE_OFFSET = -1_000_000

# Unique object used as a sentinel so `last_click` in session state is never
# equal to None or any real component value on the very first render.
_SENTINEL = object()

_NODE_COLOR = {
    START_NODE_ID: "#2ca02c",
    SUCCESS_NODE_ID: "#2ca02c",
    FAILURE_NODE_ID: "#d62728",
    END_NODE_ID: "#888888",
}

_COMPONENT_DIR = Path(__file__).parent / "_graph_component"
_graph_component = components.declare_component("behavior_graph", path=str(_COMPONENT_DIR))


def compute_pruned_graph_nodes(
    graph: BehaviorGraph,
    min_visit_prob: float,
    n_total: int,
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
) -> frozenset[int]:
    """Return the set of node IDs to EXCLUDE from the graph.

    Algorithm:
    1. Remove regular nodes whose visit frequency < min_visit_prob
       (or whose num_episodes < min_edge_count, when min_edge_count > 0).
    2. BFS from START following only edges with prob >= min_edge_prob
       (and transition_count >= min_edge_count when > 0) through surviving
       nodes; remove anything unreachable.
    Special nodes (START/SUCCESS/FAILURE) are never excluded by the threshold but
    may disappear if all paths to them are cut by node or edge pruning.
    """
    surviving = {
        nid
        for nid, node in graph.nodes.items()
        if nid in _SPECIAL_IDS
        or (
            (n_total == 0 or node.num_episodes / n_total >= min_visit_prob)
            and (min_edge_count <= 0 or node.num_episodes >= min_edge_count)
        )
    }

    reachable: set[int] = set()
    queue = [START_NODE_ID] if START_NODE_ID in surviving else []
    while queue:
        nid = queue.pop()
        if nid in reachable:
            continue
        reachable.add(nid)
        for tgt, prob in graph.transition_probs.get(nid, {}).items():
            if tgt not in surviving or tgt in reachable:
                continue
            if prob < min_edge_prob:
                continue
            if min_edge_count > 0:
                cnt = graph.transition_counts.get(nid, {}).get(tgt, 0)
                if cnt < min_edge_count:
                    continue
            queue.append(tgt)

    return frozenset(nid for nid in graph.nodes if nid not in reachable)


def _compute_layout(
    graph: BehaviorGraph,
    excluded: frozenset[int] = frozenset(),
) -> dict[int, tuple[float, float]]:
    """BFS-layered layout with barycenter refinement (adapted from the main plotting module).

    Nodes are columns by BFS depth from START; terminals pinned to the far right.
    10 passes of barycenter + spread give a clean, crossing-minimised layout.
    """
    from collections import defaultdict

    import networkx as nx

    G = nx.DiGraph()
    for nid in graph.nodes:
        if nid not in excluded:
            G.add_node(nid)
    for src, targets in graph.transition_probs.items():
        if src in excluded:
            continue
        for tgt, prob in targets.items():
            if tgt in excluded:
                continue
            if src in graph.nodes and tgt in graph.nodes and prob > 0:
                G.add_edge(src, tgt, weight=prob)

    terminal_ids = {n for n in G.nodes() if n in _SPECIAL_IDS and n != START_NODE_ID}

    try:
        distances = dict(nx.single_source_shortest_path_length(G, START_NODE_ID))
    except nx.NodeNotFound:
        distances = {START_NODE_ID: 0}

    # Reverse-BFS for unreachable nodes
    rev_distances: dict[int, int] = {}
    for term_id in terminal_ids:
        try:
            rd = dict(nx.single_source_shortest_path_length(G.reverse(), term_id))
            for n, d in rd.items():
                if n not in rev_distances or d < rev_distances[n]:
                    rev_distances[n] = d
        except (nx.NodeNotFound, nx.NetworkXError):
            pass

    max_fwd = max((d for n, d in distances.items() if n not in terminal_ids), default=1)
    for node in G.nodes():
        if node not in distances:
            distances[node] = (
                (max_fwd + 1 - rev_distances[node]) if node in rev_distances else max_fwd // 2 + 1
            )

    end_layer = max((d for n, d in distances.items() if n not in terminal_ids), default=1) + 1
    for term_id in terminal_ids:
        distances[term_id] = end_layer

    layers: dict[int, list] = defaultdict(list)
    for node, d in distances.items():
        layers[d].append(node)
    for layer_idx in layers:
        layers[layer_idx].sort(key=lambda n: (n in _SPECIAL_IDS, n))

    total_layers = max(layers.keys()) if layers else 1
    x_min, x_max = -2.5, 2.5

    pos: dict[int, tuple[float, float]] = {}
    for layer_idx, nodes in layers.items():
        x = x_min + (x_max - x_min) * layer_idx / max(total_layers, 1)
        n = len(nodes)
        y_spacing = max(1.0, 1.2 - 0.05 * n)
        for i, node in enumerate(nodes):
            pos[node] = (x, (i - (n - 1) / 2) * y_spacing)

    # Barycenter refinement
    min_gap = 0.8
    for _ in range(10):
        for layer_idx in sorted(layers.keys()):
            nodes = layers[layer_idx]
            if len(nodes) <= 1:
                continue
            for node in nodes:
                neighbours = list(G.predecessors(node)) + list(G.successors(node))
                ys = [pos[nb][1] for nb in neighbours if nb in pos]
                if ys:
                    pos[node] = (pos[node][0], 0.6 * float(np.mean(ys)) + 0.4 * pos[node][1])
            nodes_sorted = sorted(nodes, key=lambda n: pos[n][1])
            for i in range(1, len(nodes_sorted)):
                py = pos[nodes_sorted[i - 1]][1]
                cy = pos[nodes_sorted[i]][1]
                if cy - py < min_gap:
                    pos[nodes_sorted[i]] = (pos[nodes_sorted[i]][0], py + min_gap)
            ys = [pos[n][1] for n in nodes_sorted]
            center = float(np.mean(ys))
            for n in nodes_sorted:
                pos[n] = (pos[n][0], pos[n][1] - center)

    return pos


def component_layout(
    graph: BehaviorGraph,
    pos: dict[int, tuple[float, float]] | None,
    excluded: frozenset[int] = frozenset(),
) -> dict[int, tuple[float, float]]:
    """The positions the component will draw with, given what a caller passed.

    Public because the figure export has to draw the same graph as the screen,
    and the layout is the one input it cannot recompute for itself: two
    layouts of one graph are two different pictures. Both ask here.

    A caller who passes no layout gets the BFS-layered one, computed over the
    nodes that survived pruning. A caller who passes one gets it back with any
    node it does not mention pinned at the origin — the component draws exactly
    the nodes it has positions for, so a missing one would otherwise vanish
    without the caller having said to exclude it.
    """
    if pos is None:
        return _compute_layout(graph, excluded=excluded)
    filled = dict(pos)
    for nid in graph.nodes:
        filled.setdefault(nid, (0.0, 0.0))
    return filled


def _build_graph_json(
    graph: BehaviorGraph,
    pos: dict[int, tuple[float, float]],
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    symbol_override: dict[int, str] | None = None,
    color_override: dict[int, str] | None = None,
    prob_floor: float = 0.01,
    rare_edge_max_count: int = 0,
    fold_state: dict[int, str] | None = None,
    shorten_labels: bool = True,
) -> str:
    """Serialize graph data for the SVG component.

    ``prob_floor`` is the unconditional minimum probability below which
    edges are dropped from the payload; outcome-preserving graph modes pass
    ``0.0`` so no observed transition is ever hidden. Callers where every
    node has exactly one parent edge (e.g. the trajectory tree) must also
    pass ``0.0`` — compute_pruned_graph_nodes' reachability BFS uses
    min_edge_prob without a floor, so applying a higher floor here can
    orphan a node that survived pruning only via a sub-floor edge (visible
    as a tree node with no incoming edge). Edges with count
    ``<= rare_edge_max_count`` (when > 0) are flagged ``rare`` and rendered
    dashed with a visibility floor instead of fading out.

    ``fold_state`` (node id -> ``"collapsed"`` | ``"expanded"``) draws a
    small +/- badge on nodes that participate in fold/unfold — collapse-
    threshold callers only (e.g. the trajectory tree); ``None`` elsewhere.

    The drawn label is :func:`~behavior_graphs.viz.labels.short_node_label` of the
    node's name, while ``tooltip`` keeps the full one — an empty short label means
    "draw no label", which the component honours rather than drawing an empty chip.
    ``shorten_labels=False`` draws the names verbatim: shortening exists because a
    real graph has no room for "Behavior 12" on every node, which is not true of a
    small figure whose whole subject is what the nodes are.
    """
    fold_state = fold_state or {}
    symbol_override = symbol_override or {}
    color_override = color_override or {}
    nodes = []
    for nid, node in graph.nodes.items():
        if nid not in pos:
            continue
        if nid in symbol_override:
            symbol = symbol_override[nid]
            color = color_override.get(nid, CLUSTER_COLORS[nid % len(CLUSTER_COLORS)])
            tooltip = node.name
            label = node.name
            is_special = True
        elif nid in _SPECIAL_IDS:
            color = color_override.get(nid, _NODE_COLOR.get(nid, "#888"))
            symbol = (
                "star"
                if nid == SUCCESS_NODE_ID
                else "x"
                if nid == FAILURE_NODE_ID
                else "diamond"
                if nid == START_NODE_ID
                else "square"
            )
            tooltip = node.name
            label = node.name
            is_special = True
        else:
            color = color_override.get(nid, CLUSTER_COLORS[nid % len(CLUSTER_COLORS)])
            symbol = "circle"
            label = node.name
            outgoing = graph.transition_probs.get(nid, {})
            top_out = sorted(outgoing.items(), key=lambda kv: -kv[1])[:3]
            out_lines = "\n".join(
                f"  → {graph.nodes[t].name}: {p:.0%}" for t, p in top_out if t in graph.nodes
            )
            tooltip = (
                f"{node.name}\n"
                f"Episodes: {node.num_episodes}  •  Timesteps: {node.num_windows}\n"
                f"Top transitions:\n{out_lines}"
            )
            is_special = False
        # The drawn label only — `tooltip` above keeps the full name.
        if shorten_labels:
            label = short_node_label(label)
        node_dict: dict = {
            "id": nid,
            "label": label,
            "color": color,
            "symbol": symbol,
            "num_episodes": node.num_episodes,
            "num_timesteps": node.num_windows,
            "is_special": is_special,
            "tooltip": tooltip,
        }
        if nid in fold_state:
            node_dict["fold_state"] = fold_state[nid]
        nodes.append(node_dict)

    edges = []
    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob < max(prob_floor, min_edge_prob):
                continue
            cnt = graph.transition_counts.get(src, {}).get(tgt, 0)
            if min_edge_count > 0 and cnt < min_edge_count:
                continue
            if src in pos and tgt in pos:
                edge = {"src": src, "tgt": tgt, "prob": round(prob, 4)}
                if 0 < cnt <= rare_edge_max_count:
                    edge["rare"] = True
                edges.append(edge)

    positions = {str(nid): list(xy) for nid, xy in pos.items()}

    return json.dumps({"nodes": nodes, "edges": edges, "positions": positions})


def render_graph_component(
    graph: BehaviorGraph,
    height: int = 580,
    key: str = "behavior_graph",
    highlighted_path: list[int] | None = None,
    mp4_dir: Path | None = None,
    excluded_node_ids: frozenset[int] = frozenset(),
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    pos: dict[int, tuple[float, float]] | None = None,
    symbol_override: dict[int, str] | None = None,
    color_override: dict[int, str] | None = None,
    theme: str | None = None,
    edge_style: str = "lines",
    edge_width_slope: float = 5.0,
    node_size_slope: float = 16.0,
    prob_floor: float = 0.01,
    rare_edge_max_count: int = 0,
    fold_state: dict[int, str] | None = None,
    annotations: list[dict] | None = None,
    static_figure: bool = False,
    padding: dict[str, int] | None = None,
    shorten_labels: bool = True,
) -> int | None:
    """Render the custom SVG behavior graph component.

    Args:
        annotations: Callout lines, ``[{node|edge, text, side, dx, dy}]``, drawn
            from a node's rim (or an edge's midpoint) to a label at the margin.
            For a figure explaining the encoding rather than a graph to read.
        static_figure: Draw it as a figure — no dragging, no click-to-select, no
            "click any node to inspect" line.
        shorten_labels: Draw node names verbatim when False ("Behavior 2" rather
            than "B2"), for a figure with room for them.
        theme: ``"light"`` or ``"dark"``; resolved from Streamlit's own theme
            when None. It used to default to ``"dark"``, which on the reviewer
            site drew the callouts in near-white on a white canvas.
        padding: ``{left, right, top, bottom}`` in px to keep clear of the
            nodes. The layout is normalised to fill the canvas, so this — not
            the positions — is how a caller reserves room for callout text.
        highlighted_path: List of node IDs forming a path to highlight.
        mp4_dir: Directory containing MP4 files for thumbnail extraction.
        excluded_node_ids: Nodes to hide (computed by compute_pruned_graph_nodes).
        fold_state: Node id -> "collapsed" | "expanded", draws a clickable
            +/- badge. A badge click reports itself via
            ``st.session_state[f"{key}_fold_toggle"]`` instead of the normal
            selection, so it never also triggers the video/node panel for
            what's still a real, clickable node. The caller (e.g. the
            trajectory tree) interprets the toggle based on what the clicked
            id currently means (reveal a rare-branch bucket, fold one back,
            or hide/reveal an arbitrary node's subtree).

    Returns:
        The node_id that was clicked, or the previously selected node.
    """
    pos = component_layout(graph, pos, excluded_node_ids)
    # No node thumbnails: extracting them cost 3-10s of imageio decoding per
    # rerun for something only visible on tooltip-hover, and the panel below the
    # graph shows the videos anyway.
    graph_json = _build_graph_json(
        graph,
        pos,
        fold_state=fold_state,
        min_edge_prob=min_edge_prob,
        min_edge_count=min_edge_count,
        symbol_override=symbol_override,
        color_override=color_override,
        prob_floor=prob_floor,
        rare_edge_max_count=rare_edge_max_count,
        shorten_labels=shorten_labels,
    )
    selected = st.session_state.get(f"{key}_selected")
    selected_edge = st.session_state.get(f"{key}_selected_edge")
    last_seq = st.session_state.get(f"{key}_last_seq", -1)

    # X-button handlers bump `{key}_render_token` so iframe args differ and
    # Streamlit pushes a fresh render (otherwise the iframe's local state
    # would not update). Click-sequence numbers (see JS clickSeq) make
    # repeat clicks on the same node distinct so the persisted sendValue
    # replay doesn't get treated as a fresh click after dismiss.
    render_token = st.session_state.get(f"{key}_render_token", 0)

    clicked = _graph_component(
        graph_json=graph_json,
        height=height,
        selected_node_id=selected,
        selected_edge=list(selected_edge) if selected_edge else None,
        highlighted_path=highlighted_path,
        render_token=render_token,
        theme=theme if theme is not None else str(st.get_option("theme.base") or "dark"),
        edge_style=edge_style,
        edge_width_slope=float(edge_width_slope),
        node_size_slope=float(node_size_slope),
        # A legend for the drawing, drawn by the drawing (see the annotations
        # layer in _graph_component/index.html): callout lines from the nodes
        # and edges to the feature each one demonstrates.
        annotations=annotations or [],
        static_figure=static_figure,
        padding=padding,
        key=key,
        default=None,
    )

    # Protocol from JS:
    #   {svg_export: xml}          — auto-published SVG snapshot for export
    #                                (dispatched first; dicts never look
    #                                like a click).
    #   [node_id, seq]             — node click (node_id == -1 = deselect;
    #                                node_id <= _FOLD_TOGGLE_OFFSET = a
    #                                +/- fold-badge click, see below)
    #   [src, tgt, seq]            — edge click
    #
    # Fold-badge clicks are encoded into the same [id, seq] list channel as
    # node clicks (via the large negative _FOLD_TOGGLE_OFFSET, decoded
    # below) rather than a separate {fold_toggle: ...} dict message. An
    # earlier version used a distinct dict-shaped message with its own
    # st.rerun() call here; that reliably desynced the component after the
    # first click ("Received component message for unregistered
    # ComponentInstance" in the browser console — the dict shape apparently
    # confuses this raw-postMessage component protocol's value tracking in
    # a way plain lists don't). Reusing the exact list-based mechanism
    # node/edge clicks already use avoids that entirely.
    if isinstance(clicked, dict) and "svg_export" in clicked:
        xml = str(clicked.get("svg_export", ""))
        if xml and xml != st.session_state.get("captured_svg"):
            st.session_state["captured_svg"] = xml
            st.session_state[f"{key}_svg_export"] = xml
            # No st.rerun() here: this message is a passive snapshot of
            # whatever the iframe just rendered, not something the caller
            # is waiting on — the download button reads captured_svg later
            # in this same script run, so it already sees the fresh value.
            # Forcing a rerun for it turned every node/edge click into a
            # click -> rerun -> (fresh render, SVG changed) -> rerun cascade,
            # roughly doubling perceived click latency for zero benefit.
        return selected

    if isinstance(clicked, list) and len(clicked) >= 2:
        try:
            seq = int(clicked[-1])
        except (TypeError, ValueError):
            seq = None
        if seq is not None and seq != last_seq:
            st.session_state[f"{key}_last_seq"] = seq
            if len(clicked) == 2:
                try:
                    node_id = int(clicked[0])
                except (TypeError, ValueError):
                    return selected
                if node_id <= _FOLD_TOGGLE_OFFSET:
                    # A +/- fold-badge click (see _FOLD_TOGGLE_OFFSET):
                    # reported separately from normal selection so it never
                    # also opens the video/node panel for what's still a
                    # real, clickable node. The caller reads
                    # f"{key}_fold_toggle" and interprets what the toggle
                    # means for this particular node.
                    st.session_state[f"{key}_fold_toggle"] = (
                        _FOLD_TOGGLE_OFFSET - node_id,
                        seq,
                    )
                    st.rerun(scope="fragment")
                if node_id == -1:
                    _cleared = bool(
                        st.session_state.pop(f"{key}_selected", None) is not None
                        or st.session_state.pop(f"{key}_selected_edge", None) is not None
                    )
                    # key = f"{key_prefix}_graph" — also clear path selection
                    _kp = key[:-6] if key.endswith("_graph") else key
                    for _k in (
                        "_path_ep_list",
                        "_highlighted_path",
                        "_path_label",
                        "_path_synth_ids",
                    ):
                        if st.session_state.pop(f"{_kp}{_k}", None) is not None:
                            _cleared = True
                    if _cleared:
                        st.rerun(scope="fragment")
                if node_id in graph.nodes:
                    st.session_state[f"{key}_selected"] = node_id
                    st.session_state.pop(f"{key}_selected_edge", None)
                    # Force a second pass so the iframe args carry the
                    # newly-selected node id; otherwise the iframe message
                    # handler overwrites the JS-local halo back to the
                    # pre-click `selected`. The seq dedup prevents the
                    # second pass from re-processing the click.
                    st.rerun(scope="fragment")
            elif len(clicked) == 3:
                try:
                    src_v, tgt_v = int(clicked[0]), int(clicked[1])
                except (TypeError, ValueError):
                    return selected
                st.session_state[f"{key}_selected_edge"] = (src_v, tgt_v)
                st.session_state.pop(f"{key}_selected", None)
                # Same reasoning: ensure args reflect the new edge selection
                # so the iframe halo lands on the correct edge.
                st.rerun(scope="fragment")

    return selected
