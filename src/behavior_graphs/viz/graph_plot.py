"""Plotly rendering of a behavior graph.

Pure functions: data in, ``plotly.graph_objects.Figure`` out. No streamlit.
Ported (condensed) from policy_doctor's plotting/plotly/behavior_graph.py.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)

_SPECIAL_COLORS = {
    START_NODE_ID: "#636efa",
    SUCCESS_NODE_ID: "#00cc96",
    FAILURE_NODE_ID: "#ef553b",
    END_NODE_ID: "#aaaaaa",
}
_CLUSTER_COLOR = "#ffa15a"


def layered_layout(graph: BehaviorGraph) -> dict[int, tuple[float, float]]:
    """Left-to-right layout: x = BFS depth from START, y = barycenter-refined.

    Terminal nodes are pinned to the rightmost layer.
    """
    depth: dict[int, int] = {START_NODE_ID: 0}
    queue = deque([START_NODE_ID])
    while queue:
        node = queue.popleft()
        for tgt in graph.transition_probs.get(node, {}):
            if tgt not in depth:
                depth[tgt] = depth[node] + 1
                queue.append(tgt)
    for nid in graph.nodes:
        depth.setdefault(nid, 1)
    max_depth = max(depth.values()) if depth else 1
    for nid in graph.terminal_node_ids:
        depth[nid] = max_depth = max(max_depth, max(depth.values()) + 0)
        depth[nid] = max(depth.values())
    terminal_depth = max(depth.values()) + 1 if graph.terminal_node_ids else max_depth
    for nid in graph.terminal_node_ids:
        depth[nid] = terminal_depth

    layers: dict[int, list[int]] = {}
    for nid, d in depth.items():
        layers.setdefault(d, []).append(nid)
    pos: dict[int, tuple[float, float]] = {}
    for d, nodes in layers.items():
        for i, nid in enumerate(sorted(nodes)):
            pos[nid] = (float(d), float(i) - (len(nodes) - 1) / 2.0)

    neighbors: dict[int, list[int]] = {nid: [] for nid in graph.nodes}
    for src, targets in graph.transition_probs.items():
        for tgt in targets:
            neighbors.setdefault(src, []).append(tgt)
            neighbors.setdefault(tgt, []).append(src)
    for _ in range(4):  # barycenter refinement to reduce edge crossings
        for d, nodes in layers.items():
            order = sorted(
                nodes,
                key=lambda n: (
                    np.mean([pos[m][1] for m in neighbors[n]]) if neighbors[n] else pos[n][1]
                ),
            )
            for i, nid in enumerate(order):
                pos[nid] = (pos[nid][0], float(i) - (len(nodes) - 1) / 2.0)
    return pos


def _bezier(
    p0: tuple[float, float], p1: tuple[float, float], curvature: float = 0.18, n: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Quadratic bezier between two points, bowed perpendicular to the chord."""
    p0, p1 = np.asarray(p0), np.asarray(p1)
    mid = (p0 + p1) / 2.0
    chord = p1 - p0
    norm = np.linalg.norm(chord)
    perp = np.array([-chord[1], chord[0]]) / (norm + 1e-9)
    ctrl = mid + curvature * norm * perp
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t**2 * p1
    return pts[:, 0], pts[:, 1]


def behavior_graph_figure(
    graph: BehaviorGraph,
    node_values: dict[int, float] | None = None,
    min_edge_probability: float = 0.0,
    highlight_path: Sequence[int] | None = None,
    title: str | None = None,
    height: int = 640,
) -> go.Figure:
    """Render the behavior graph.

    Cluster nodes are sized by episode coverage and colored by ``node_values``
    when given (RdYlGn); special nodes keep fixed colors. Edge width and
    opacity scale with transition probability; ``highlight_path`` (a node-id
    sequence as returned by ``enumerate_paths``) is drawn on top in blue.
    """
    pos = layered_layout(graph)
    fig = go.Figure()
    highlight_edges = set()
    if highlight_path:
        highlight_edges = set(zip(highlight_path[:-1], highlight_path[1:]))

    for src, targets in graph.transition_probs.items():
        for tgt, prob in targets.items():
            if prob < min_edge_probability or src not in pos or tgt not in pos:
                continue
            xs, ys = _bezier(pos[src], pos[tgt])
            highlighted = (src, tgt) in highlight_edges
            count = graph.transition_counts.get(src, {}).get(tgt, 0)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(
                        width=2.5 + 6.0 * prob if highlighted else 1.0 + 5.0 * prob,
                        color="#1f77ff" if highlighted else "rgba(120,120,120,0.45)",
                    ),
                    opacity=1.0 if highlighted else max(0.25, prob),
                    hoverinfo="text",
                    text=f"{_name(graph, src)} → {_name(graph, tgt)}<br>p={prob:.2f} (n={count})",
                    showlegend=False,
                )
            )

    values = node_values or {}
    cluster_ids = [nid for nid, n in graph.nodes.items() if not n.is_special]
    vmin = min((values[c] for c in cluster_ids if c in values), default=-1.0)
    vmax = max((values[c] for c in cluster_ids if c in values), default=1.0)
    xs, ys, sizes, colors, texts, labels = [], [], [], [], [], []
    for nid, node in graph.nodes.items():
        if nid not in pos:
            continue
        x, y = pos[nid]
        xs.append(x)
        ys.append(y)
        frac = node.num_episodes / max(graph.num_episodes, 1)
        sizes.append(18 + 30 * frac)
        if node.is_special:
            colors.append(_SPECIAL_COLORS[nid])
        elif nid in values and vmax > vmin:
            colors.append(_value_color((values[nid] - vmin) / (vmax - vmin)))
        else:
            colors.append(_CLUSTER_COLOR)
        value_str = f"<br>V={values[nid]:.3f}" if nid in values else ""
        texts.append(
            f"<b>{node.name}</b><br>{node.num_episodes}/{graph.num_episodes} episodes"
            f"<br>{node.num_windows} windows{value_str}"
        )
        labels.append(node.name if node.is_special else str(nid))
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(size=sizes, color=colors, line=dict(width=1.5, color="#333")),
            text=labels,
            textposition="top center",
            hoverinfo="text",
            hovertext=texts,
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=50 if title else 20, b=20),
    )
    return fig


def _name(graph: BehaviorGraph, nid: int) -> str:
    node = graph.nodes.get(nid)
    return node.name if node else str(nid)


def _value_color(t: float) -> str:
    """Red (t=0) → yellow → green (t=1)."""
    t = float(np.clip(t, 0.0, 1.0))
    # two-segment interpolation: (239,85,59) -> (253,231,37) -> (0,204,150)
    if t < 0.5:
        a = t * 2
        rgb = (239 + (253 - 239) * a, 85 + (231 - 85) * a, 59 + (37 - 59) * a)
    else:
        a = (t - 0.5) * 2
        rgb = (253 + (0 - 253) * a, 231 + (204 - 231) * a, 37 + (150 - 37) * a)
    return f"rgb({int(rgb[0])},{int(rgb[1])},{int(rgb[2])})"
