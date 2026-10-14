"""Plotly scatter plots of windowed embeddings in 2-D.

Pure functions: arrays + metadata in, ``plotly.graph_objects.Figure`` out.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from behavior_graphs.core.types import WindowSet


def _hover_text(windows: WindowSet) -> list:
    texts = []
    for w in windows.windows:
        success = windows.episodes[w.episode_idx].success
        outcome = {True: "success", False: "failure", None: "?"}[success]
        texts.append(f"episode {w.episode_idx} ({outcome})<br>t ∈ [{w.start}, {w.end})")
    return texts


def cluster_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    windows: WindowSet | None = None,
    title: str | None = None,
    height: int = 600,
) -> go.Figure:
    """2-D scatter colored by cluster label (noise label ``-1`` in grey)."""
    labels = np.asarray(labels)
    hover = _hover_text(windows) if windows is not None else None
    fig = go.Figure()
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        fig.add_trace(
            go.Scattergl(
                x=coords_2d[mask, 0],
                y=coords_2d[mask, 1],
                mode="markers",
                name="noise" if label == -1 else f"cluster {label}",
                marker=dict(
                    size=5,
                    color="#cccccc" if label == -1 else None,
                ),
                hovertext=[hover[i] for i in np.flatnonzero(mask)] if hover else None,
                hoverinfo="text" if hover else None,
            )
        )
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        legend=dict(itemsizing="constant"),
    )
    return fig


def value_scatter(
    coords_2d: np.ndarray,
    values: np.ndarray,
    windows: WindowSet | None = None,
    colorbar_title: str = "value",
    title: str | None = None,
    height: int = 600,
) -> go.Figure:
    """2-D scatter colored by a continuous per-window quantity (Q, advantage...)."""
    hover = _hover_text(windows) if windows is not None else None
    fig = go.Figure(
        go.Scattergl(
            x=coords_2d[:, 0],
            y=coords_2d[:, 1],
            mode="markers",
            marker=dict(
                size=5,
                color=np.asarray(values, dtype=float),
                colorscale="RdYlGn",
                colorbar=dict(title=colorbar_title),
            ),
            hovertext=hover,
            hoverinfo="text" if hover else None,
        )
    )
    fig.update_layout(title=title, height=height, xaxis_title="dim 1", yaxis_title="dim 2")
    return fig


def episode_label_strip(
    labels: np.ndarray,
    windows: WindowSet,
    episode_idx: int,
    height: int = 120,
) -> go.Figure:
    """One episode's cluster-label sequence over time as a colored strip."""
    items = [
        (w.start, int(labels[i]))
        for i, w in enumerate(windows.windows)
        if w.episode_idx == episode_idx
    ]
    items.sort()
    starts = [s for s, _ in items]
    labs = [lab for _, lab in items]
    fig = go.Figure(
        go.Heatmap(
            z=[labs],
            x=starts,
            showscale=False,
            colorscale="Turbo",
            hovertext=[[f"t={s}: cluster {lab}" for s, lab in items]],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        height=height,
        yaxis=dict(visible=False),
        xaxis_title="timestep",
        margin=dict(l=10, r=10, t=10, b=30),
    )
    return fig
