"""The construction of a behavior graph, one stage at a time.

A reader who has only seen the finished graph has been shown the conclusion. The
figures here are the three steps in front of it, drawn from the artifacts a
pipeline run already writes, so each one is the run's own data rather than an
illustration of it:

``embedding_figure``
    Every temporal window of every rollout, in the two dimensions the reduction
    produced. Grey, it is the unstructured pile the method starts from; coloured
    by cluster, it is the behaviors it found.
``segment_figure``
    The same assignment read along time instead of across the embedding: one row
    per rollout, banded by behavior. This is what "collapse consecutive windows
    into segments" means, and it is where a reader can see that the
    decomposition lines up with the phases of the task.

Both legends run in :func:`behavior_order` -- the order the behaviors start in
-- rather than in the order k-means happened to number its centroids.

The graph itself is not here. It is drawn by the trajectory-tree component,
which is the same one the study's participants read, and swapping in a
reimplementation for the sake of a matching code path would mean the website
showed a graph nobody was ever shown.

No Streamlit imports, as everywhere in this package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from behavior_graphs.viz.plotly.clusters import CLUSTER_COLORS

#: What a window looks like before the clustering step has an opinion about it.
#: Dark enough to read on the white page this site serves.
UNASSIGNED_COLOR = "#6c757d"


def behavior_color(behavior: int) -> str:
    """The colour of one behavior, wrapping for a K past the palette's length.

    The same palette the demo app's cluster views use, so a behavior keeps its
    colour between the two.
    """
    return CLUSTER_COLORS[int(behavior) % len(CLUSTER_COLORS)]


#: Characters per legend line. The generated captions run to 67 characters on
#: the cooking task, and fifteen of those on one legend take more width than
#: the plot beside it; Plotly does not wrap legend text, so it is wrapped here.
_LEGEND_WRAP = 34


def wrap_label(text: str, width: int = _LEGEND_WRAP) -> str:
    """*text* broken onto lines of about *width*, at word boundaries."""
    lines: list[str] = [""]
    for word in text.split():
        if lines[-1] and len(lines[-1]) + 1 + len(word) > width:
            lines.append(word)
        else:
            lines[-1] = f"{lines[-1]} {word}".strip()
    return "<br>".join(line for line in lines if line)


def behavior_label(behavior: int, names: Mapping[int, str] | None = None) -> str:
    """What to call a behavior: its generated caption, or a number.

    No index in front of the caption. The legends are ordered by when a
    behavior happens (:func:`behavior_order`), and a cluster id printed in that
    order reads as a scrambled list; the graph component numbers nothing
    either, so there is nothing for it to cross-reference. Where a caption
    repeats, ``viz.labels.number_repeats`` has already suffixed it before it
    gets here.

    The fallback is numbered from one, the arrays being zero-based and a reader
    never being shown the zero.
    """
    name = (names or {}).get(int(behavior))
    return wrap_label(name) if name else f"Behavior {int(behavior) + 1}"


def behavior_order(labels: Sequence[int], metadata: Sequence[Mapping]) -> list[int]:
    """The behaviors by when they start, earliest first.

    A legend in cluster-id order is in the order k-means happened to number its
    centroids, which is no order at all. Sorted by start time it reads as the
    phases of the task, which is also the order the progress estimator stacks
    the behaviors in (``stages.progress``: it orders them by mean time of
    occurrence to accumulate each one's intercept).

    "Start" is per rollout the first window the behavior appears in, as a
    fraction of that rollout's length, averaged over the rollouts that visited
    it. For a behavior the policy enters once that is just when it happens; for
    one it returns to, the first visit is what "starts" means, and a mean over
    every visit would drag a recurring early behavior toward the middle.
    """
    firsts: dict[int, list[float]] = {}
    per_rollout: dict[int, list[tuple[int, int]]] = {}
    for index, entry in enumerate(metadata):
        rollout = int(entry["rollout_idx"])
        per_rollout.setdefault(rollout, []).append((int(entry["window_start"]), int(labels[index])))
    for windows in per_rollout.values():
        windows.sort()
        total = len(windows)
        seen: set[int] = set()
        for position, behavior in enumerate(windows):
            if behavior[1] in seen:
                continue
            seen.add(behavior[1])
            firsts.setdefault(behavior[1], []).append(position / total)
    # Ties broken by cluster id, so the order is stable rather than dependent
    # on dict iteration.
    return sorted(firsts, key=lambda b: (sum(firsts[b]) / len(firsts[b]), b))


def embedding_figure(
    coords: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Mapping],
    names: Mapping[int, str] | None = None,
    *,
    clustered: bool = True,
    height: int = 460,
    max_points: int = 12000,
) -> go.Figure:
    """The reduced windows as a scatter, grey or coloured by behavior.

    *clustered* is the before and after of one pipeline stage drawn on one set
    of points, which is the comparison worth making: the geometry does not move
    when the clustering runs, only the colour.

    Thinned to *max_points* by an even stride when a run has more. A run of 25k
    windows renders as a solid mass either way, and the stride keeps every
    rollout represented rather than dropping the tail of the collection.
    """
    coords = np.asarray(coords, dtype=float)
    labels = np.asarray(labels).astype(int)
    # Before the thinning: the order is a property of the run, not of whatever
    # survives a stride.
    order = behavior_order(labels, metadata)
    stride = max(1, int(np.ceil(len(coords) / max_points)))
    coords, labels = coords[::stride], labels[::stride]

    figure = go.Figure()
    if not clustered:
        figure.add_trace(
            go.Scattergl(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={"size": 3, "color": UNASSIGNED_COLOR, "opacity": 0.75},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        drawn = set(labels.tolist())
        for behavior in [b for b in order if b in drawn]:
            mask = labels == behavior
            figure.add_trace(
                go.Scattergl(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=behavior_label(behavior, names),
                    marker={"size": 3, "color": behavior_color(behavior), "opacity": 0.7},
                    hovertemplate="%{fullData.name}<extra></extra>",
                )
            )
    # No tick labels: the axes are two directions a reduction chose and the
    # numbers on them mean nothing a reader could use.
    figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    figure.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"itemsizing": "constant", "font": {"size": 11}},
        showlegend=clustered,
    )
    return figure


def segments(labels: Sequence[int], metadata: Sequence[Mapping]) -> dict[int, list[tuple]]:
    """``{rollout: [(start, end, behavior)]}`` in fractions of the rollout.

    Consecutive windows carrying the same behavior are one segment, which is the
    collapsing step of the construction: it is what turns a per-window labelling
    into the short sequence a graph is built over. Only *consecutive* ones — a
    rollout that returns to an earlier behavior visited it twice, and merging
    the two would draw one band over whatever happened in between.

    Measured in **windows**, as a fraction of how many the rollout has: window
    *i* of *n* spans ``i/n`` to ``(i+1)/n``. That makes the segments tile the
    row exactly and makes two rollouts of different lengths comparable, which
    an absolute axis would not — a row per rollout drawn in frames says more
    about duration than about structure. Frame indices are the wrong unit for
    it in any case, because the windows overlap: at width W and stride S the
    last window still covers W frames after its start, so dividing starts by
    the last start puts the final segment at zero width.

    ``window_start`` is read for the order only. The metadata arrives grouped
    by rollout already, but nothing in the file format promises that.
    """
    per_rollout: dict[int, list[tuple[int, int]]] = {}
    for index, entry in enumerate(metadata):
        rollout = int(entry["rollout_idx"])
        per_rollout.setdefault(rollout, []).append((int(entry["window_start"]), int(labels[index])))

    out: dict[int, list[tuple[float, float, int]]] = {}
    for rollout, windows in per_rollout.items():
        windows.sort()
        total = len(windows)
        runs: list[tuple[float, float, int]] = []
        for position, (_, behavior) in enumerate(windows):
            start, end = position / total, (position + 1) / total
            if runs and runs[-1][2] == behavior:
                # Extend the open run over this window rather than opening a
                # second one beside it.
                runs[-1] = (runs[-1][0], end, behavior)
            else:
                runs.append((start, end, behavior))
        out[rollout] = runs
    return out


def segment_figure(
    labels: Sequence[int],
    metadata: Sequence[Mapping],
    names: Mapping[int, str] | None = None,
    *,
    row_height: int = 13,
    min_height: int = 320,
) -> go.Figure:
    """One row per rollout, banded by the behavior it was in.

    **Every** rollout, in **rollout order**. Two earlier versions of this got
    both of those wrong: one drew the first forty of sixty, which on the
    cooking task left three of its fifteen behaviors with no band at all
    (they occur only in rollouts that were not drawn); the other sorted the
    rows by outcome, which groups the failures into a block and makes the
    chart look like a finding rather than like the data.
    """
    bands = segments(labels, metadata)
    order = sorted(bands)
    rows = {rollout: position for position, rollout in enumerate(order)}
    height = max(min_height, 70 + row_height * len(order))

    figure = go.Figure()
    drawn = {b for rollout in order for _, _, b in bands[rollout]}
    for behavior in [b for b in behavior_order(labels, metadata) if b in drawn]:
        starts, widths, positions = [], [], []
        for rollout in order:
            for start, end, band in bands[rollout]:
                if band != behavior:
                    continue
                starts.append(start)
                widths.append(end - start)
                positions.append(rows[rollout])
        figure.add_trace(
            go.Bar(
                x=widths,
                y=positions,
                base=starts,
                orientation="h",
                name=behavior_label(behavior, names),
                marker={"color": behavior_color(behavior), "line": {"width": 0}},
                hovertemplate="%{fullData.name}<extra></extra>",
            )
        )
    figure.update_layout(
        # Overlay, because every bar carries its own `base`. Plotly excludes
        # base-setting traces from a stack and draws them overlaid anyway, so
        # saying "stack" here was a claim about the figure that was not true of
        # it -- and one a future edit could have made true by dropping `base`.
        barmode="overlay",
        bargap=0.25,
        height=height,
        margin={"l": 30, "r": 10, "t": 10, "b": 50},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # `normal`, because a stacked horizontal bar chart otherwise lists its
        # legend bottom-up and the behaviors read 11 down to 1.
        legend={"font": {"size": 11}, "traceorder": "normal"},
    )
    figure.update_xaxes(
        title_text="Fraction of the rollout", range=[0, 1], tickformat=".0%", showgrid=False
    )
    # Reversed, so rollout 0 is the top row rather than the bottom one.
    figure.update_yaxes(
        showticklabels=False,
        showgrid=False,
        range=[-0.6, len(order) - 0.4],
        autorange="reversed",
    )
    return figure
