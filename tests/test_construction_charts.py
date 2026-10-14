"""The construction walkthrough: the three stages in front of the graph.

Small fixtures throughout. These test what the figures *say* — that the
clustering stage moves no point, that a segment is a run of like-labelled
windows, that the two outcome groups end up in separate blocks — rather than
what one particular run happens to look like.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavior_graphs.viz.plotly import construction


def _metadata(rollouts: dict[int, tuple[bool, int]]) -> list[dict]:
    """``{rollout: (success, n windows)}`` as the metadata a run carries."""
    rows = []
    for rollout, (success, count) in rollouts.items():
        rows += [
            {
                "rollout_idx": rollout,
                "window_start": index,
                "window_end": index + 8,
                "success": success,
            }
            for index in range(count)
        ]
    return rows


# ── Naming and colour ────────────────────────────────────────────────────────


def test_a_captioned_behavior_is_called_by_its_caption_alone():
    """No cluster id in front of it: the legends run in start order, where an
    id reads as a scrambled list, and the graph component numbers nothing for
    it to cross-reference. The fallback is numbered from one, the arrays being
    zero-based and a reader never being shown the zero."""
    assert construction.behavior_label(0, {0: "reaching"}) == "reaching"
    assert construction.behavior_label(0) == "Behavior 1"


def test_a_behavior_past_the_palettes_end_still_gets_a_colour():
    """K is swept up to 20 and a run with more would otherwise raise."""
    assert construction.behavior_color(0) == construction.behavior_color(
        len(construction.CLUSTER_COLORS)
    )


def test_behaviors_are_coloured_as_the_demo_apps_cluster_views_colour_them():
    from behavior_graphs.viz.plotly.clusters import CLUSTER_COLORS

    assert construction.behavior_color(3) == CLUSTER_COLORS[3]


# ── The embedding ────────────────────────────────────────────────────────────


def test_clustering_moves_no_point():
    """The before and after of one stage on one set of points, which is the
    whole comparison: if the geometry changed between them, the chart would be
    saying the reduction ran twice."""
    coords = np.random.default_rng(0).normal(size=(200, 2))
    labels = np.arange(200) % 4
    metadata = _metadata({0: (True, 200)})
    grey = construction.embedding_figure(coords, labels, metadata, clustered=False)
    coloured = construction.embedding_figure(coords, labels, metadata, clustered=True)
    assert len(grey.data) == 1
    assert len(coloured.data) == 4
    assert sorted(x for trace in grey.data for x in trace.x) == pytest.approx(
        sorted(x for trace in coloured.data for x in trace.x)
    )


def test_an_oversized_run_is_thinned_by_a_stride_not_a_truncation():
    """A stride keeps every rollout represented; a head slice would drop the
    tail of the collection and quietly draw a different set of rollouts."""
    coords = np.column_stack([np.arange(5000.0), np.arange(5000.0)])
    figure = construction.embedding_figure(
        coords, np.zeros(5000, int), _metadata({0: (True, 5000)}), max_points=1000
    )
    drawn = sorted(figure.data[0].x)
    assert len(drawn) <= 1000
    # Spans the whole run rather than its first fifth.
    assert drawn[0] == 0.0
    assert drawn[-1] > 4900


def test_the_embedding_axes_carry_no_tick_labels():
    """They are two directions a reduction chose; the numbers on them are not
    a quantity a reader can use."""
    figure = construction.embedding_figure(
        np.zeros((10, 2)), np.zeros(10, int), _metadata({0: (True, 10)})
    )
    assert figure.layout.xaxis.showticklabels is False
    assert figure.layout.yaxis.showticklabels is False


# ── Segments ─────────────────────────────────────────────────────────────────


def test_consecutive_windows_of_one_behavior_are_one_segment():
    """The collapsing step: four windows in two behaviors are two segments,
    not four."""
    labels = [0, 0, 1, 1]
    found = construction.segments(labels, _metadata({7: (True, 4)}))
    assert [behavior for _, _, behavior in found[7]] == [0, 1]


def test_a_behavior_revisited_later_is_a_second_segment():
    """Collapsing is of *consecutive* windows. A rollout that goes back to an
    earlier behavior visited it twice, and a chart merging the two would draw
    one band over the top of whatever happened in between."""
    found = construction.segments([0, 1, 0], _metadata({0: (True, 3)}))
    assert [behavior for _, _, behavior in found[0]] == [0, 1, 0]


def test_segments_span_the_rollout_from_end_to_end():
    found = construction.segments([0, 0, 1], _metadata({0: (True, 3)}))
    spans = found[0]
    assert spans[0][0] == 0.0
    assert spans[-1][1] == 1.0
    # And they tile it: each one starts where the last ended.
    assert [round(end, 6) for _, end, _ in spans[:-1]] == [
        round(start, 6) for start, _, _ in spans[1:]
    ]


def test_rollouts_are_measured_in_fractions_so_two_lengths_are_comparable():
    """A short rollout and a long one that go through the same behaviors in
    the same proportions draw the same row."""
    short = construction.segments([0, 0, 1, 1], _metadata({0: (True, 4)}))
    long = construction.segments([0] * 20 + [1] * 20, _metadata({0: (True, 40)}))
    assert [round(end, 2) for _, end, _ in short[0]] == [round(end, 2) for _, end, _ in long[0]]


def test_rollouts_are_drawn_in_their_own_order():
    """Not sorted by outcome. Sorting groups the failures into a block, which
    makes the chart look like a finding rather than like the data."""
    metadata = _metadata({2: (True, 2), 0: (False, 2), 1: (True, 2)})
    labels = [0, 1] * 3
    figure = construction.segment_figure(labels, metadata)
    # Reversed, so rollout 0 is the top row.
    assert figure.layout.yaxis.autorange == "reversed"
    assert {tuple(trace.y) for trace in figure.data} == {(0, 1, 2)}


def test_the_segment_legend_reads_downwards():
    """A stacked horizontal bar chart lists its legend bottom-up by default,
    which numbers the behaviors backwards."""
    figure = construction.segment_figure([0, 1], _metadata({0: (True, 2)}))
    assert figure.layout.legend.traceorder == "normal"
    assert [trace.name for trace in figure.data] == ["Behavior 1", "Behavior 2"]


def test_every_rollout_is_drawn():
    """An earlier version drew the first forty of sixty, which on the cooking
    task left three of its fifteen behaviors with no band at all: they occur
    only in rollouts that were not drawn. The height grows with the count
    instead."""
    metadata = _metadata({index: (index % 2 == 0, 2) for index in range(90)})
    figure = construction.segment_figure([0, 1] * 90, metadata)
    assert {len(trace.y) for trace in figure.data} == {90}
    assert figure.layout.height > 90 * 10


def test_a_behavior_is_never_dropped_for_sitting_in_a_late_rollout():
    """The failure the rollout cap caused, as a test: behavior 2 occurs in one
    rollout only, and that rollout is the last."""
    metadata = _metadata({0: (True, 2), 1: (True, 2), 2: (True, 2)})
    labels = [0, 1, 0, 1, 2, 2]
    figure = construction.segment_figure(labels, metadata)
    assert len(figure.data) == 3


def test_long_captions_are_wrapped_so_the_legend_keeps_its_width():
    """Plotly does not wrap legend text, and the generated captions run to 67
    characters; fifteen of those on one legend take more width than the plot
    beside it."""
    long = "stalling above the metal bowl when attempting to pick up the carrot"
    wrapped = construction.behavior_label(0, {0: long})
    assert "<br>" in wrapped
    assert max(len(line) for line in wrapped.split("<br>")) <= 34
    # And the words survive the wrapping.
    assert wrapped.replace("<br>", " ") == long


# ── Legend order ─────────────────────────────────────────────────────────────


def test_behaviors_are_ordered_by_when_they_start():
    """A legend in cluster-id order is in the order k-means happened to number
    its centroids, which is no order at all. Behavior 2 runs first here, so it
    leads."""
    labels = [2, 2, 0, 0, 1]
    assert construction.behavior_order(labels, _metadata({0: (True, 5)})) == [2, 0, 1]


def test_the_order_is_the_first_visit_not_the_mean_of_every_visit():
    """A behavior the policy returns to late still starts early, and a mean
    over every visit would drag it toward the middle. Here 0 starts first and
    comes back last; it must still lead."""
    labels = [0, 1, 1, 1, 0]
    assert construction.behavior_order(labels, _metadata({0: (True, 5)})) == [0, 1]


def test_the_order_is_averaged_over_the_rollouts_that_visited_each_behavior():
    """One rollout where 1 comes first and three where 0 does: 0 leads."""
    metadata = _metadata({0: (True, 2), 1: (True, 2), 2: (True, 2), 3: (True, 2)})
    labels = [1, 0] + [0, 1] * 3
    assert construction.behavior_order(labels, metadata) == [0, 1]


def test_a_tie_is_broken_by_cluster_id_so_the_order_is_stable():
    """Two behaviors starting at the same point must not come out in whatever
    order a dict iterated."""
    metadata = _metadata({0: (True, 2), 1: (True, 2)})
    assert construction.behavior_order([3, 1, 1, 3], metadata) == [1, 3]


def test_both_legends_run_in_that_order():
    """The scatter and the ribbons are two readings of one assignment, and a
    reader moving between them should not have to re-find a behavior."""
    labels = [2, 2, 0, 0, 1]
    metadata = _metadata({0: (True, 5)})
    coords = np.random.default_rng(0).normal(size=(5, 2))
    scatter = construction.embedding_figure(coords, np.array(labels), metadata)
    ribbons = construction.segment_figure(labels, metadata)
    expected = [construction.behavior_label(b) for b in (2, 0, 1)]
    assert [trace.name for trace in scatter.data] == expected
    assert [trace.name for trace in ribbons.data] == expected
