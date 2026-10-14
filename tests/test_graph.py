"""Unit tests for graph construction, simplification, and smoothing."""

import numpy as np
import pytest

from behavior_graphs.core.graph import (
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)
from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths
from behavior_graphs.stages.graph import (
    TransitionGraphBuilder,
    apply_merge_map,
    degree_one_prune_to_fixed_point,
    median_filter,
    prune_edges_by_count,
    sticky_decode,
)
from behavior_graphs.stages.graph.smoothing import cluster_centroids


def make_window_set(label_seqs, successes, dim=2):
    """One window per timestep; features placed at the label's centroid."""
    lengths = [len(seq) for seq in label_seqs]
    episodes = episodes_from_lengths(lengths, successes)
    windows = []
    rows = []
    for ep_idx, seq in enumerate(label_seqs):
        for t, label in enumerate(seq):
            windows.append(WindowInfo(episode_idx=ep_idx, start=t, end=t + 1))
            rows.append(np.full(dim, float(label), dtype=np.float32))
    features = np.asarray(rows, dtype=np.float32)
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in label_seqs])
    return labels, WindowSet(features=features, windows=windows, episodes=episodes)


def assert_probs_normalized(graph):
    for src, targets in graph.transition_probs.items():
        assert sum(targets.values()) == pytest.approx(1.0), f"node {src}"


class TestBehaviorGraph:
    def test_from_window_labels_structure(self):
        labels, windows = make_window_set([[0, 0, 1, 1], [0, 1, 1, 2]], successes=[True, False])
        graph = BehaviorGraph.from_window_labels(labels, windows)
        assert START_NODE_ID in graph.nodes
        assert set(graph.cluster_nodes) == {0, 1, 2}
        assert graph.num_episodes == 2
        assert_probs_normalized(graph)
        # START -> 0 in both episodes.
        assert graph.transition_probs[START_NODE_ID][0] == pytest.approx(1.0)

    def test_save_load_roundtrip(self, tmp_path):
        labels, windows = make_window_set(
            [[0, 1, 2], [0, 2, 2], [1, 1, 2]], successes=[True, False, None]
        )
        graph = BehaviorGraph.from_window_labels(labels, windows)
        path = tmp_path / "graph.json"
        graph.save(path)
        loaded = BehaviorGraph.load(path)
        assert loaded.transition_counts == graph.transition_counts
        assert loaded.transition_probs == graph.transition_probs
        assert loaded.num_episodes == graph.num_episodes
        assert set(loaded.nodes) == set(graph.nodes)
        for nid in graph.nodes:
            assert loaded.nodes[nid].to_dict() == graph.nodes[nid].to_dict()

    def test_compute_values_finite(self):
        labels, windows = make_window_set([[0, 1], [0, 1], [0, 2]], successes=[True, True, False])
        graph = BehaviorGraph.from_window_labels(labels, windows)
        values = graph.compute_values()
        assert set(values) >= set(graph.cluster_nodes)
        assert all(np.isfinite(v) for v in values.values())
        assert values[SUCCESS_NODE_ID] == 1.0


class TestSmoothing:
    def test_median_filter_removes_flicker(self):
        labels = np.array([0, 0, 1, 0, 0, 2, 2, 2], dtype=np.int64)
        out = median_filter(labels, window=3)
        assert out[2] == 0  # flicker removed
        assert list(out[5:]) == [2, 2, 2]
        assert labels[2] == 1  # input untouched

    def test_median_filter_window_one_is_identity(self):
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        out = median_filter(labels, window=1)
        assert np.array_equal(out, labels)
        assert out is not labels

    def test_median_filter_ignores_noise(self):
        labels = np.array([0, -1, 0, 0], dtype=np.int64)
        out = median_filter(labels, window=3)
        assert list(out) == [0, 0, 0, 0]

    def test_sticky_decode_reduces_transitions(self):
        labels = np.array([0, 0, 1, 0, 0, 1, 1, 1], dtype=np.int64)
        features = np.array(
            [[0.0], [0.1], [0.4], [0.0], [0.1], [1.0], [0.9], [1.1]], dtype=np.float32
        )
        centroids = cluster_centroids(labels, features)
        out = sticky_decode(labels, features, centroids, lambda_stick=5.0)
        n_trans = int((np.diff(out) != 0).sum())
        assert n_trans <= int((np.diff(labels) != 0).sum())
        assert set(np.unique(out)) <= {0, 1}
        assert labels[2] == 1  # input untouched

    def test_sticky_decode_empty_centroids(self):
        labels = np.array([-1, -1], dtype=np.int64)
        out = sticky_decode(labels, np.zeros((2, 1), dtype=np.float32), {}, 1.0)
        assert np.array_equal(out, labels)


class TestSimplify:
    def test_prune_edges_renormalizes(self):
        labels, windows = make_window_set(
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            successes=[True, True, True, False],
        )
        graph = BehaviorGraph.from_window_labels(labels, windows)
        # node 0 -> {1: 3, 2: 1}
        pruned = prune_edges_by_count(graph, min_count=2)
        assert 2 not in pruned.transition_probs[0]
        assert pruned.transition_probs[0][1] == pytest.approx(1.0)
        assert_probs_normalized(pruned)
        # Original graph untouched.
        assert 2 in graph.transition_probs[0]

    def test_apply_merge_map_chain(self):
        labels = np.array([0, 1, 2, -1], dtype=np.int64)
        out = apply_merge_map(labels, {0: 1, 1: 2})
        assert list(out) == [2, 2, 2, -1]

    def test_degree_one_prune(self):
        # Chain 0 -> 1 -> 2 -> {3 or 4}: nodes 0 and 1 each have a single
        # cluster out-neighbor, so the chain head merges downstream.
        labels, windows = make_window_set([[0, 1, 2, 3], [0, 1, 2, 4]], successes=[True, False])
        graph = BehaviorGraph.from_window_labels(labels, windows)
        new_graph, new_labels, rounds, merged, merge_map = degree_one_prune_to_fixed_point(
            labels, windows
        )
        assert merged >= 1
        assert len(new_graph.cluster_nodes) < len(graph.cluster_nodes)
        assert_probs_normalized(new_graph)
        # The composed map resolves every original id to where it ended up.
        for orig, cur in zip(labels, new_labels):
            assert merge_map[int(orig)] == int(cur)
        # Input labels untouched.
        assert 1 in labels


class TestTransitionGraphBuilder:
    def test_build_does_not_mutate_labels(self):
        labels, windows = make_window_set(
            [[0, 0, 1, 0, 0], [0, 0, 0, 1, 1]], successes=[True, False]
        )
        original = labels.copy()
        builder = TransitionGraphBuilder(smoothing="median", smoothing_window=3)
        graph = builder.build(labels, windows)
        assert np.array_equal(labels, original)
        assert builder.last_labels_ is not None
        assert_probs_normalized(graph)
        # Median smoothing removed the lone flicker in episode 0.
        assert builder.last_labels_[2] == 0

    def test_build_sticky(self):
        labels, windows = make_window_set([[0, 1, 0, 0], [1, 1, 0, 1]], successes=[True, False])
        builder = TransitionGraphBuilder(smoothing="sticky", sticky_lambda=10.0)
        graph = builder.build(labels, windows)
        smoothed = builder.last_labels_
        # Strong stickiness: each episode collapses to a single label.
        for lo, hi in [(0, 4), (4, 8)]:
            assert len(np.unique(smoothed[lo:hi])) == 1
        assert_probs_normalized(graph)

    def test_min_edge_count(self):
        labels, windows = make_window_set(
            [[0, 1], [0, 1], [0, 1], [0, 2]],
            successes=[True, True, True, False],
        )
        graph = TransitionGraphBuilder(min_edge_count=2).build(labels, windows)
        assert 2 not in graph.transition_probs.get(0, {})
        assert_probs_normalized(graph)

    def test_degree_one_prune_updates_labels(self):
        labels, windows = make_window_set([[0, 1, 2, 3], [0, 1, 2, 4]], successes=[True, False])
        builder = TransitionGraphBuilder(degree_one_prune=True)
        graph = builder.build(labels, windows)
        assert len(graph.cluster_nodes) < 5
        assert not np.array_equal(builder.last_labels_, labels)

    def test_invalid_smoothing(self):
        with pytest.raises(ValueError):
            TransitionGraphBuilder(smoothing="hmm")

    def test_relabel_matches_last_labels_on_same_data(self):
        # Same fixture as test_degree_one_prune_updates_labels: relabeling
        # the exact labels build() was given must reproduce last_labels_ --
        # this is what held-out data needs to land in the graph's id space.
        labels, windows = make_window_set([[0, 1, 2, 3], [0, 1, 2, 4]], successes=[True, False])
        builder = TransitionGraphBuilder(degree_one_prune=True)
        builder.build(labels, windows)
        assert np.array_equal(builder.relabel(labels, windows), builder.last_labels_)

    def test_relabel_unseen_label_passes_through(self):
        labels, windows = make_window_set([[0, 1, 2, 3], [0, 1, 2, 4]], successes=[True, False])
        builder = TransitionGraphBuilder(degree_one_prune=True)
        builder.build(labels, windows)
        # 99 was never seen during training, so there's nothing learned to
        # remap it to -- it must pass through unchanged, not raise or vanish.
        held_out_labels, held_out_windows = make_window_set([[99]], successes=[True])
        assert builder.relabel(held_out_labels, held_out_windows)[0] == 99

    def test_relabel_sticky_reuses_train_centroids_not_held_out_data(self):
        train_labels, train_windows = make_window_set(
            [[0, 0, 1, 1], [0, 0, 1, 1]], successes=[True, True]
        )
        builder = TransitionGraphBuilder(smoothing="sticky", sticky_lambda=0.0)
        builder.build(train_labels, train_windows)
        assert builder._sticky_centroids is not None
        cluster1_centroid = builder._sticky_centroids[1]

        # Held-out episode: both windows raw-labeled 0, but with features
        # that sit exactly at cluster 1's *train* centroid. Reusing the
        # train centroids must decode both to cluster 1 (nearest centroid
        # wins with sticky_lambda=0); re-deriving centroids from just this
        # held-out pair (both labeled 0) could only ever produce a single
        # candidate centroid equal to those very points, forcing the
        # decode to stay at 0 -- this is exactly what the bug did.
        held_out_labels = np.array([0, 0], dtype=np.int64)
        held_out_windows = WindowSet(
            features=np.stack([cluster1_centroid, cluster1_centroid]),
            windows=[
                WindowInfo(episode_idx=0, start=0, end=1),
                WindowInfo(episode_idx=0, start=1, end=2),
            ],
            episodes=episodes_from_lengths([2], [True]),
        )
        relabeled = builder.relabel(held_out_labels, held_out_windows)
        assert list(relabeled) == [1, 1]

    def test_relabel_before_build_raises(self):
        # Matches StateMergingGraphBuilder.relabel's guard for the same
        # misuse -- both implementations of the duck-typed relabel
        # protocol must fail fast rather than silently returning unmerged
        # labels.
        builder = TransitionGraphBuilder()
        labels, windows = make_window_set([[0, 1]], successes=[True])
        with pytest.raises(RuntimeError):
            builder.relabel(labels, windows)
