"""Unit tests for on-demand terminal/leaf sub-clustering."""

import numpy as np
import pytest

from behavior_graphs.core.subclusters import (
    UNCLUSTERED,
    assign_episode_subclusters,
    available_subcluster_configs,
    load_subcluster_clustering,
    load_subcluster_reduction,
    reduce_for_subclustering,
    stable_subcluster_id,
    terminal_last_segment_window_spans,
    terminal_last_segment_windows,
)
from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet
from behavior_graphs.stages.clustering.kmeans import KMeansClustering


def _build(ep_windows: dict[int, list[tuple[int, list[float]]]]) -> tuple[np.ndarray, WindowSet]:
    """``ep_windows[ep]`` is an ordered list of ``(label, feature)`` for
    consecutive one-timestep-wide windows in that episode."""
    max_ep = max(ep_windows)
    episodes = [
        EpisodeInfo(
            index=i,
            num_samples=len(ep_windows.get(i, [])),
            sample_start_idx=0,
            sample_end_idx=len(ep_windows.get(i, [])),
        )
        for i in range(max_ep + 1)
    ]
    windows = []
    labels = []
    rows = []
    for ep in sorted(ep_windows):
        for t, (label, feat) in enumerate(ep_windows[ep]):
            windows.append(WindowInfo(episode_idx=ep, start=t, end=t + 1))
            labels.append(label)
            rows.append(feat)
    return (
        np.asarray(labels, dtype=np.int64),
        WindowSet(features=np.asarray(rows, dtype=np.float32), windows=windows, episodes=episodes),
    )


class TestTerminalLastSegmentWindows:
    def test_multi_window_trailing_segment(self):
        labels, windows = _build({0: [(3, [0.0]), (3, [0.1]), (5, [1.0]), (5, [1.1]), (5, [1.2])]})
        ep_ids, feats = terminal_last_segment_windows([0], labels, windows)
        assert ep_ids == [0, 0, 0]
        assert sorted(feats[:, 0].tolist()) == pytest.approx([1.0, 1.1, 1.2])

    def test_single_window_trailing_segment(self):
        labels, windows = _build({0: [(2, [0.0]), (7, [9.0])]})
        ep_ids, feats = terminal_last_segment_windows([0], labels, windows)
        assert ep_ids == [0]
        assert feats[:, 0].tolist() == pytest.approx([9.0])

    def test_noise_gap_within_trailing_label_run(self):
        # Noise windows are skipped entirely, so the two label-4 windows on
        # either side of the noise window are still one trailing run.
        labels, windows = _build({0: [(4, [1.0]), (-1, [999.0]), (4, [2.0])]})
        ep_ids, feats = terminal_last_segment_windows([0], labels, windows)
        assert ep_ids == [0, 0]
        assert sorted(feats[:, 0].tolist()) == pytest.approx([1.0, 2.0])

    def test_ignores_episodes_and_windows_outside_request(self):
        labels, windows = _build(
            {
                0: [(1, [0.0]), (2, [1.0])],
                1: [(1, [5.0]), (2, [6.0])],
            }
        )
        ep_ids, _feats = terminal_last_segment_windows([0], labels, windows)
        assert set(ep_ids) == {0}

    def test_episode_with_only_noise_contributes_nothing(self):
        labels, windows = _build({0: [(-1, [0.0]), (-1, [1.0])], 1: [(3, [2.0])]})
        ep_ids, feats = terminal_last_segment_windows([0, 1], labels, windows)
        assert ep_ids == [1]
        assert feats.shape == (1, 1)

    def test_empty_result_has_correct_feature_width(self):
        labels, windows = _build({0: [(-1, [0.0, 0.0])]})
        ep_ids, feats = terminal_last_segment_windows([0], labels, windows)
        assert ep_ids == []
        assert feats.shape == (0, 2)


class TestTerminalLastSegmentWindowSpans:
    def test_spans_align_with_features_row_for_row(self):
        # Same fixture as test_multi_window_trailing_segment: the trailing
        # label-5 run is timesteps 2, 3, 4 -> spans (2,3), (3,4), (4,5).
        labels, windows = _build({0: [(3, [0.0]), (3, [0.1]), (5, [1.0]), (5, [1.1]), (5, [1.2])]})
        feat_ep_ids, feats = terminal_last_segment_windows([0], labels, windows)
        span_ep_ids, spans = terminal_last_segment_window_spans([0], labels, windows)
        assert span_ep_ids == feat_ep_ids
        # Row i's span covers the same window as row i's feature.
        by_feat_value = {round(f, 3): s for f, s in zip(feats[:, 0].tolist(), spans)}
        assert by_feat_value[1.0] == (2, 3)
        assert by_feat_value[1.1] == (3, 4)
        assert by_feat_value[1.2] == (4, 5)

    def test_empty_result(self):
        labels, windows = _build({0: [(-1, [0.0])]})
        ep_ids, spans = terminal_last_segment_window_spans([0], labels, windows)
        assert ep_ids == []
        assert spans == []


class TestAssignEpisodeSubclusters:
    def test_conservation_and_majority_vote(self):
        window_eps = [0, 0, 0, 1, 1, 3, 3]
        sub_labels = np.array([1, 1, 2, 1, 2, -1, -1])
        groups = assign_episode_subclusters([0, 1, 2, 3], window_eps, sub_labels)
        all_eps = sorted(e for members in groups.values() for e in members)
        assert all_eps == [0, 1, 2, 3]
        # episode 0: 2 votes for label 1 vs 1 for label 2 -> label 1.
        # episode 1: 1 vote each for label 1 and 2 -> tie breaks to lower id, label 1.
        assert groups[1] == [0, 1]
        assert 2 not in groups  # episode 2 never appears in window_eps at all
        assert UNCLUSTERED in groups
        assert set(groups[UNCLUSTERED]) == {2, 3}

    def test_tie_breaks_to_lower_label(self):
        window_eps = [5, 5]
        sub_labels = np.array([3, 1])
        groups = assign_episode_subclusters([5], window_eps, sub_labels)
        assert groups == {1: [5]}

    def test_noise_winner_groups_as_unclustered(self):
        window_eps = [0, 0, 0]
        sub_labels = np.array([-1, -1, 2])
        groups = assign_episode_subclusters([0], window_eps, sub_labels)
        assert groups == {UNCLUSTERED: [0]}


class TestStableSubclusterId:
    def test_deterministic(self):
        a = stable_subcluster_id("(1, 2, -5)", 0)
        b = stable_subcluster_id("(1, 2, -5)", 0)
        assert a == b

    def test_distinguishes_sub_label_and_parent(self):
        ids = {
            stable_subcluster_id("(1, 2, -5)", 0),
            stable_subcluster_id("(1, 2, -5)", 1),
            stable_subcluster_id("(1, 3, -5)", 0),
        }
        assert len(ids) == 3

    def test_never_collides_with_real_or_terminal_ids(self):
        real_and_terminal_ids = set(range(-5, 20))
        for k in range(50):
            assert stable_subcluster_id("some/path", k) not in real_and_terminal_ids

    def test_collision_probing_is_shared_via_used_ids(self):
        used: set[int] = set()
        first = stable_subcluster_id("p", 0, used_ids=used)
        # Force a collision by pre-seeding the id the next call would land on.
        used.add(stable_subcluster_id("p", 1, used_ids=set()))
        second = stable_subcluster_id("p", 1, used_ids=used)
        assert first != second
        assert second in used


class TestSubclusterConfigs:
    def test_available_configs_lists_kmeans(self):
        assert {"kmeans"} <= set(available_subcluster_configs())

    def test_load_kmeans_matches_yaml_defaults(self):
        clustering = load_subcluster_clustering("kmeans")
        assert isinstance(clustering, KMeansClustering)
        assert clustering.n_clusters == 3
        assert clustering.n_init == 10

    def test_load_with_override(self):
        clustering = load_subcluster_clustering("kmeans", n_clusters=5)
        assert clustering.n_clusters == 5

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError):
            load_subcluster_clustering("not-a-real-algorithm")


class TestReduceForSubclustering:
    def test_noop_below_min_points(self):
        feats = np.random.default_rng(0).normal(size=(3, 50)).astype(np.float32)
        out = reduce_for_subclustering(feats, n_components=10)
        assert out is feats

    def test_noop_when_target_not_smaller(self):
        feats = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
        out = reduce_for_subclustering(feats, n_components=10)
        assert out is feats

    def test_reduces_dimensionality(self):
        pytest.importorskip("umap")
        feats = np.random.default_rng(0).normal(size=(50, 100)).astype(np.float32)
        out = reduce_for_subclustering(feats, n_components=10)
        assert out.shape == (50, 10)

    def test_clamps_n_components_to_n_points_minus_one(self):
        pytest.importorskip("umap")
        feats = np.random.default_rng(0).normal(size=(6, 100)).astype(np.float32)
        out = reduce_for_subclustering(feats, n_components=10)
        assert out.shape[0] == 6
        assert out.shape[1] < 100

    def test_load_umap_matches_yaml_defaults(self):
        pytest.importorskip("umap")
        reducer = load_subcluster_reduction("umap", n_components=4)
        assert reducer.n_components == 4

    def test_pca_ignores_umap_only_neighbor_override(self):
        feats = np.random.default_rng(0).normal(size=(12, 20)).astype(np.float32)
        out = reduce_for_subclustering(feats, n_components=3, reduction="pca")
        assert out.shape == (12, 3)

    def test_identity_ignores_unsupported_overrides(self):
        feats = np.random.default_rng(0).normal(size=(12, 20)).astype(np.float32)
        out = reduce_for_subclustering(feats, n_components=3, reduction="identity")
        assert out.shape == feats.shape


class TestEndToEnd:
    def test_gather_cluster_assign_separates_two_groups(self):
        # Episodes 0-2 end near (0, 0); episodes 3-5 end near (10, 10).
        ep_windows = {}
        for ep in range(3):
            ep_windows[ep] = [(1, [0.0 + 0.01 * ep, 0.0])]
        for ep in range(3, 6):
            ep_windows[ep] = [(1, [10.0 + 0.01 * ep, 10.0])]
        labels, windows = _build(ep_windows)
        episode_indices = list(range(6))

        ep_ids, feats = terminal_last_segment_windows(episode_indices, labels, windows)
        clustering = load_subcluster_clustering("kmeans", n_clusters=2, n_init=5)
        sub_labels = clustering.fit_predict(feats)
        groups = assign_episode_subclusters(episode_indices, ep_ids, sub_labels)

        all_eps = sorted(e for members in groups.values() for e in members)
        assert all_eps == episode_indices
        member_sets = [set(v) for v in groups.values()]
        assert {0, 1, 2} in member_sets
        assert {3, 4, 5} in member_sets
