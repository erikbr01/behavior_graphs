"""Regression tests for trajectory-tree sub-cluster caching and reuse."""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.subclusters import SubClusterGroup, SubClusterSet
from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet
from behavior_graphs.viz.streamlit.components import trajectory_tree_view as tree


def _windows() -> WindowSet:
    return WindowSet(
        features=np.zeros((1, 2), dtype=np.float32),
        windows=[WindowInfo(episode_idx=0, start=0, end=1)],
        episodes=[
            EpisodeInfo(
                index=0,
                num_samples=1,
                sample_start_idx=0,
                sample_end_idx=1,
                success=False,
            )
        ],
    )


def test_cache_namespace_separates_runs(monkeypatch):
    calls: list[tuple[int, ...]] = []

    def fake_subcluster(episode_indices, *_args, **_kwargs):
        calls.append(tuple(episode_indices))
        return {0: list(episode_indices)}, {}

    monkeypatch.setattr(tree.subclusters, "subcluster_episodes", fake_subcluster)
    tree._cached_subcluster_groups.clear()
    labels = np.asarray([1])
    windows = _windows()
    args = (labels, windows, "run-a", (1, -4), (0,), "kmeans", 2, None, 3)

    tree._cached_subcluster_groups(*args)
    tree._cached_subcluster_groups(*args)
    tree._cached_subcluster_groups(
        labels,
        windows,
        "run-b",
        (1, -4),
        (0,),
        "kmeans",
        2,
        None,
        3,
    )

    assert calls == [(0,), (0,)]
    tree._cached_subcluster_groups.clear()


def test_precomputed_reuse_requires_every_compute_parameter(monkeypatch):
    artifact_groups = {0: [0]}
    artifact_timeline = {0: [(0, 1, 0)]}
    artifact = SubClusterSet(
        params={
            "algorithm": "kmeans",
            "k": 2,
            "reduction": "pca",
            "n_components": 3,
            "keep_window_detail": True,
            "algorithm_config": tree.subclusters.subcluster_algo_config("kmeans"),
            "reduction_config": tree.subclusters.subcluster_reduction_config("pca"),
        },
        groups=[
            SubClusterGroup(
                path=(1, -4),
                episode_subclusters={0: 0},
                window_subclusters=artifact_timeline,
            )
        ],
    )
    live = ({1: [0]}, {0: [(0, 1, 1)]})
    monkeypatch.setattr(tree, "_cached_subcluster_groups", lambda *_args: live)

    matched = tree._subcluster_groups_for_leaf(
        (1, -4),
        [0],
        np.asarray([1]),
        _windows(),
        "run-a",
        "kmeans",
        2,
        "pca",
        3,
        artifact,
    )
    mismatched = tree._subcluster_groups_for_leaf(
        (1, -4),
        [0],
        np.asarray([1]),
        _windows(),
        "run-a",
        "kmeans",
        2,
        "identity",
        3,
        artifact,
    )

    assert matched == (artifact_groups, artifact_timeline)
    assert mismatched == live


def test_artifact_without_window_detail_recomputes(monkeypatch):
    artifact = SubClusterSet(
        params={
            "algorithm": "kmeans",
            "k": 2,
            "reduction": None,
            "n_components": 3,
            "keep_window_detail": False,
            "algorithm_config": tree.subclusters.subcluster_algo_config("kmeans"),
        },
        groups=[SubClusterGroup(path=(1, -4), episode_subclusters={0: 0})],
    )
    live = ({0: [0]}, {0: [(0, 1, 0)]})
    monkeypatch.setattr(tree, "_cached_subcluster_groups", lambda *_args: live)

    result = tree._subcluster_groups_for_leaf(
        (1, -4),
        [0],
        np.asarray([1]),
        _windows(),
        "run-a",
        "kmeans",
        2,
        None,
        3,
        artifact,
    )

    assert result == live
