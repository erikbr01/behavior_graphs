"""Per-episode cluster paths — the collapse the trajectory tree and the
subclustering stage share."""

from __future__ import annotations

import numpy as np

from behavior_graphs.core.graph import END_NODE_ID, FAILURE_NODE_ID, SUCCESS_NODE_ID
from behavior_graphs.core.trajectory_paths import (
    collapse_episode_paths,
    episode_paths_from_windows,
    terminal_leaf_groups,
)
from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet
from behavior_graphs.viz.compat import meta_from_windows
from behavior_graphs.viz.simplification import build_trajectory_tree


def _windows(ep_labels: dict[int, list[int]], successes: dict[int, bool | None]) -> tuple:
    """One-timestep windows per episode from ``{ep: [label, ...]}``."""
    episodes = [
        EpisodeInfo(
            index=ep,
            num_samples=len(labels),
            sample_start_idx=0,
            sample_end_idx=len(labels),
            success=successes.get(ep),
        )
        for ep, labels in sorted(ep_labels.items())
    ]
    windows, labels = [], []
    for ep in sorted(ep_labels):
        for t, label in enumerate(ep_labels[ep]):
            windows.append(WindowInfo(episode_idx=ep, start=t, end=t + 1))
            labels.append(label)
    features = np.zeros((len(windows), 2), dtype=np.float32)
    return np.asarray(labels, dtype=np.int64), WindowSet(
        features=features, windows=windows, episodes=episodes
    )


class TestCollapseEpisodePaths:
    def test_run_length_collapse_and_terminal_token(self):
        labels, windows = _windows({0: [1, 1, 2, 2, 2, 3]}, {0: True})
        (path,) = episode_paths_from_windows(labels, windows)
        assert path.episode_idx == 0
        assert path.path == (1, 2, 3, SUCCESS_NODE_ID)
        assert path.success is True

    def test_failure_and_unknown_outcomes(self):
        labels, windows = _windows({0: [1], 1: [1]}, {0: False, 1: None})
        paths = {p.episode_idx: p.path for p in episode_paths_from_windows(labels, windows)}
        assert paths[0] == (1, FAILURE_NODE_ID)
        assert paths[1] == (1, END_NODE_ID)

    def test_noise_windows_are_dropped_not_treated_as_a_visit(self):
        # The -1 between the two 4s must not split them into 4 -> 4.
        labels, windows = _windows({0: [4, -1, 4]}, {0: True})
        (path,) = episode_paths_from_windows(labels, windows)
        assert path.path == (4, SUCCESS_NODE_ID)

    def test_all_noise_episode_still_reaches_its_outcome(self):
        labels, windows = _windows({0: [-1, -1]}, {0: False})
        (path,) = episode_paths_from_windows(labels, windows)
        assert path.path == (FAILURE_NODE_ID,)

    def test_windows_are_ordered_by_sort_key_not_input_order(self):
        # Same episode, windows supplied out of temporal order.
        labels = np.asarray([2, 1], dtype=np.int64)
        paths = collapse_episode_paths([0, 0], [5, 0], labels, {0: True})
        assert paths[0].path == (1, 2, SUCCESS_NODE_ID)

    def test_episodes_come_back_in_first_seen_order(self):
        labels = np.asarray([1, 1, 1], dtype=np.int64)
        paths = collapse_episode_paths([7, 3, 7], [0, 0, 1], labels, {})
        assert [p.episode_idx for p in paths] == [7, 3]


class TestTerminalLeafGroups:
    def test_groups_episodes_sharing_a_full_path(self):
        labels, windows = _windows(
            {0: [1, 2], 1: [1, 2], 2: [1, 3]},
            {0: True, 1: True, 2: True},
        )
        groups = terminal_leaf_groups(episode_paths_from_windows(labels, windows))
        assert groups[(1, 2, SUCCESS_NODE_ID)] == [0, 1]
        assert groups[(1, 3, SUCCESS_NODE_ID)] == [2]

    def test_same_behaviors_different_outcome_are_different_leaves(self):
        labels, windows = _windows({0: [1], 1: [1]}, {0: True, 1: False})
        groups = terminal_leaf_groups(episode_paths_from_windows(labels, windows))
        assert set(groups) == {(1, SUCCESS_NODE_ID), (1, FAILURE_NODE_ID)}

    def test_min_episodes_drops_small_leaves(self):
        labels, windows = _windows(
            {0: [1], 1: [1], 2: [2]},
            {0: True, 1: True, 2: True},
        )
        groups = terminal_leaf_groups(episode_paths_from_windows(labels, windows), min_episodes=2)
        assert list(groups) == [(1, SUCCESS_NODE_ID)]

    def test_ordered_by_descending_size(self):
        labels, windows = _windows(
            {0: [1], 1: [1], 2: [2], 3: [3], 4: [3], 5: [3]},
            dict.fromkeys(range(6), True),
        )
        groups = terminal_leaf_groups(episode_paths_from_windows(labels, windows))
        assert [len(v) for v in groups.values()] == [3, 2, 1]

    def test_every_episode_lands_in_exactly_one_leaf(self):
        labels, windows = _windows(
            {0: [1, 2], 1: [1], 2: [-1], 3: [2, 2]},
            {0: True, 1: False, 2: True, 3: None},
        )
        groups = terminal_leaf_groups(episode_paths_from_windows(labels, windows))
        assert sorted(ep for eps in groups.values() for ep in eps) == [0, 1, 2, 3]


def test_agrees_with_the_trajectory_tree_the_app_renders():
    """The stage's leaves must be the tree's leaves, or a persisted split
    could never be looked up by the leaf a user expanded."""
    labels, windows = _windows(
        {0: [1, 1, 2], 1: [1, 2], 2: [1, 3], 3: [-1, 2], 4: [2]},
        {0: True, 1: True, 2: False, 3: None, 4: False},
    )
    groups = terminal_leaf_groups(episode_paths_from_windows(labels, windows))

    tree = build_trajectory_tree(labels, meta_from_windows(windows))
    by_path = {tuple(nd["path"]): nd for nd in tree}
    parents = {tuple(nd["parent_path"]) for nd in tree if nd["parent_path"] is not None}
    tree_leaves = {p: sorted(by_path[p]["episode_indices"]) for p in by_path if p not in parents}

    assert groups == tree_leaves
