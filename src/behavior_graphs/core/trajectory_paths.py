"""Per-episode cluster paths — the sequence view the trajectory tree and
terminal sub-clustering are both built on.

An episode's *path* is its run-length-collapsed cluster sequence with a
terminal token appended (SUCCESS / FAILURE / END), e.g. ``(3, 6, 4, -4)``
for "visited behaviors 3 -> 6 -> 4, then succeeded". This is the same
collapse :meth:`BehaviorGraph.from_window_labels` applies before counting
transitions, and the same one :func:`behavior_graphs.viz.simplification.
build_trajectory_tree` builds its trie out of.

Two representations feed in — a :class:`WindowSet` (pipeline stages) and the
viz layer's legacy per-window metadata dicts — so the collapse itself takes
plain parallel sequences and each caller supplies its own adapter. One
implementation keeps the tree the app renders and the sub-clusters the
pipeline persists talking about the same paths.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from behavior_graphs.core.graph import END_NODE_ID, FAILURE_NODE_ID, SUCCESS_NODE_ID

if TYPE_CHECKING:
    from behavior_graphs.core.types import WindowSet


class EpisodePath(NamedTuple):
    """One episode's collapsed cluster path and its outcome."""

    episode_idx: int
    path: tuple[int, ...]
    success: bool | None


def terminal_token(success: bool | None) -> int:
    """The path-closing node id for an episode outcome."""
    if success is True:
        return SUCCESS_NODE_ID
    if success is False:
        return FAILURE_NODE_ID
    return END_NODE_ID


def collapse_episode_paths(
    episode_ids: Sequence[int],
    sort_keys: Sequence[float],
    labels: np.ndarray,
    successes: Mapping[int, bool | None],
) -> list[EpisodePath]:
    """Collapse per-window cluster labels into one path per episode.

    The three sequences are parallel and window-aligned: window ``i`` belongs
    to episode ``episode_ids[i]``, orders within that episode by
    ``sort_keys[i]``, and carries cluster ``labels[i]``. Noise windows
    (``label == -1``) are dropped, consecutive equal labels collapse into one
    visit, and ``successes[ep]`` picks the terminal token.

    Episodes are returned in first-seen order, so callers that need a stable
    order across runs get one without sorting. An episode whose windows are
    all noise still yields a path — just the bare terminal token — matching
    the trajectory tree, where such an episode still reaches an outcome.
    """
    labels = np.asarray(labels)
    # setdefault, not append-only: an all-noise episode must still get an
    # (empty) entry so it ends up with a bare-terminal-token path rather than
    # disappearing, matching the trajectory tree, where it still reaches an
    # outcome. `order` preserves first-seen episode order.
    per_episode: dict[int, list[tuple[float, int]]] = {}
    order: list[int] = []
    for i, raw_ep in enumerate(episode_ids):
        ep = int(raw_ep)
        if ep not in per_episode:
            per_episode[ep] = []
            order.append(ep)
        label = int(labels[i])
        if label == -1:
            continue
        per_episode[ep].append((sort_keys[i], label))

    paths: list[EpisodePath] = []
    for ep in order:
        entries = sorted(per_episode[ep], key=lambda e: e[0])
        seq: list[int] = []
        for _, label in entries:
            if not seq or seq[-1] != label:
                seq.append(label)
        success = successes.get(ep)
        seq.append(terminal_token(success))
        paths.append(EpisodePath(ep, tuple(seq), success))
    return paths


def episode_paths_from_windows(labels: np.ndarray, windows: WindowSet) -> list[EpisodePath]:
    """:func:`collapse_episode_paths` over a :class:`WindowSet`."""
    successes = {ep.index: ep.success for ep in windows.episodes}
    return collapse_episode_paths(
        [w.episode_idx for w in windows.windows],
        [w.start for w in windows.windows],
        labels,
        successes,
    )


def terminal_leaf_groups(
    paths: Sequence[EpisodePath], min_episodes: int = 1
) -> dict[tuple[int, ...], list[int]]:
    """Group episodes by their full path — the trajectory tree's leaves.

    Every episode ends at exactly one leaf, and a leaf's episode set is
    exactly the episodes sharing its full path, so grouping on the path is
    the whole trie's leaf layer without building the trie. Groups smaller
    than *min_episodes* are dropped (splitting two episodes into
    sub-clusters says nothing).

    Returns ``{path: sorted episode indices}``, iterating in descending
    group size then path order so the biggest leaves come first.
    """
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for ep in paths:
        groups[ep.path].append(ep.episode_idx)
    kept = {path: sorted(eps) for path, eps in groups.items() if len(eps) >= min_episodes}
    return dict(sorted(kept.items(), key=lambda kv: (-len(kv[1]), kv[0])))
