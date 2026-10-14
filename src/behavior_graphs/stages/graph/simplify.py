"""Graph simplification: edge pruning and degree-one node merging.

Ported from policy_doctor's ``behavior_graph.graph`` /
``graph_simplification`` and adapted to :class:`WindowSet`-based graphs.
"""

from __future__ import annotations

import logging

import numpy as np

from behavior_graphs.core.graph import SPECIAL_NODE_IDS, BehaviorGraph
from behavior_graphs.core.types import WindowSet

logger = logging.getLogger(__name__)


def prune_edges_by_count(graph: BehaviorGraph, min_count: int) -> BehaviorGraph:
    """Drop transitions with count < ``min_count`` and renormalize probabilities."""
    new_counts: dict[int, dict[int, int]] = {}
    for src, targets in graph.transition_counts.items():
        kept = {tgt: c for tgt, c in targets.items() if c >= min_count}
        if kept:
            new_counts[src] = kept
    new_probs: dict[int, dict[int, float]] = {
        src: {tgt: c / total for tgt, c in targets.items()}
        for src, targets in new_counts.items()
        if (total := sum(targets.values())) > 0
    }
    return BehaviorGraph(
        nodes=dict(graph.nodes),
        transition_counts=new_counts,
        transition_probs=new_probs,
        num_episodes=graph.num_episodes,
    )


def _follow_merge_chain(cluster_id: int, merge_map: dict[int, int]) -> int:
    """Resolve ``cluster_id`` through a merge map (path compression, cycle-safe)."""
    seen: set[int] = set()
    cur = cluster_id
    while cur in merge_map:
        if cur in seen:
            return cur
        seen.add(cur)
        cur = merge_map[cur]
    return cur


def apply_merge_map(labels: np.ndarray, merge_map: dict[int, int]) -> np.ndarray:
    """Rewrite ``labels`` so each id is replaced by its merge-chain root."""
    if not merge_map:
        return labels.copy()
    resolved = {
        int(u): -1 if u == -1 else _follow_merge_chain(int(u), merge_map) for u in np.unique(labels)
    }
    out = labels.copy()
    for old, new in resolved.items():
        if old != new:
            out[labels == old] = new
    return out


def degree_one_prune_to_fixed_point(
    labels: np.ndarray, windows: WindowSet
) -> tuple[BehaviorGraph, np.ndarray, int, int, dict[int, int]]:
    """Merge cluster nodes with a single in- or out-neighbor until none remain.

    Each round rebuilds the graph from the current labels, merges every
    non-special node whose only out-neighbor (or, failing that, only
    in-neighbor) is another cluster node into that neighbor, and relabels.

    Args:
        labels: (n_windows,) cluster labels aligned with ``windows.windows``.
        windows: The window set the labels were computed on.

    Returns:
        ``(graph, new_labels, num_rounds, num_nodes_merged, full_merge_map)``;
        the input labels are not modified. ``full_merge_map`` composes every
        round's merges into one original-id -> final-id map (via
        :func:`apply_merge_map`), so the same relabeling can be replayed on
        new raw labels that share this cluster-id space (e.g. held-out data
        from the same fitted clusterer).
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    full_map: dict[int, int] = {int(c): int(c) for c in np.unique(labels) if c != -1}
    num_rounds = 0
    num_merged = 0
    max_rounds = max(500, len(np.unique(labels)) * 20)
    while num_rounds < max_rounds:
        graph = BehaviorGraph.from_window_labels(labels, windows)
        merge_map: dict[int, int] = {}
        for nid in sorted(graph.nodes):
            if nid in SPECIAL_NODE_IDS:
                continue
            in_sources = [s for s, c, _ in graph.get_incoming_transitions(nid) if c > 0]
            out_targets = [t for t, c, _ in graph.get_outgoing_transitions(nid) if c > 0]
            if len(out_targets) == 1:
                repl = out_targets[0]
            elif len(in_sources) == 1:
                repl = in_sources[0]
            else:
                continue
            if repl in SPECIAL_NODE_IDS:
                continue
            merge_map[nid] = repl
        if not merge_map:
            break
        num_rounds += 1
        num_merged += len(merge_map)
        new_labels = apply_merge_map(labels, merge_map)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for orig, cur in full_map.items():
            full_map[orig] = _follow_merge_chain(cur, merge_map)
    else:
        graph = BehaviorGraph.from_window_labels(labels, windows)
    if num_merged:
        logger.info("degree-one pruning merged %d nodes in %d rounds", num_merged, num_rounds)
    return graph, labels, num_rounds, num_merged, full_map
