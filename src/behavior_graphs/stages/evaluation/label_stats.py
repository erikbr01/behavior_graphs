"""Temporal label-quality metrics over the per-window cluster labels.

Ported from policy_doctor's ``compute_clustering_metrics._coverage_metrics``
(see also ``docs/graph_evaluation.md`` §2.2.1-2.2.2 there), adapted to
per-window labels + :class:`WindowSet`.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import Evaluator
from behavior_graphs.core.types import WindowSet


def _episode_label_sequences(
    labels: np.ndarray, windows: WindowSet
) -> dict[int, list[tuple[int, int, int]]]:
    """Per-episode ``(start, end, label)`` of non-noise windows, in temporal order."""
    per_episode: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for i, w in enumerate(windows.windows):
        label = int(labels[i])
        if label == -1:
            continue
        per_episode[w.episode_idx].append((w.start, w.end, label))
    for seq in per_episode.values():
        seq.sort(key=lambda x: x[0])
    return per_episode


#: Stand-in for the terminal token `build_trajectory_tree` appends to every
#: episode path (SUCCESS / FAILURE / END). Any value no cluster id can take.
_TERMINAL = -1000


def trie_stats(collapsed: list[list[int]]) -> tuple[int, float]:
    """``(tree_nodes, prefix_sharing)`` for run-length-collapsed sequences.

    A trajectory-tree node is a distinct *prefix*, so the trie can be sized
    without building it, from the sequences this module already has. Two
    details are needed to make the count equal what
    :func:`behavior_graphs.viz.simplification.build_trajectory_tree` draws, and
    both were off by exactly their amount before being added: that builder
    prepends a **root** node and appends a **terminal token** to every episode
    path, so it renders ``1 + prefixes(seq + [terminal])``.

    One terminal per episode is right whenever the episodes share an outcome
    -- true for ABC, where every episode is a demonstration. On a dataset with
    both successes and failures the builder splits the terminal by outcome, so
    two episodes with the same cluster path but different outcomes get two leaf
    nodes and the real count is up to one higher per distinct path. Verified
    equal on chess (24,875) and whiteboard (437).

    ``prefix_sharing = 1 - nodes / symbols`` is how much of their path the
    episodes hold in common. 0 means every episode takes its own route and the
    "graph" is really N separate chains; the annotations themselves run 0.32
    (legos) to 0.82 (whiteboard), so the target is task-specific and a delivered
    0.04 is a structural failure however good the boundaries are.

    Recorded because it is the one quantity measured so far that separates a
    labeling from a random relabeling of the same segmentation: a boundary
    metric scores the `relabel` control at F1@0.75 = 0.941, and sharing is what
    notices. It is also very nearly invariant to the window grid (bottles' truth
    is 2,805 nodes at every width/stride tried), because run-length collapse
    removes the sampling rate.
    """
    prefixes: set[tuple[int, ...]] = set()
    symbols = 0
    for seq in collapsed:
        full = [*seq, _TERMINAL]
        symbols += len(full)
        for i in range(1, len(full) + 1):
            prefixes.add(tuple(full[:i]))
    if not symbols:
        return 0, float("nan")
    nodes = len(prefixes) + 1  # the root
    return nodes, 1.0 - nodes / symbols


class LabelStatsEvaluator(Evaluator):
    """Temporal coherence and behavioral-resolution statistics of the labels.

    Metrics (noise windows, label ``-1``, are excluded throughout):

    - ``swap_rate_per_frame``: stride-fair label-change density — total number
      of adjacent same-episode window pairs with differing labels, divided by
      the total number of observation frames (the highest window ``end`` seen
      per episode). Independent of the windowing stride.
    - ``swap_rate_per_window``: the same swap count divided by the number of
      adjacent window pairs, i.e. the fraction of consecutive same-episode
      window pairs whose labels differ. Not stride-fair; kept for reference.
    - ``distinct_per_episode`` / ``distinct_per_episode_median``: number of
      distinct cluster labels visited per episode after run-length collapse
      (mean / median over episodes). Should match the true number of task
      phases.
    - ``mean_segment_length``: mean length, in windows, of consecutive
      same-label runs. The dual view of the swap rate: long segments mean
      temporally coherent labels.
    - ``cluster_coverage_per_episode``: mean over episodes of the fraction of
      all (global, non-noise) clusters visited by that episode.
    """

    name = "label_stats"

    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        labels = np.asarray(result.labels)
        episodes = _episode_label_sequences(labels, result.windows)
        n_clusters = int(len({int(c) for c in labels if c != -1}))

        n_swaps = 0
        n_pairs = 0
        n_frames = 0
        distinct_counts: list[int] = []
        segment_lengths: list[int] = []
        collapsed: list[list[int]] = []
        for seq in episodes.values():
            labs = [label for _, _, label in seq]
            n_swaps += sum(1 for j in range(1, len(labs)) if labs[j] != labs[j - 1])
            n_pairs += max(len(labs) - 1, 0)
            # Frame total: highest window end seen for the episode.
            n_frames += max(end for _, end, _ in seq)
            # Run-length collapse: one entry per consecutive same-label run.
            rle: list[int] = []
            run_len = 0
            for lab in labs:
                if rle and rle[-1] == lab:
                    run_len += 1
                else:
                    if run_len:
                        segment_lengths.append(run_len)
                    rle.append(lab)
                    run_len = 1
            if run_len:
                segment_lengths.append(run_len)
            distinct_counts.append(len(set(rle)))
            collapsed.append(rle)

        tree_nodes, prefix_share = trie_stats(collapsed)

        out: dict[str, Any] = {
            "n_episodes": len(episodes),
            "n_windows": int(sum(len(seq) for seq in episodes.values())),
            "n_clusters": n_clusters,
            "swap_rate_per_frame": (n_swaps / n_frames) if n_frames > 0 else None,
            "swap_rate_per_window": (n_swaps / n_pairs) if n_pairs > 0 else None,
            "distinct_per_episode": (float(np.mean(distinct_counts)) if distinct_counts else None),
            "distinct_per_episode_median": (
                float(np.median(distinct_counts)) if distinct_counts else None
            ),
            "mean_segment_length": (float(np.mean(segment_lengths)) if segment_lengths else None),
            # Trajectory-tree size and path agreement; see `trie_stats`.
            "tree_nodes": tree_nodes or None,
            "prefix_sharing": prefix_share if prefix_share == prefix_share else None,
            "cluster_coverage_per_episode": (
                float(np.mean([d / n_clusters for d in distinct_counts]))
                if distinct_counts and n_clusters > 0
                else None
            ),
        }
        return out
