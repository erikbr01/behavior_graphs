"""Sub-clustering of every terminal leaf of the trajectory tree."""

from __future__ import annotations

import logging

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import SubClustering
from behavior_graphs.core.subclusters import (
    DEFAULT_MIN_SUBCLUSTER_EPISODES,
    DEFAULT_SUBCLUSTER_ALGORITHM,
    DEFAULT_SUBCLUSTER_COMPONENTS,
    DEFAULT_SUBCLUSTER_K,
    DEFAULT_SUBCLUSTER_REDUCTION,
    SubClusterGroup,
    SubClusterSet,
    subcluster_algo_config,
    subcluster_episodes,
    subcluster_reduction_config,
    validate_subcluster_params,
)
from behavior_graphs.core.trajectory_paths import episode_paths_from_windows, terminal_leaf_groups

log = logging.getLogger(__name__)


class TerminalSubClustering(SubClustering):
    """Split each trajectory-tree terminal leaf into feature-space sub-clusters.

    The unit of splitting is a *leaf*: the set of episodes sharing one full
    run-length-collapsed cluster path (e.g. everything that went
    ``3 -> 6 -> 4`` and then failed), not one outcome node. That is finer
    than "all failures" and is exactly the unit the demo app's trajectory
    tree lets a user expand, so a persisted split lines up one-to-one with
    what the app renders.

    Cost scales with leaf count — each leaf refits its own local reducer —
    so ``min_episodes`` doubles as the cost control: leaves below it are
    skipped, and on a long-tailed tree most leaves hold one or two episodes.
    Raise ``max_leaves`` only if you want the tail too.

    Args:
        algorithm: Clusterer config name under
            ``configs/pipeline/subclustering/algo/`` (e.g. ``kmeans``, ``gmm``).
        k: Sub-clusters per leaf, clamped down when a leaf has fewer points.
        reduction: Reducer config name under ``configs/pipeline/reduction/``
            to refit locally per leaf, or null to cluster the raw features.
        n_components: Target dimensionality for that local refit.
        min_episodes: Skip leaves with fewer episodes than this.
        max_leaves: Cap on how many leaves are split, largest first; null
            for no cap.
        keep_window_detail: Persist each episode's per-window sub-labels
            alongside the per-episode vote. Needed for the app's per-window
            timeline; set false to keep ``subclusters.json`` small.
    """

    name = "subclustering"

    def __init__(
        self,
        algorithm: str = DEFAULT_SUBCLUSTER_ALGORITHM,
        k: int = DEFAULT_SUBCLUSTER_K,
        reduction: str | None = DEFAULT_SUBCLUSTER_REDUCTION,
        n_components: int = DEFAULT_SUBCLUSTER_COMPONENTS,
        min_episodes: int = DEFAULT_MIN_SUBCLUSTER_EPISODES,
        max_leaves: int | None = 50,
        keep_window_detail: bool = True,
    ) -> None:
        validate_subcluster_params(
            k=k,
            n_components=n_components,
            min_episodes=min_episodes,
            max_leaves=max_leaves,
        )
        self.algorithm = algorithm
        self.k = k
        self.reduction = reduction
        self.n_components = n_components
        self.min_episodes = min_episodes
        self.max_leaves = max_leaves
        self.keep_window_detail = keep_window_detail

    @property
    def params(self) -> dict:
        """The settings that produced a split, recorded into the artifact."""
        params = {
            "stage": type(self).__name__,
            "algorithm": self.algorithm,
            "k": self.k,
            "reduction": self.reduction,
            "n_components": self.n_components,
            "min_episodes": self.min_episodes,
            "max_leaves": self.max_leaves,
            "keep_window_detail": self.keep_window_detail,
            "algorithm_config": subcluster_algo_config(self.algorithm),
        }
        if self.reduction is not None:
            params["reduction_config"] = subcluster_reduction_config(self.reduction)
        return params

    def cache_fingerprint(self) -> dict:
        return self.params

    def split(self, result: PipelineResult) -> SubClusterSet:
        if result.windows is None or result.labels is None:
            raise ValueError("subclustering needs both windows and labels; run clustering first")

        paths = episode_paths_from_windows(result.labels, result.windows)
        leaves = terminal_leaf_groups(paths, min_episodes=self.min_episodes)
        n_total = len(leaves)
        if self.max_leaves is not None and n_total > self.max_leaves:
            # terminal_leaf_groups orders by descending size, so truncating
            # keeps the leaves worth splitting. Logged rather than silent:
            # a capped run covers less of the tree than its config implies.
            leaves = dict(list(leaves.items())[: self.max_leaves])
            log.warning(
                "subclustering: %d leaves have >= %d episodes but max_leaves=%d; "
                "splitting the %d largest and skipping the rest",
                n_total,
                self.min_episodes,
                self.max_leaves,
                self.max_leaves,
            )

        groups: list[SubClusterGroup] = []
        for path, episode_indices in leaves.items():
            episode_groups, window_timeline = subcluster_episodes(
                episode_indices,
                result.labels,
                result.windows,
                algorithm=self.algorithm,
                k=self.k,
                n_components=self.n_components,
                reduction=self.reduction,
            )
            if not episode_groups:
                # No non-noise windows anywhere in the leaf — nothing to split.
                continue
            groups.append(
                SubClusterGroup(
                    path=path,
                    episode_subclusters={
                        ep: sub_label for sub_label, eps in episode_groups.items() for ep in eps
                    },
                    window_subclusters=window_timeline if self.keep_window_detail else {},
                )
            )

        log.info(
            "subclustering: split %d/%d terminal leaves (>= %d episodes) into <= %d "
            "sub-clusters each",
            len(groups),
            n_total,
            self.min_episodes,
            self.k,
        )
        return SubClusterSet(groups=groups, params=self.params)
