"""Abstract base classes for every stage of the behavior-graph pipeline.

The pipeline is::

    TrajectorySet (rollouts or featurized demos)
      -> FeatureExtractor  -> FeatureSet      (per-timestep features)
      -> Windowing         -> WindowSet       (aggregated windows)
      -> DimReduction      -> np.ndarray      (reduced window features)
      -> Clustering        -> np.ndarray      (cluster label per window)
      -> GraphConstruction -> BehaviorGraph
      -> SubClustering     -> SubClusterSet   (optional)
      -> Evaluator(s)      -> metrics dict
      -> NodeLabeler       -> node_labels.json (optional)

Every concrete implementation is a plain class whose constructor arguments
are its hyperparameters; stages are instantiated from hydra configs via
``_target_``, which makes every hyperparameter sweepable with ``hydra -m``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from behavior_graphs.core.types import FeatureSet, WindowSet

if TYPE_CHECKING:
    from pathlib import Path

    from behavior_graphs.core.env import TrajectorySet
    from behavior_graphs.core.graph import BehaviorGraph
    from behavior_graphs.core.result import PipelineResult
    from behavior_graphs.core.subclusters import SubClusterSet

#: Hyperparameter-bearing config groups, in pipeline order. The single
#: source of truth for everything that iterates "all stages": the sweep
#: runner (upstream-first job ordering), the runs index (column layout),
#: and the GUI pipeline builder.
#:
#: Deliberately excludes the optional, nullable stages (``labeling``,
#: ``subclustering``): it is off by default, so sweeping
#: them would multiply every job, and they add no column to the runs index.
STAGE_GROUPS = ("features", "windowing", "reduction", "clustering", "graph")


class FeatureExtractor(ABC):
    """Turns a trajectory set (rollouts or demos) into per-timestep features."""

    @abstractmethod
    def extract(self, trajectories: TrajectorySet) -> FeatureSet:
        """Return one feature vector per trajectory timestep."""


class Windowing(ABC):
    """Aggregates per-timestep features into per-window features.

    Implementations must not create windows that cross episode boundaries.
    """

    @abstractmethod
    def transform(self, features: FeatureSet) -> WindowSet:
        """Aggregate features into windows."""


class DimReduction(ABC):
    """Reduces window features to a lower-dimensional space.

    Follows the sklearn fit/transform contract so that a reducer fit on one
    set of windows (e.g. demos) can be applied to another (e.g. rollouts).
    """

    def set_window_context(self, windows: WindowSet | None) -> None:
        """Offer the window set the features came from, before :meth:`fit`.

        No-op by default. ``fit`` deliberately takes features alone, so that a
        reducer can be fit on one window set and applied to another — but a
        reducer whose objective involves *which windows are adjacent in time*
        cannot recover that from the feature matrix. Overriding this is how such a
        reducer asks for it, without every other one having to accept an argument
        it ignores.
        """

    @abstractmethod
    def fit(self, features: np.ndarray) -> DimReduction:
        """Fit the reducer; returns self."""

    @abstractmethod
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Map (n, d) features to (n, n_components)."""

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)


class Clustering(ABC):
    """Assigns a cluster label to every window.

    Labels are non-negative integers; ``-1`` marks noise (e.g. HDBSCAN).
    """

    def set_window_context(self, windows: WindowSet | None) -> None:
        """Offer the window set the features came from, before :meth:`fit`.

        No-op by default, and for the same reason as
        :meth:`DimReduction.set_window_context`: a point clusterer needs only the
        features, but a sequence model needs to know which windows form which
        episode and in what order. Without it, an HMM learns a transition across
        every episode boundary that no trajectory ever made.
        """

    @abstractmethod
    def fit(self, features: np.ndarray) -> Clustering:
        """Fit the clusterer; returns self."""

    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return (n,) int cluster labels for new data."""

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).predict(features)


class GraphConstruction(ABC):
    """Builds a behavior graph from per-window cluster labels.

    A builder that relabels windows during :meth:`build` (smoothing,
    degree-one pruning, state merging) may optionally expose a
    ``relabel(labels, windows) -> np.ndarray`` method mapping new (e.g.
    held-out) raw cluster labels into the same node-id space -- see
    ``TransitionGraphBuilder.relabel``/``StateMergingGraphBuilder.relabel``
    and :func:`behavior_graphs.outcome_classifier.classify.predict`, its only
    caller. Not part of the formal interface (like ``last_labels_``, below):
    most builders don't relabel at all, so callers duck-type with
    ``getattr(builder, "relabel", None)`` rather than every implementation
    carrying a trivial identity override.
    """

    @abstractmethod
    def build(self, labels: np.ndarray, windows: WindowSet) -> BehaviorGraph:
        """Construct the graph; ``labels[i]`` corresponds to ``windows.windows[i]``."""


class SubClustering(ABC):
    """Computes finer groupings for trajectory-tree leaves.

    This optional stage (``cfg.subclustering``, off by default) produces an
    overlay rather than modifying the behavior graph. It runs after graph
    construction only because smoothing/pruning may replace the window
    labels used to define trajectory-tree paths.

    The result is persisted as ``subclusters.json`` and is what the demo app
    serves when a user expands a terminal leaf; without the stage the app
    computes the same split on demand (see
    :func:`behavior_graphs.core.subclusters.subcluster_episodes`).
    """

    name: str = "subclustering"

    @abstractmethod
    def split(self, result: PipelineResult) -> SubClusterSet:
        """Sub-cluster selected trajectory leaves into a SubClusterSet.

        Pure: reads ``result`` and returns the overlay without mutating the
        behavior graph. The expensive local reducer fits can therefore be
        served directly from the stage cache.
        """

    def cache_fingerprint(self) -> dict[str, Any]:
        """Everything determining this stage's output, for the stage cache.

        Must include any config the stage reads *by name* at compute time
        (which the top-level pipeline config only references), or a cache hit
        could differ from a fresh compute. Defaults to nothing — override
        when the stage has parameters.
        """
        return {}

    def summarize(self, subclusters: SubClusterSet) -> dict[str, Any]:
        """Small JSON-serializable summary of a split, for ``metrics.json``.

        The split itself lives in ``subclusters.json``; this is what shows up
        in the runs index and sweep tables. Implementations may extend it.
        """
        per_node = [len(g.episode_groups()) for g in subclusters.groups]
        return {
            "num_nodes": len(subclusters.groups),
            "num_episodes": sum(g.num_episodes for g in subclusters.groups),
            "mean_subclusters_per_node": (
                round(sum(per_node) / len(per_node), 3) if per_node else 0.0
            ),
            "params": subclusters.params,
        }


class Evaluator(ABC):
    """Computes metrics over the (partial or full) pipeline result.

    Evaluators are composable: the evaluation stage runs a list of them and
    merges their outputs under ``metrics[evaluator.name]``.
    """

    name: str = "evaluator"

    @abstractmethod
    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        """Return a JSON-serializable metrics dict."""


class NodeLabeler(ABC):
    """Assigns human-readable labels to behavior-graph nodes.

    Runs as its own optional pipeline step after evaluation (``cfg.labeling``).
    Unlike evaluators, a labeler needs filesystem context — it reads episode
    videos from *rollout_dir* and persists ``node_labels.json`` to *run_dir* —
    so both are passed explicitly.

    The labels themselves live only in ``node_labels.json`` (loaded via
    :mod:`behavior_graphs.core.node_labels`), which stays editable after the
    run; the return value is a small JSON-serializable summary (e.g. model,
    node count) that the pipeline merges into the metrics dict.
    """

    name: str = "labeler"

    @abstractmethod
    def label(
        self,
        result: PipelineResult,
        run_dir: Path | None = None,
        rollout_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Label nodes, persist them to *run_dir*, and return a summary dict."""
