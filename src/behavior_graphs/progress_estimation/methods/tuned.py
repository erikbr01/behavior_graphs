"""Choose the graph's hyperparameters on a split of the TRAIN set, never on eval.

``sweep_progress_allds.py`` fit a graph at each candidate K, scored every one of
them on the held-out folds, and reported the best -- so the number it published
was a maximum over K taken on the same data it was claiming to be held out on.
With five candidates that is a real optimistic bias, and no inner split existed
to absorb it.

The fix is structural rather than procedural. :class:`TunedGraphBackbone` is a
:class:`~...methods.graph.GraphBackbone` whose ``fit`` carves a validation split
out of the train view it was handed, scores each candidate on that, and only
then refits the winner on the whole train view. ``BenchmarkRunner`` passes
``fit`` the train view and nothing else, so eval is not merely unused here --
it is unreachable. Every method sharing the backbone (stage, absorb,
frame-index) inherits the selection without knowing it happened.

The cost is honesty about variance: K is re-selected inside every outer fold, so
different folds may disagree, and :attr:`TunedGraphBackbone.history` records what
each one chose. A single K reported across folds was always a summary of a
choice, not a fitted parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from behavior_graphs.progress_estimation.io import EpisodeView
from behavior_graphs.progress_estimation.methods.graph import GraphBackbone
from behavior_graphs.progress_estimation.metrics.base import ProgressMetric
from behavior_graphs.stages import KMeansClustering

#: What a candidate may set. Anything else in a grid entry is a typo, and a
#: silently ignored hyperparameter would make the whole search a no-op.
TUNABLE = frozenset({"n_clusters", "window_width", "window_stride"})


class TunedGraphBackbone(GraphBackbone):
    """A graph backbone that picks its own hyperparameters on inner validation.

    ``grid`` is a sequence of partial parameter maps, e.g.
    ``[{"n_clusters": 8}, {"n_clusters": 16}, {"n_clusters": 32}]``; unset keys
    fall back to this instance's own values, so the grid says only what varies.

    ``val_frac`` of the train view (stratified on the success flag) becomes the
    inner validation set. ``metric`` scores candidates -- VOC of the stage
    method by default, the number the sweep reports -- and its ``higher_better``
    decides the direction, so a loss-like criterion is not maximised by mistake.
    """

    def __init__(
        self,
        grid: Sequence[Mapping[str, Any]],
        *,
        metric: ProgressMetric | None = None,
        val_frac: float = 0.25,
        val_seed: int = 0,
        seed: int = 0,
        **backbone_kwargs: Any,
    ):
        super().__init__(**backbone_kwargs)
        if not grid:
            raise ValueError("TunedGraphBackbone needs at least one candidate in `grid`")
        unknown = {k for cand in grid for k in cand} - TUNABLE
        if unknown:
            raise ValueError(
                f"grid sets untunable keys {sorted(unknown)}; tunable: {sorted(TUNABLE)}"
            )
        self.grid = [dict(cand) for cand in grid]
        self.val_frac = val_frac
        self.val_seed = val_seed
        self.seed = seed
        if metric is None:
            from behavior_graphs.progress_estimation.metrics import VOC

            metric = VOC()
        self.metric = metric
        #: One entry per fold: the winning params and every candidate's score.
        self.history: list[dict[str, Any]] = []
        self.chosen: dict[str, Any] | None = None

    # ------------------------------------------------------------------ inner
    def _candidate(self, params: Mapping[str, Any]) -> GraphBackbone:
        """A plain backbone at *params*, defaulting to this instance's settings."""
        n_clusters = params.get("n_clusters", getattr(self.clustering, "n_clusters", 16))
        return GraphBackbone(
            window_width=params.get("window_width", self.window_width),
            window_stride=params.get("window_stride", self.window_stride),
            reduction=self.reduction,
            clustering=KMeansClustering(n_clusters=n_clusters, random_state=self.seed),
        )

    def _inner_split(self, train: EpisodeView) -> tuple[EpisodeView, EpisodeView]:
        """(fit, validation) views over *train* only, stratified on success."""
        from behavior_graphs.progress_estimation.runner import KFoldSplit

        # n_folds=1 is a single stratified holdout, which is what an inner
        # validation split is; reused rather than reimplemented so the two
        # cannot drift in how they stratify.
        ((fit_pos, val_pos),) = KFoldSplit(
            n_folds=1, test_frac=self.val_frac, seed=self.val_seed
        ).folds(train.flags)
        parent = train.parent
        to_parent = [train.indices[i] for i in fit_pos], [train.indices[i] for i in val_pos]
        return EpisodeView(parent, to_parent[0]), EpisodeView(parent, to_parent[1])

    def _score(self, backbone: GraphBackbone, fit_view, val_view) -> float:
        from behavior_graphs.progress_estimation.methods.graph import GraphStage

        backbone.fit(fit_view)
        method = GraphStage(backbone=backbone)
        return float(self.metric.compute(method.predict(val_view), val_view.flags))

    # -------------------------------------------------------------------- fit
    def fit(self, train) -> TunedGraphBackbone:
        if self.fitted:
            return self
        fit_view, val_view = self._inner_split(train)
        if not len(fit_view) or not len(val_view):
            # Too few episodes to hold any back. Selecting on nothing is worse
            # than not selecting, so take the first candidate and say so.
            self.chosen = dict(self.grid[0])
            self.history.append({"chosen": self.chosen, "scores": {}, "reason": "train too small"})
        else:
            scores: dict[str, float] = {}
            best_params, best_score = None, None
            for params in self.grid:
                score = self._score(self._candidate(params), fit_view, val_view)
                scores[repr(sorted(params.items()))] = score
                if score != score:  # NaN: this candidate produced no number
                    continue
                better = best_score is None or (
                    score > best_score if self.metric.higher_better else score < best_score
                )
                if better:
                    best_params, best_score = dict(params), score
            # Every candidate NaN means the criterion could not be evaluated at
            # all; falling back to the first is arbitrary but is recorded.
            self.chosen = best_params if best_params is not None else dict(self.grid[0])
            self.history.append(
                {
                    "chosen": self.chosen,
                    "score": best_score,
                    "scores": scores,
                    "n_fit": len(fit_view),
                    "n_val": len(val_view),
                }
            )

        # Refit the winner on the WHOLE train view: the validation split existed
        # to choose, not to shrink the training set the reported model saw.
        winner = self._candidate(self.chosen)
        self.window_width = winner.window_width
        self.window_stride = winner.window_stride
        self.clustering = winner.clustering
        return super().fit(train)


__all__ = ["TunedGraphBackbone", "TUNABLE"]
