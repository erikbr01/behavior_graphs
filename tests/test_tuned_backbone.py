"""Hyperparameters are chosen on the train split, and eval cannot reach them.

The cross-dataset sweep fit a graph at each K, scored all of them on the
held-out folds, and reported the maximum -- selection on the test set. The
replacement puts the search inside ``GraphBackbone.fit``, which the runner only
ever hands the train view, so the guarantee is structural rather than a rule
someone has to remember. These tests hold it to that: perturbing eval must not
move the choice, and the inner validation episodes must come from train.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavior_graphs.progress_estimation.io import EpisodeSet, EpisodeView
from behavior_graphs.progress_estimation.methods import GraphStage, TunedGraphBackbone
from behavior_graphs.progress_estimation.metrics import MAE, VOC
from behavior_graphs.progress_estimation.runner import BenchmarkRunner, KFoldSplit

GRID = [{"n_clusters": 3}, {"n_clusters": 6}, {"n_clusters": 12}]


def _set(n=40, dim=5, seed=0, scale=1.0) -> EpisodeSet:
    """Monotone-drift episodes, so a graph has real stage structure to find."""
    rng = np.random.default_rng(seed)
    flags = [(i % 5) != 0 for i in range(n)]
    embs = []
    for _ in range(n):
        T = int(rng.integers(20, 40))
        t = np.linspace(0, 1, T)[:, None]
        embs.append(
            (scale * t * rng.normal(size=(1, dim)) + 0.1 * rng.normal(size=(T, dim))).astype(float)
        )
    ds = EpisodeSet("synth", "/nonexistent", flags)
    ds._embeddings = embs
    return ds


def _view(ds, idx) -> EpisodeView:
    return EpisodeView(ds, list(idx))


class TestSelectionUsesTrainOnly:
    def test_the_eval_split_cannot_change_the_choice(self):
        # The property the old design could not offer: rewrite every eval
        # episode and the selected hyperparameters must be identical, because
        # `fit` never saw them.
        ds = _set()
        train, ev = list(range(0, 28)), list(range(28, 40))

        bb = TunedGraphBackbone(GRID)
        bb.fit(_view(ds, train))
        first = bb.chosen

        wrecked = _set()
        rng = np.random.default_rng(7)
        for i in ev:  # noise where the eval episodes were
            wrecked._embeddings[i] = rng.normal(size=wrecked._embeddings[i].shape)
        bb2 = TunedGraphBackbone(GRID)
        bb2.fit(_view(wrecked, train))

        assert first == bb2.chosen
        assert first in GRID

    def test_validation_episodes_are_drawn_from_train(self):
        ds = _set()
        train = list(range(0, 30))
        bb = TunedGraphBackbone(GRID, val_frac=0.25)
        fit_view, val_view = bb._inner_split(_view(ds, train))

        assert set(fit_view.indices) | set(val_view.indices) == set(train)
        assert not set(fit_view.indices) & set(val_view.indices)
        assert len(val_view) > 0 and len(fit_view) > 0

    def test_every_candidate_is_scored_and_the_winner_recorded(self):
        ds = _set()
        bb = TunedGraphBackbone(GRID)
        bb.fit(_view(ds, range(30)))
        (entry,) = bb.history
        assert len(entry["scores"]) == len(GRID)
        assert entry["chosen"] in GRID
        assert entry["n_fit"] + entry["n_val"] == 30

    def test_the_winner_is_refit_on_the_whole_train_split(self):
        # The validation split exists to choose, not to shrink what the
        # reported model was fit on.
        ds = _set()
        bb = TunedGraphBackbone([{"n_clusters": 4}])
        train = _view(ds, range(30))
        bb.fit(train)
        assert bb.fitted
        # Windows over all 30 train episodes, not the ~22 the inner fit saw.
        assert len(bb.window_lengths(train)) == 30


class TestDirectionAndDegenerateCases:
    def test_a_lower_is_better_metric_is_minimised(self):
        ds = _set()
        picked = {}
        for metric in (VOC(), MAE()):
            bb = TunedGraphBackbone(GRID, metric=metric)
            bb.fit(_view(ds, range(30)))
            scored = {k: v for k, v in bb.history[0]["scores"].items() if v == v}
            want = (max if metric.higher_better else min)(scored, key=scored.get)
            got = repr(sorted(bb.history[0]["chosen"].items()))
            picked[metric.name] = (got, want)
        for name, (got, want) in picked.items():
            assert got == want, name

    def test_a_train_split_too_small_to_hold_any_back_says_so(self):
        # val_frac rounds down to zero episodes here, so there is nothing to
        # select on. Choosing arbitrarily and recording that is honest; choosing
        # on an empty validation set would not be.
        ds = _set(n=10)
        bb = TunedGraphBackbone(GRID, val_frac=0.01)
        bb.fit(_view(ds, [1, 2]))  # two successful episodes, nothing to hold back
        assert bb.history[0].get("reason") == "train too small"
        assert bb.chosen == GRID[0]

    def test_an_untunable_key_is_rejected_rather_than_ignored(self):
        # Silently dropping it would make the whole search a no-op.
        with pytest.raises(ValueError, match="untunable"):
            TunedGraphBackbone([{"n_clusters": 8, "reduction": "umap"}])
        with pytest.raises(ValueError, match="at least one candidate"):
            TunedGraphBackbone([])


def test_the_runner_reselects_per_outer_fold():
    # K is re-chosen inside each fold, so `history` grows with the folds and the
    # reported spread includes the variance of the choice itself.
    ds = _set(n=45)
    bb = TunedGraphBackbone(GRID)
    summary = BenchmarkRunner([VOC()], KFoldSplit(n_folds=3, seed=0)).run(
        ds, [GraphStage(backbone=bb)]
    )
    assert len(bb.history) == 3
    # The reported mean must be a real number, not NaN: every outer fold has
    # to have produced a score for the spread to mean anything. (This line
    # used to compare a value to itself, which no arithmetic can fail.)
    assert np.isfinite(summary["graph_stage"]["voc"][0])
