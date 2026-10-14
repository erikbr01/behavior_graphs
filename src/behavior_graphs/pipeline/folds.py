"""Fit/score episode splits for a pipeline run.

`run_pipeline` fits every stage on the episodes it is given and scores the same
ones, which is what you want for describing a dataset and wrong for measuring
whether a configuration generalises. The stage contracts already support the
split -- :class:`~behavior_graphs.core.stages.DimReduction` says so outright,
*"so that a reducer fit on one set of windows (e.g. demos) can be applied to
another (e.g. rollouts)"* -- so all that was missing was a way to say which
episodes are which, and for the cache key to know.

A split is opt-in via ``cfg.split``. Two forms:

    split:
      file: /path/to/splits.json
      fold: cv0            # nth entry of cv_folds -> its fit/score lists
    split:
      file: /path/to/splits.json
      fold: val            # fit on `train`, score on `val`

or explicit, for callers that already have the indices::

    split:
      fit: [0, 3, 7]
      score: [1, 2]

Indices are positions in the rollout directory's episode ordering, which is what
``splits.json`` files written by `scripts/wgo_bench` and `scripts/ablation_study`
contain.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.types import WindowSet


class Split:
    """Which episodes a run fits on, and which it is scored on."""

    __slots__ = ("fit", "score", "label")

    def __init__(self, fit: list[int], score: list[int], label: str) -> None:
        self.fit = sorted(int(i) for i in fit)
        self.score = sorted(int(i) for i in score)
        self.label = label
        overlap = set(self.fit) & set(self.score)
        if overlap:
            raise ValueError(
                f"split {label!r} has {len(overlap)} episode(s) in both fit and score "
                f"(e.g. {sorted(overlap)[:5]}); that leaks the held-out half"
            )

    def identity(self) -> dict[str, Any]:
        """What the cache key must include: the fit set, not the score set.

        A reduction fitted on set A is the same artifact regardless of which
        episodes are later transformed through it, so keying on ``fit`` alone
        keeps two folds that share a fit half sharing the cache entry -- while
        never serving a fold's artifact to a fold that fitted on different data.
        """
        return {"fit": self.fit, "label": self.label}

    def __repr__(self) -> str:
        return f"Split({self.label!r}, {len(self.fit)} fit, {len(self.score)} score)"


def resolve_split(node: Any, n_episodes: int) -> Split | None:
    """``cfg.split`` -> a :class:`Split`, or None when unset."""
    if node is None:
        return None
    spec = {k: v for k, v in dict(node).items() if v is not None}
    # The root config declares the keys so they are overridable, so "no split"
    # arrives as a mapping of Nones rather than as None itself.
    if not spec:
        return None
    if "fit" in spec and "score" in spec:
        return Split(list(spec["fit"]), list(spec["score"]), str(spec.get("label", "explicit")))

    path = spec.get("file")
    if not path:
        raise ValueError("split needs either explicit fit/score lists or a `file`")
    doc = json.loads(Path(path).read_text())
    fold = str(spec.get("fold", "val"))

    if fold in ("val", "inner"):
        fit_key, score_key = ("train", "val") if fold == "val" else ("train_inner", "test_inner")
        missing = [k for k in (fit_key, score_key) if k not in doc]
        if missing:
            raise ValueError(f"{path} has no {missing} for fold={fold!r} (keys: {sorted(doc)})")
        return Split(doc[fit_key], doc[score_key], fold)

    if not fold.startswith("cv"):
        raise ValueError(f"fold must be 'val', 'inner' or 'cv<N>'; got {fold!r}")
    idx = int(fold[2:])
    folds = doc.get("cv_folds") or []
    if idx >= len(folds):
        raise ValueError(f"{path} has {len(folds)} cv_folds; fold={fold!r} is out of range")
    return Split(folds[idx]["fit"], folds[idx]["score"], fold)


def window_mask(windows: WindowSet, episodes: list[int]) -> np.ndarray:
    """Boolean mask over ``windows`` selecting those from ``episodes``.

    ``WindowInfo.episode_idx`` is a position within the set handed to the run, so
    this assumes the run saw the whole rollout directory -- which is exactly why
    the split selects *rows* rather than being pushed down into the data loader:
    features and windowing stay identical across folds and keep sharing their
    cache entries, and only the reduction and clustering see a subset.
    """
    want = set(int(i) for i in episodes)
    return np.asarray([w.episode_idx in want for w in windows.windows], dtype=bool)


def subset_windows(windows: WindowSet, mask: np.ndarray) -> WindowSet:
    """``windows`` restricted to ``mask``, for a stage that needs the context.

    Sequence models read which windows form which episode and in what order (see
    ``set_window_context``), so handing them the full set while fitting on a
    subset would let them learn transitions across a boundary no trajectory
    made.
    """
    from dataclasses import replace

    kept = [w for w, keep in zip(windows.windows, mask) if keep]
    return replace(windows, features=windows.features[mask], windows=kept)
