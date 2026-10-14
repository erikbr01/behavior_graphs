"""Score cached labels on an arbitrary episode set, without re-running anything.

Every job's ``labels.npy`` covers all episodes of the dataset, because the
pipeline fits on the fit rows and then transforms everything. So re-scoring on a
different episode set -- the val half, the test half, the episodes a policy never
saw -- is a read plus a metric call, not a re-run. That is what makes the whole
held-out story affordable: 5,176 jobs in 72 minutes on CPU against ~20 hours to
recompute them.

The one thing worth caching hard is the windowing. A sweep spans only a couple
of dozen distinct ``(layer, width, stride)`` combinations, and each one costs a
feature extraction, so they are memoised per process.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable
from typing import Any

import numpy as np

from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.envs.trajectory_dir import RolloutDirectory
from behavior_graphs.stages.evaluation._boundary_metrics import (
    build_ground_truth,
    score_boundaries,
)
from behavior_graphs.stages.evaluation.subtask_boundaries import predicted_boundaries
from behavior_graphs.stages.windowing import SlidingWindow


def default_tolerance(cfg: DatasetConfig, gt: dict[int, Any], fit: Iterable[int]) -> float:
    """0.13 x the median ground-truth segment over the fit episodes.

    Derived from the fit half rather than the scored one: deriving it per score
    set makes every fold a different metric. Only the boundary metric uses it --
    the segmental metric is IoU-based, which is why a dataset with a different
    tempo needs no recalibration.

    Callers should generally read ``Scorer.tolerance``, which pins this to the
    train split once per instance. This is the underlying rule.
    """
    if cfg.evaluation.tolerance is not None:
        return float(cfg.evaluation.tolerance)
    dur = [b - a for i in fit if i in gt for a, b in gt[i]["segments"]]
    if not dur:
        raise SystemExit(f"{cfg.name}: cannot derive a tolerance from an empty fit set")
    return round(0.13 * float(np.median(dur)), 4)


class Scorer:
    """Scores cached labels for one dataset, memoising features and windows."""

    def __init__(self, cfg: DatasetConfig, gt_set: str | None = None) -> None:
        from behavior_graphs.ablation_study import splits

        self.cfg = cfg
        self.rollouts = RolloutDirectory(cfg.root)
        self.gt = build_ground_truth(cfg.root, gt_set=gt_set or cfg.evaluation.gt_set)
        #: One tolerance for every score this instance produces, fixed at the
        #: train split. Resolved here rather than per call: a caller that leaves
        #: it None -- which every report does, since a config may set it null to
        #: be derived -- would otherwise get a tolerance derived from whichever
        #: episodes it happened to be scoring, making the val and test columns
        #: of one table two different metrics.
        self.tolerance = default_tolerance(cfg, self.gt, splits.resolve(cfg)[splits.TRAIN])
        self._features: dict[str, Any] = {}
        self._windows: dict[tuple[str, int, int], Any] = {}

    # ------------------------------------------------------------------ features
    def _extractor(self, layer: str):
        """Everything but ``state`` is read from ``policy_embeddings/<layer>.npz``.

        That includes the frozen vision backbone. cuML and torch do not coexist
        in one environment here -- the sweep runs under a RAPIDS venv with no
        torch, and the backbone needs torch with no cuML -- so vision features
        are extracted once by ``extract_vision_features.py`` and land in the
        same directory as the policy layers. The sweep and this scorer then read
        an array either way, and neither needs to know which produced it.
        """
        if layer == "state":
            from behavior_graphs.stages.features.state import StateFeatures

            return StateFeatures()
        if layer.startswith("state_"):
            # A concat arm: proprioception joined to the named embedding, with
            # the same per-block l2 the sweep config uses. Rebuilt here rather
            # than read from a file, because nothing writes a concatenated
            # array to policy_embeddings -- the sweep composes it per job.
            from behavior_graphs.stages.features.concat import ConcatFeatures
            from behavior_graphs.stages.features.policy_embedding import PolicyEmbeddings
            from behavior_graphs.stages.features.state import StateFeatures

            return ConcatFeatures(
                [StateFeatures(), PolicyEmbeddings(layer=layer[len("state_") :])],
                block_norm="l2",
            )
        from behavior_graphs.stages.features.policy_embedding import PolicyEmbeddings

        return PolicyEmbeddings(layer=layer)

    def windows(self, layer: str, width: int, stride: int):
        key = (layer, int(width), int(stride))
        if key not in self._windows:
            if layer not in self._features:
                self._features[layer] = self._extractor(layer).extract(self.rollouts)
            self._windows[key] = SlidingWindow(width=width, stride=stride).transform(
                self._features[layer]
            )
        return self._windows[key]

    # -------------------------------------------------------------------- scoring
    def score(
        self,
        job_dir: str | pathlib.Path,
        layer: str,
        width: int,
        stride: int,
        episodes: Iterable[int],
        min_segment: float,
        tolerance: float | None = None,
    ) -> dict[str, Any]:
        """Full metric block for one job on one episode set."""
        eps = sorted(i for i in episodes if i in self.gt)
        if not eps:
            raise SystemExit("no annotated episodes in the requested set")
        lp = pathlib.Path(job_dir) / "labels.npy"
        if not lp.exists():
            # rescored.json names a job that has since been moved or deleted --
            # normally into a pre_clean_policy quarantine, because its features
            # were replaced. A bare FileNotFoundError here reads as a bug in the
            # report rather than as a stale artefact, which is what it is.
            raise SystemExit(
                f"{lp} is gone; {self.cfg.rescored} is stale. "
                f"Run `study.py rescore --dataset {self.cfg.name}`."
            )
        labels = np.load(lp)
        win = self.windows(layer, width, stride)
        if labels.shape[0] != win.num_windows:
            raise SystemExit(
                f"{job_dir}: {labels.shape[0]} labels against {win.num_windows} windows"
            )
        gt = {i: self.gt[i] for i in eps}
        preds = predicted_boundaries(labels, win, {i: g["fps"] for i, g in gt.items()})
        tol = self.tolerance if tolerance is None else float(tolerance)
        return score_boundaries(
            preds, gt, eps, tol, float(min_segment), self.cfg.evaluation.density_band
        )

    def score_cell(self, cell, episodes: Iterable[int], tolerance: float | None = None):
        """Convenience wrapper taking a :class:`~behavior_graphs.ablation_study.cells.Cell`."""
        return self.score(
            cell.dir,
            cell.layer,
            cell.width,
            cell.stride,
            episodes,
            float(cell.min_segment),
            tolerance,
        )
