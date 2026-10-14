"""Agreement between window-label changes and human subtask boundaries.

The read-out `wgo_bench` was built around, as an :class:`Evaluator` so it flows
through the normal pipeline: `run_pipeline` puts it in ``metrics.json``, which
`run_sweep` turns into ``sweep_index.csv``, ``best_per_k.csv`` and
``sensitivity_<metric>.csv`` without any per-study reporting code.

An episode's label sequence changes value exactly where the graph says the policy
moved from one behavior node to the next, and that is the boundary prediction.
The scoring itself lives in :mod:`._boundary_metrics` so this and the offline
re-scorer are provably the same measurement.

Reported per ground-truth set: ``task`` (non-task segments -- mistake, reset, adjust,
trim -- become ignore regions) is primary, ``all`` is the sensitivity check.
Within each: boundary precision/recall/F1, the count-matched uniform floor and
``gain``, macro F1, density and whether it is in a plausible band, segmental F1 at
four IoU thresholds, and frame-wise ARI/NMI against ground-truth subtask identity.

**Scores are in-sample unless ``cfg.split`` is set.** With a split, the reduction,
clustering and graph were fitted on the fit episodes and only the score episodes
are measured; ``scored_on`` records which.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import Evaluator
from behavior_graphs.core.types import WindowSet
from behavior_graphs.stages.evaluation._boundary_metrics import (
    build_ground_truth,
    label_agreement,
    score_boundaries,
)


def predicted_boundaries(
    labels: np.ndarray, windows: WindowSet, fps_of: dict[int, float]
) -> dict[int, np.ndarray]:
    """Per-episode times (SECONDS) where the label sequence changes value.

    ``WindowInfo.start``/``end`` are episode-local *frame* indices, so the
    conversion to seconds is mandatory -- ground-truth segments are in seconds, and
    comparing frames to seconds silently produces a near-zero F1 rather than an
    error (measured: 0.018 where the same labels score ~0.34).
    """
    per_ep: dict[int, list[tuple[float, int]]] = {}
    for i, w in enumerate(windows.windows):
        lab = int(labels[i])
        if lab == -1:  # noise, e.g. HDBSCAN
            continue
        per_ep.setdefault(w.episode_idx, []).append(((w.start + w.end - 1) / 2.0, lab))
    out: dict[int, np.ndarray] = {}
    for ep, seq in per_ep.items():
        seq.sort(key=lambda x: x[0])
        centres = np.asarray([c for c, _ in seq], dtype=np.float64)
        labs = np.asarray([lab for _, lab in seq])
        if labs.size < 2:
            out[ep] = np.empty(0)
            continue
        at = np.nonzero(labs[1:] != labs[:-1])[0]
        # A change between two windows localises the event no better than halfway
        # between their centres; the half frame centres the exposure.
        mid = (centres[at] + centres[at + 1]) / 2.0
        out[ep] = (mid + 0.5) / fps_of.get(ep, 10.0)
    return out


class SubtaskBoundaryEvaluator(Evaluator):
    """Boundary and subtask-identity agreement with human annotations.

    Args:
        tolerance: match window in seconds, **frozen** rather than derived from
            the episodes being scored. Deriving it per score set makes each fold a
            different metric -- on ABC bottles that is 0.5062s on cv4 against
            0.5410s on cv0 -- so a cross-fold mean would average over different
            measurements. 0.5273 is 0.13 x the median ground-truth segment over the train
            half of the 174-episode root.
        loose_multiple: a second, looser tolerance is always reported. Every arm
            of the CPU study cleared its uniform floor at the primary tolerance
            and fell *below* it at 2x, because at density>2 evenly spaced
            boundaries already land inside a wide window.
        min_segments: swept internally -- merging short segments is a post-hoc
            operation on a boundary list, so every value costs microseconds here
            where a grid axis would cost a job apiece.
        density_band: the regime in which a gain is meaningful at all. Raw F1 can
            be bought by emitting more boundaries; gain removes that but can be
            bought by emitting almost none, since a sparse predictor's uniform
            floor is weak.
        gt_sets: ``task`` first (primary).
    """

    name = "subtask_boundaries"

    def __init__(
        self,
        tolerance: float = 0.5273,
        loose_multiple: float = 2.0,
        min_segments: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0),
        density_band: tuple[float, float] = (0.8, 1.5),
        gt_sets: tuple[str, ...] = ("task", "all"),
    ) -> None:
        self.tolerance = float(tolerance)
        self.loose_multiple = float(loose_multiple)
        self.min_segments = tuple(float(m) for m in min_segments)
        self.density_band = (float(density_band[0]), float(density_band[1]))
        self.gt_sets = tuple(gt_sets)

    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        root = Path(result.metrics.get("rollout_dir", "."))
        split = result.metrics.get("split")

        out: dict[str, Any] = {
            "tolerance": self.tolerance,
            "loose_tolerance": round(self.tolerance * self.loose_multiple, 4),
            "min_segments": list(self.min_segments),
            "density_band": list(self.density_band),
            "scored_on": "held_out" if split else "in_sample",
            "split": (split or {}).get("label", "none"),
            "primary_gt_set": self.gt_sets[0],
        }

        for gs in self.gt_sets:
            gt = build_ground_truth(root, gt_set=gs)
            eps = list(result.windows.episodes)
            local_to_gt = {}
            for local, ep in enumerate(eps):
                gid = getattr(ep, "index", None)
                gid = local if gid is None else int(gid)
                if gid in gt:
                    local_to_gt[local] = gid
            if split and split.get("score") is not None:
                keep = {int(i) for i in split["score"]}
                local_to_gt = {k: v for k, v in local_to_gt.items() if k in keep}
            scored = sorted(local_to_gt)
            # Gold is keyed by annotation index; windows by position in the set
            # handed to the run. They coincide for a whole rollout directory,
            # which is what a split-based run always passes.
            gt_local = {local: gt[gid] for local, gid in local_to_gt.items()}

            preds = predicted_boundaries(
                np.asarray(result.labels),
                result.windows,
                {local: g["fps"] for local, g in gt_local.items()},
            )
            pref = "" if gs == self.gt_sets[0] else f"_{gs}"
            out[f"n_episodes_scored{pref}"] = len(scored)

            for ti, tol in enumerate(
                (self.tolerance, round(self.tolerance * self.loose_multiple, 4))
            ):
                for ms in self.min_segments:
                    s = score_boundaries(preds, gt_local, scored, tol, ms, self.density_band)
                    tag = ("" if ti == 0 else f"@{tol:g}") + ("" if ms == 0.0 else f"_ms{ms:g}")
                    for k, v in s.items():
                        if k in ("tolerance", "min_segment"):
                            continue
                        out[f"{k}{tag}{pref}"] = v

            best_ms = max(
                self.min_segments,
                key=lambda m: (
                    out.get(f"in_band{'' if m == 0.0 else f'_ms{m:g}'}{pref}", False),
                    out.get(f"gain{'' if m == 0.0 else f'_ms{m:g}'}{pref}", float("-inf")),
                ),
            )
            out[f"best_min_segment{pref}"] = best_ms
            tag = "" if best_ms == 0.0 else f"_ms{best_ms:g}"
            out[f"gain_best_ms{pref}"] = out[f"gain{tag}{pref}"]

            out[f"label_agreement{pref}"] = label_agreement(
                np.asarray(result.labels), result.windows, gt_local, scored
            )
        return out
