"""Pipeline evaluator for graph-derived progress labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import Evaluator
from behavior_graphs.progress_estimation.estimation import (
    episode_label_sequences,
    monotonicity,
    stage_progress,
    stage_schedule,
    success_absorption,
    success_auroc,
)


class ProgressEvaluator(Evaluator):
    """Quality metrics of the progress labels a fitted graph induces.

    Reports, per pipeline run:

    - ``absorption_*``: distribution of P(absorb at SUCCESS | node) over the
      cluster nodes — a collapsed absorption range means the graph cannot
      tell doomed states from success-bound ones.
    - ``n_stages`` / ``stage_order``: the discovered stage schedule
      (clusters ordered by mean normalized time on successful episodes).
    - ``stage_progress_monotonicity``: mean Spearman rho of stage progress
      vs window index over successful episodes.
    - ``tail_progress_auroc``: AUROC of the final-window stage progress
      against the episode success flag — can the labels separate successes
      from failures at the end of the episode?

    Stage metrics are ``None`` when no episode is marked successful (the
    schedule needs at least one success); AUROC is ``None`` when outcomes
    are one-sided or unknown.
    """

    name = "progress"

    def evaluate(self, result: PipelineResult) -> dict[str, Any]:
        labels = np.asarray(result.labels)
        windows = result.windows
        label_seqs = episode_label_sequences(labels, windows)
        successes = [ep.success is True for ep in windows.episodes]
        known = [ep.success is not None for ep in windows.episodes]

        out: dict[str, Any] = {
            "n_episodes": len(windows.episodes),
            "n_success": int(sum(successes)),
        }

        if result.graph is not None:
            absorb = success_absorption(result.graph)
            values = list(absorb.values())
            out["absorption_n_nodes"] = len(values)
            out["absorption_mean"] = float(np.mean(values)) if values else None
            out["absorption_min"] = float(np.min(values)) if values else None
            out["absorption_max"] = float(np.max(values)) if values else None
        else:
            out["absorption_n_nodes"] = None
            out["absorption_mean"] = None
            out["absorption_min"] = None
            out["absorption_max"] = None

        try:
            schedule = stage_schedule(label_seqs, successes)
        except ValueError:
            out.update(
                n_stages=None,
                stage_order=None,
                stage_progress_monotonicity=None,
                tail_progress_auroc=None,
            )
            return out

        out["n_stages"] = len(schedule.order)
        out["stage_order"] = [int(c) for c in schedule.order]

        progress_seqs = [stage_progress(seq, schedule) for seq in label_seqs]
        out["stage_progress_monotonicity"] = _none_if_nan(monotonicity(progress_seqs, successes))

        tails = np.array([p[-1] if len(p) else 0.0 for p in progress_seqs])
        mask = np.asarray(known)
        out["tail_progress_auroc"] = _none_if_nan(
            success_auroc(tails[mask], np.asarray(successes)[mask])
        )
        return out


def _none_if_nan(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)
