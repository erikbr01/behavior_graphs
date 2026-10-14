"""Graph-derived progress estimation and demo-curation scoring.

Ported from the ``online_rl`` ``graph_guided`` progress library onto
behavior_graphs types: SARM-style stage schedules and progress labels
(:mod:`~behavior_graphs.progress_estimation.estimation`), demo-scoring schemes for
weighted BC (:mod:`~behavior_graphs.progress_estimation.scoring`), and a pipeline
evaluator (:mod:`~behavior_graphs.progress_estimation.evaluator`).
"""

from behavior_graphs.progress_estimation.estimation import (
    StageSchedule,
    absorption_progress,
    chunk_deltas,
    episode_label_sequences,
    frame_index_progress,
    monotonicity,
    per_episode_progress_metrics,
    progress_regression_metrics,
    rabc_weights,
    same_state_label_std,
    sarm_soft_clip,
    stage_progress,
    stage_schedule,
    success_absorption,
    success_auroc,
)
from behavior_graphs.progress_estimation.evaluator import ProgressEvaluator
from behavior_graphs.progress_estimation.scoring import (
    dwell_fraction_stats,
    ess,
    expand_to_steps,
    length_scores,
    oracle_scores,
    ramp_scores,
    relu_scores,
    seg_trim_scores,
    shuffled_scores,
    stage_balance_scores,
    zero_frac,
)

__all__ = [
    "StageSchedule",
    "stage_schedule",
    "stage_progress",
    "absorption_progress",
    "frame_index_progress",
    "chunk_deltas",
    "rabc_weights",
    "sarm_soft_clip",
    "same_state_label_std",
    "success_auroc",
    "monotonicity",
    "progress_regression_metrics",
    "per_episode_progress_metrics",
    "success_absorption",
    "episode_label_sequences",
    "ramp_scores",
    "relu_scores",
    "length_scores",
    "dwell_fraction_stats",
    "seg_trim_scores",
    "stage_balance_scores",
    "oracle_scores",
    "shuffled_scores",
    "expand_to_steps",
    "ess",
    "zero_frac",
    "ProgressEvaluator",
]
