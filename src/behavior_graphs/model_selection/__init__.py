"""Model selection: run indexing, scoring, and sensitivity analysis.

This package turns a population of pipeline run directories (e.g. one
sweep) into answers:

- :mod:`~behavior_graphs.model_selection.index` — flatten every run's config +
  metrics into one DataFrame row.
- :mod:`~behavior_graphs.model_selection.scoring` — metric registry (directions),
  population-normalized linear-combination scoring with hard constraints,
  Pareto fronts, and best-hyperparameters-per-K.
- :mod:`~behavior_graphs.model_selection.sensitivity` — per-hyperparameter main
  effects and matched-pair sensitivity of any metric.
"""

from behavior_graphs.model_selection.index import flatten_config, load_runs_index
from behavior_graphs.model_selection.scoring import (
    METRICS,
    CompositeSelection,
    MetricSpec,
    Objective,
    best_per_k,
    composite_scores,
    normalize_metric_columns,
    pareto_mask,
    score_metrics,
)
from behavior_graphs.model_selection.sensitivity import sensitivity_table

__all__ = [
    "METRICS",
    "CompositeSelection",
    "MetricSpec",
    "Objective",
    "best_per_k",
    "composite_scores",
    "flatten_config",
    "load_runs_index",
    "normalize_metric_columns",
    "pareto_mask",
    "score_metrics",
    "sensitivity_table",
]
