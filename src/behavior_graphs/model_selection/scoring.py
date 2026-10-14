"""Metric registry and model selection over a population of pipeline runs.

Two layers:

- :func:`score_metrics` scores ONE run's ``metrics.json`` dict with a raw
  linear combination (no population normalization possible) — used by the
  pipeline's ``objective`` config so hydra/optuna sweepers can optimize it.
- :func:`composite_scores` scores a DataFrame of runs (one row per run, see
  :mod:`~behavior_graphs.model_selection.index`) with a *population-normalized*
  linear combination: each metric is min-max scaled over the population and
  direction-corrected so 1.0 is always best, then combined with the given
  weights. This makes weights comparable across metrics with wildly
  different scales (silhouette is in [-1, 1], Calinski-Harabasz is
  unbounded).

Hard constraints express "necessary but not sufficient" criteria — e.g. the
silhouette score is a useful filter for good graphs but maximizing it alone
is not enough, so a typical selection is: constrain silhouette above a
floor, then rank by a combination of the remaining criteria.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricSpec:
    """One proxy metric: where it lives in metrics.json and which way is up."""

    key: str  # column name in the runs index
    label: str  # human-readable name
    path: str  # dotted path into metrics.json
    direction: str | None  # "higher" | "lower" | None (diagnostic only)
    fmt: str = ".4f"


#: Every proxy metric the pipeline emits, in display order. ``direction``
#: None means diagnostic-only: shown, plottable, but never given a weight
#: by default and excluded from "best" picks unless the user opts in.
METRICS: dict[str, MetricSpec] = {
    m.key: m
    for m in [
        MetricSpec("silhouette", "Silhouette", "cluster_quality.silhouette", "higher"),
        MetricSpec(
            "calinski_harabasz",
            "Calinski-Harabasz",
            "cluster_quality.calinski_harabasz",
            "higher",
            ".1f",
        ),
        MetricSpec("davies_bouldin", "Davies-Bouldin", "cluster_quality.davies_bouldin", "lower"),
        MetricSpec("swap_rate", "Swap rate / frame", "label_stats.swap_rate_per_frame", "lower"),
        MetricSpec(
            "mean_segment_length", "Segment length", "label_stats.mean_segment_length", None, ".2f"
        ),
        MetricSpec(
            "distinct_per_episode",
            "Distinct/episode",
            "label_stats.distinct_per_episode",
            None,
            ".2f",
        ),
        MetricSpec(
            "noise_fraction", "Noise fraction", "cluster_quality.noise_fraction", None, ".3f"
        ),
        MetricSpec(
            "mean_transition_entropy",
            "Transition entropy",
            "graph_stats.mean_transition_entropy",
            None,
            ".3f",
        ),
        MetricSpec("n_clusters", "Clusters found", "cluster_quality.n_clusters", None, "d"),
        # How big the *tree* view will be, which is the complaint a viewer
        # actually has ("this graph is insanely large") and which nothing
        # recorded. The trajectory tree is a trie over run-length-collapsed
        # label sequences, so its size is set by how many distinct paths the
        # episodes take -- not by K. Measured on 14 chess cells spanning
        # 750 to 22,868 tree nodes, rank correlation against node count:
        #
        #     swap_rate_per_window   +0.96     mean_segment_length  -0.96
        #     distinct_per_episode   +0.86     num_edges            +0.70
        #     K                      +0.58
        #
        # So K -- the dial a viewer reaches for -- is the *worst* predictor of
        # the thing they are reacting to, and the swap rate is very nearly a
        # perfect one. `lower` because a viewer hunting a legible graph wants
        # the small end; it is a size proxy, not a quality claim.
        MetricSpec(
            "swap_rate_per_window",
            "Swap rate / window (≈ tree size)",
            "label_stats.swap_rate_per_window",
            "lower",
            ".4f",
        ),
        MetricSpec("num_edges", "Graph edges", "graph_stats.num_edges", None, "d"),
        # The tree the app actually draws, and how much of their path the
        # episodes hold in common. `tree_nodes` is exact, not a proxy -- it is
        # verified equal to `build_trajectory_tree` + `tree_size` on chess
        # (24,875) and whiteboard (437, 1,225).
        #
        # `prefix_sharing` is the one measure so far that separates a labeling
        # from a random relabeling of the same segmentation: the `relabel`
        # control in probes/metric_validity.py scores F1@0.75 = 0.941 while
        # carrying no label information at all, and sharing is what notices.
        # Direction is None rather than "higher" on purpose -- the annotations
        # themselves run 0.32 (legos) to 0.82 (whiteboard), so more is not
        # better, *closer to this task's ground truth* is, and a single
        # direction cannot express that.
        MetricSpec("tree_nodes", "Tree nodes", "label_stats.tree_nodes", "lower", "d"),
        MetricSpec("prefix_sharing", "Path sharing", "label_stats.prefix_sharing", None, ".3f"),
        # Subtask-boundary read-out. `boundary_gain` is the one to select on:
        # raw F1 here is dominated by how many boundaries a config emits, so a
        # density-2.35 config can beat a density-1.07 config on F1 while scoring
        # 10x less over its own count-matched uniform floor.
        MetricSpec("boundary_gain", "Boundary gain", "subtask_boundaries.gain", "higher", ".4f"),
        MetricSpec("boundary_f1", "Boundary F1", "subtask_boundaries.f1", "higher", ".4f"),
        MetricSpec(
            "boundary_density", "Boundary density", "subtask_boundaries.density", None, ".2f"
        ),
    ]
}

#: A sensible starting point for model selection: silhouette as the primary
#: criterion plus outcome-MI and temporal coherence, with a silhouette floor
#: encoding "necessary but not sufficient".
DEFAULT_WEIGHTS: dict[str, float] = {
    "silhouette": 1.0,
    "davies_bouldin": 0.25,
    "swap_rate": 0.25,
}
DEFAULT_SILHOUETTE_FLOOR = 0.05


def metric_value(metrics: Mapping[str, Any], dotted: str) -> float | None:
    """Walk a dotted path into a (nested) metrics dict; None when absent."""
    node: Any = metrics
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)


def score_metrics(metrics: Mapping[str, Any], objective: Mapping[str, Any]) -> dict[str, Any]:
    """Raw linear combination of one run's metrics (no normalization).

    ``objective`` is ``{"weights": {<dotted path>: w, ...},
    "constraints": {<dotted path>: {"min": x, "max": y}, ...}}``. A missing
    metric contributes nothing but is reported. Constraints only *flag* the
    run (``feasible`` / ``violated``) — the run is already computed, so it
    is never discarded and the score stays the plain weighted sum.
    """
    weights: Mapping[str, float] = objective.get("weights") or {}
    constraints: Mapping[str, Mapping[str, float]] = objective.get("constraints") or {}

    terms: dict[str, float | None] = {}
    score = 0.0
    for path, weight in weights.items():
        value = metric_value(metrics, path)
        terms[path] = value
        if value is not None:
            score += float(weight) * value

    feasible = True
    violated: list[str] = []
    for path, bounds in constraints.items():
        value = metric_value(metrics, path)
        lo = bounds.get("min")
        hi = bounds.get("max")
        if value is None or (lo is not None and value < lo) or (hi is not None and value > hi):
            feasible = False
            violated.append(path)

    return {
        "score": score,
        "feasible": feasible,
        "violated": violated,
        "terms": terms,
        "weights": dict(weights),
    }


def normalize_metric_columns(df: pd.DataFrame, metric_keys: Iterable[str]) -> pd.DataFrame:
    """Min-max scale each metric over the population, direction-corrected.

    Returns a DataFrame of the same index with one column per metric in
    [0, 1], where 1 is always best. ``direction=None`` metrics are treated
    as higher-is-better (callers should weight them deliberately).
    Constant columns map to 0.5 (no information either way).
    """
    out = pd.DataFrame(index=df.index)
    for key in metric_keys:
        spec = METRICS.get(key)
        col = (
            pd.to_numeric(df[key], errors="coerce")
            if key in df
            else pd.Series(np.nan, index=df.index)
        )
        lo, hi = col.min(), col.max()
        if pd.isna(lo) or hi == lo:
            out[key] = np.where(col.notna(), 0.5, np.nan)
            continue
        scaled = (col - lo) / (hi - lo)
        if spec is not None and spec.direction == "lower":
            scaled = 1.0 - scaled
        out[key] = scaled
    return out


def composite_scores(
    df: pd.DataFrame,
    weights: Mapping[str, float],
    constraints: Mapping[str, Mapping[str, float]] | None = None,
) -> pd.DataFrame:
    """Population-normalized linear-combination score per run.

    Returns a DataFrame with columns ``score`` (NaN when any weighted metric
    is missing for that run) and ``feasible`` (constraint satisfaction on the
    RAW metric values; e.g. ``{"silhouette": {"min": 0.05}}``). ``feasible``
    is an informational flag — every run was already computed, so nothing
    downstream discards rows on it; tables and plots only mark them.
    Weights are applied to the normalized [0, 1] columns and divided by the
    sum of |weights| so the score itself stays in [0, 1] regardless of how
    many criteria are combined.
    """
    active = {k: float(w) for k, w in weights.items() if w}
    norm = normalize_metric_columns(df, active.keys())
    total = sum(abs(w) for w in active.values()) or 1.0
    score = pd.Series(0.0, index=df.index)
    missing = pd.Series(False, index=df.index)
    for key, w in active.items():
        # Negative weights penalize a (direction-corrected) metric.
        score = score + (w / total) * norm[key].fillna(0.0)
        missing = missing | norm[key].isna()
    score[missing] = np.nan

    feasible = pd.Series(True, index=df.index)
    for key, bounds in (constraints or {}).items():
        col = pd.to_numeric(df.get(key), errors="coerce")
        lo = bounds.get("min")
        hi = bounds.get("max")
        ok = col.notna()
        if lo is not None:
            ok &= col >= lo
        if hi is not None:
            ok &= col <= hi
        feasible &= ok

    return pd.DataFrame({"score": score, "feasible": feasible}, index=df.index)


def pareto_front_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated points for two maximize-axes.

    A skyline sweep — O(n log n) — where the pairwise definition ("nothing
    else is at least as good on both axes and strictly better on one") is
    O(n²). Duplicated points don't dominate each other, so they all stay on
    the front. NaNs are never on the front.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    n = len(xs)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    finite = ~(np.isnan(xs) | np.isnan(ys))
    order = [i for i in np.lexsort((-ys, -xs)) if finite[i]]  # x desc, then y desc
    best_y = -np.inf
    for i in order:
        if ys[i] > best_y:
            mask[i] = True
            best_y = ys[i]
    if mask.any():
        front_pts = set(zip(xs[mask].tolist(), ys[mask].tolist()))
        for i in range(n):
            if finite[i] and not mask[i] and (xs[i], ys[i]) in front_pts:
                mask[i] = True
    return mask


def pareto_mask(df: pd.DataFrame, metric_keys: list[str]) -> pd.Series:
    """Boolean mask of Pareto-optimal rows over the given metrics.

    Direction-corrected via the registry (``None`` treated as
    higher-is-better). Rows with any NaN among the metrics are never on the
    front.
    """
    cols = []
    for key in metric_keys:
        col = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
        spec = METRICS.get(key)
        if spec is not None and spec.direction == "lower":
            col = -col
        cols.append(col)
    values = np.stack(cols, axis=1)
    valid = ~np.isnan(values).any(axis=1)
    front = valid.copy()
    idx = np.flatnonzero(valid)
    for i in idx:
        if not front[i]:
            continue
        dominated = (values[idx] >= values[i]).all(axis=1) & (values[idx] > values[i]).any(axis=1)
        if dominated.any():
            front[i] = False
    return pd.Series(front, index=df.index)


def best_per_k(
    df: pd.DataFrame,
    score_col: str = "score",
    k_col: str = "k",
) -> pd.DataFrame:
    """For each K (model complexity), the best-scoring run.

    K is the user's complexity choice, not something to optimize — so the
    deliverable is one recommended configuration per K. Every run
    participates; a ``feasible`` column (constraint flag), if present, rides
    along for display but never excludes a run.
    """
    sub = df.dropna(subset=[score_col, k_col])
    if sub.empty:
        return sub
    idx = sub.groupby(k_col)[score_col].idxmax()
    return df.loc[idx.values].sort_values(k_col).reset_index(drop=True)


# ── _target_-instantiable wrappers ───────────────────────────────────────────
# The scoring functions above are the API the GUI and tests call directly.
# These thin dataclasses expose the same logic for hydra ``instantiate`` so a
# selection rule (population-level) or per-run objective lives in config as a
# ``_target_`` block, like every pipeline stage.


@dataclass
class CompositeSelection:
    """Population-level model selection: a weighted, normalized score per run.

    ``weights`` and ``constraints`` carry the same meaning as
    :func:`composite_scores` (the underlying implementation). Configure it as::

        selection:
          _target_: behavior_graphs.model_selection.scoring.CompositeSelection
          weights: {silhouette: 1.0, boundary_gain: 1.0}
          constraints: {silhouette: {min: 0.05}}
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    constraints: dict[str, dict[str, float]] = field(
        default_factory=lambda: {"silhouette": {"min": DEFAULT_SILHOUETTE_FLOOR}}
    )

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        return composite_scores(df, dict(self.weights), _plain(self.constraints))


@dataclass
class Objective:
    """Per-run scalar objective (raw weighted sum + constraint flags).

    Wraps :func:`score_metrics`; used by the pipeline's ``objective`` config
    so a hydra/optuna sweep can optimize a linear combination of metrics.
    """

    weights: dict[str, float] = field(default_factory=dict)
    constraints: dict[str, dict[str, float]] = field(default_factory=dict)

    def score(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        return score_metrics(
            metrics, {"weights": dict(self.weights), "constraints": _plain(self.constraints)}
        )


def _plain(constraints: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Coerce a (possibly OmegaConf) constraints mapping to plain dicts."""
    return {k: {kk: vv for kk, vv in v.items()} for k, v in dict(constraints).items()}
