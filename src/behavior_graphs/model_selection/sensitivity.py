"""Per-hyperparameter sensitivity of proxy metrics over a sweep.

Given the runs index (one row per run) this answers, for every
hyperparameter that varied in the sweep: *how much does this knob matter
for metric X, and which setting wins?* Two complementary views:

- **Main effects** — group runs by the parameter's value; report each
  group's mean and best metric, and the variance explained by the grouping
  (eta-squared: between-group sum of squares over total sum of squares).
  Confounded when the sweep grid is not full-factorial.
- **Matched pairs** — group runs by *all other* varying hyperparameters;
  within each matched group (where only this parameter changes) measure the
  metric spread the parameter induces and which value wins. This isolates
  the parameter's effect and is the number to trust for "does this knob
  matter, all else equal".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from behavior_graphs.model_selection.scoring import METRICS


def _direction_sign(metric: str) -> float:
    spec = METRICS.get(metric)
    return -1.0 if spec is not None and spec.direction == "lower" else 1.0


def eta_squared(df: pd.DataFrame, param: str, metric: str) -> float | None:
    """Variance in ``metric`` explained by grouping on ``param`` (0..1)."""
    sub = df.dropna(subset=[metric])
    if sub.empty or sub[param].astype(str).nunique() < 2:
        return None
    values = sub[metric].astype(float)
    grand = values.mean()
    ss_total = float(((values - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    ss_between = float(
        sub.groupby(sub[param].astype(str))[metric]
        .agg(lambda g: len(g) * (g.mean() - grand) ** 2)
        .sum()
    )
    return ss_between / ss_total


def matched_effects(
    df: pd.DataFrame, param: str, metric: str, other_params: list[str]
) -> dict[str, object]:
    """Effect of ``param`` on ``metric`` holding all other params fixed.

    Returns mean within-group spread (best minus worst, direction-corrected
    so positive = the knob can buy you that much), the number of matched
    groups, and per-value win counts (how often each setting is the group's
    best).
    """
    sub = df.dropna(subset=[metric]).copy()
    sign = _direction_sign(metric)
    sub["_v"] = sub[metric].astype(float) * sign
    others = [c for c in other_params if c != param and c in sub.columns]
    group_keys = (
        sub[others].fillna("").astype(str).agg("|".join, axis=1)
        if others
        else pd.Series("", index=sub.index)
    )
    spreads: list[float] = []
    wins: dict[str, int] = {}
    n_groups = 0
    for _, g in sub.groupby(group_keys):
        if g[param].astype(str).nunique() < 2:
            continue
        n_groups += 1
        # Best metric achievable per setting of `param` within this group.
        per_value = g.groupby(g[param].astype(str))["_v"].max()
        spreads.append(float(per_value.max() - per_value.min()))
        winner = str(per_value.idxmax())
        wins[winner] = wins.get(winner, 0) + 1
    return {
        "n_matched_groups": n_groups,
        "mean_spread": float(np.mean(spreads)) if spreads else None,
        "max_spread": float(np.max(spreads)) if spreads else None,
        "wins": wins,
    }


def sensitivity_table(
    df: pd.DataFrame,
    params: list[str],
    metric: str,
) -> pd.DataFrame:
    """One row per hyperparameter: how sensitive ``metric`` is to it.

    Columns: ``n_values``, ``eta_squared`` (main effect), ``mean_spread`` /
    ``max_spread`` (matched-pair effect, direction-corrected so positive is
    the gain available from tuning this knob), ``best_value`` (most frequent
    matched-group winner; falls back to the best group mean when no matched
    groups exist), ``n_matched_groups``.
    """
    rows: list[dict[str, object]] = []
    sign = _direction_sign(metric)
    for param in params:
        if param not in df.columns:
            continue
        sub = df.dropna(subset=[metric])
        n_values = sub[param].astype(str).nunique()
        if n_values < 2:
            continue
        matched = matched_effects(df, param, metric, params)
        wins: dict[str, int] = matched["wins"]  # type: ignore[assignment]
        if wins:
            best_value = max(wins.items(), key=lambda kv: kv[1])[0]
        else:
            means = sub.groupby(sub[param].astype(str))[metric].mean() * sign
            best_value = str(means.idxmax()) if not means.empty else None
        rows.append(
            {
                "param": param,
                "n_values": n_values,
                "eta_squared": eta_squared(df, param, metric),
                "mean_spread": matched["mean_spread"],
                "max_spread": matched["max_spread"],
                "n_matched_groups": matched["n_matched_groups"],
                "best_value": best_value,
                "wins": ", ".join(
                    f"{k}×{v}" for k, v in sorted(wins.items(), key=lambda kv: -kv[1])
                )
                or None,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("eta_squared", ascending=False, na_position="last").reset_index(
            drop=True
        )
    return out
