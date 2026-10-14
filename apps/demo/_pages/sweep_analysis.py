"""Sweep Analysis — per-metric winners + Pareto frontiers over pipeline runs.

Every pipeline run (including each job of a hydra multirun sweep) writes a
self-contained run dir with ``config.yaml`` (parameters) and ``metrics.json``
(evaluator outputs). This page indexes every run found under
``$BEHAVIOR_GRAPHS_RUNS_ROOT`` (default ``./runs:./multirun``) and compares
them on the cluster-quality metrics:

  - silhouette            (higher = better) — geometric cluster cohesion.
  - calinski_harabasz     (higher = better) — between/within dispersion ratio.
  - davies_bouldin        (lower  = better) — mean cluster similarity.

Plus diagnostic-only axes (no clear direction):

  - noise_fraction          fraction of windows labeled as noise (-1)
  - mean_transition_entropy mean outgoing-edge entropy per cluster node

This page answers two questions:

  1. **Which (clustering, K, W, S, reduction) wins each metric?**
  2. **What does the metric vs. each parameter axis look like?** — per
     parameter, a scatter of all runs overlaid with the Pareto-best envelope.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from behavior_graphs.model_selection.index import varying_columns
from behavior_graphs.model_selection.scoring import METRICS as _METRIC_SPECS
from behavior_graphs.model_selection.scoring import pareto_front_2d
from behavior_graphs.viz.streamlit.data import (
    display_run_dirs,
    resolve_data_path,
    runs_index,
    runs_roots,
    task_display_name,
)
from behavior_graphs.viz.streamlit.lazy import lazy_section

_DATA_PATH = resolve_data_path()

st.header("Sweep Analysis")


# ── Metric directions ────────────────────────────────────────────────────────

# View of the shared registry (behavior_graphs.model_selection.scoring.METRICS)
# in the shape this page renders:
#   "higher" — more is unambiguously better
#   "lower"  — less is unambiguously better
#   None     — informative but ambiguous; do not pick winners or draw a frontier
METRICS: dict[str, dict] = {
    spec.key: {"label": spec.label, "direction": spec.direction, "fmt": spec.fmt}
    for spec in _METRIC_SPECS.values()
}

#: Core parameter axes; any other hyperparameter that varies in the indexed
#: runs is appended dynamically below.
PARAM_AXES = [
    ("k", "K (clusters)", "numeric"),
    ("w", "W (window)", "numeric"),
    ("s", "S (stride)", "numeric"),
    ("clustering", "Clustering", "categorical"),
    ("reduction", "Reduction", "categorical"),
]


# ── Index loader (shared with the sweep runner + Graph Demo) ─────────────────


_SHORT_COLUMNS = {"k": "k", "w": "windowing.width", "s": "windowing.stride"}


def _with_short_columns(df: pd.DataFrame) -> pd.DataFrame:
    """The shared runs index plus this page's legacy k/w/s short columns.

    Cheap and vectorised (three ``to_numeric`` passes), so it runs per rerun on
    the shared cached frame rather than justifying a second cache entry — see
    :func:`~behavior_graphs.viz.streamlit.data.runs_index` for why there is only
    one of those.

    A source column can be missing entirely — a sweep of purely legacy-manifest
    runs has ``windowing`` but no ``windowing.width`` — and ``df.get`` then
    returns None, which ``pd.to_numeric`` turns into a bare scalar nan with no
    ``.fillna``. Fill from an explicit all-zero Series instead of tracebacking
    the whole page over an absent hyperparameter.
    """
    if df.empty:
        return df
    for short, source in _SHORT_COLUMNS.items():
        col = df[source] if source in df.columns else pd.Series(0, index=df.index)
        df[short] = pd.to_numeric(col, errors="coerce").fillna(0).astype(int)
    return df


_run_dirs = display_run_dirs()
df = _with_short_columns(runs_index(_run_dirs))

# Append any other varying hyperparameter as an axis (e.g. umap n_neighbors
# or graph smoothing when a sweep ablates them).
_known_axes = {a for a, _, _ in PARAM_AXES} | {"windowing.width", "windowing.stride"}
if not df.empty:
    for col in varying_columns(df):
        if col in _known_axes or col in ("clustering", "reduction"):
            continue
        kind = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        PARAM_AXES.append((col, col, kind))

if df.empty:
    st.error(
        "No run directories found under "
        + ", ".join(str(r) for r in runs_roots() or [Path("runs"), Path("multirun")])
        + ". Run the pipeline (optionally with `-m` for a sweep) first, or set "
        "`$BEHAVIOR_GRAPHS_RUNS_ROOT`."
    )
    st.stop()

# Scoping to one dataset first keeps every chart below meaningful *and*
# small: a full sweep root is thousands of runs from unrelated tasks, and
# scattering all of them ships a multi-megabyte Plotly payload per figure.
_datasets = sorted(str(d) for d in df.get("rollout_dir", pd.Series(dtype=object)).dropna().unique())
if len(_datasets) > 1:
    _picked = st.selectbox(
        "Dataset (rollout dir)",
        ["(all datasets)"] + _datasets,
        format_func=lambda d: task_display_name(Path(d).name),
        key="sa_dataset",
    )
    sub = df if _picked == "(all datasets)" else df[df["rollout_dir"] == _picked]
    sub = sub.reset_index(drop=True)
else:
    sub = df.copy()

st.caption(
    f"{len(sub)} runs indexed  "
    f"({sub['clustering'].nunique()} clustering(s), {sub['reduction'].nunique()} reduction(s), "
    f"{sub['k'].nunique()} K value(s), {sub['w'].nunique()} W value(s), "
    f"{sub['s'].nunique()} S value(s))"
)


# ── 1. Optimal config per metric ─────────────────────────────────────────────

st.subheader("Optimal config per metric")

winners_rows: list[dict] = []
for metric, info in METRICS.items():
    if metric not in sub.columns:
        continue
    if info.get("direction") is None:
        continue  # ambiguous metric — no "best" to report
    valid = sub.dropna(subset=[metric])
    if valid.empty:
        continue
    higher = info["direction"] == "higher"
    idx = valid[metric].idxmax() if higher else valid[metric].idxmin()
    best = valid.loc[idx]
    winners_rows.append(
        {
            "metric": info["label"],
            "value": f"{best[metric]:{info['fmt']}}",
            "clustering": best["clustering"],
            "reduction": best["reduction"],
            "K": int(best["k"]),
            "W": int(best["w"]),
            "S": int(best["s"]),
            "run": best["run"],
        }
    )

if winners_rows:
    winners_df = pd.DataFrame(winners_rows).set_index("metric")
    st.dataframe(winners_df, use_container_width=True)
    ambig = [info["label"] for m, info in METRICS.items() if info.get("direction") is None]
    if ambig:
        st.caption(
            "Excluded (no unambiguous direction — diagnostic only): "
            + ", ".join(f"**{x}**" for x in ambig)
            + "."
        )
else:
    st.info("No metric values available.")


# ── 2. Pareto frontier per parameter ─────────────────────────────────────────

st.subheader("Pareto frontier per parameter")
st.caption(
    "For each (metric, parameter) pair, scatter shows every run's score. "
    "Metrics with a clear direction (↑/↓) get an overlay line showing the best "
    "achievable score at each parameter value; ambiguous/diagnostic metrics "
    "get scatter only."
)

metric_pick = st.selectbox(
    "Metric",
    list(METRICS.keys()),
    format_func=lambda k: METRICS[k]["label"],
    index=0,
    key="sa_metric",
)


# ── Per-metric formula / definition ─────────────────────────────────────────
_METRIC_FORMULAS: dict[str, str] = {
    "silhouette": r"""
**Silhouette** — geometric cluster cohesion in the reduced embedding space.

Per-sample score:
$$
a_i = \frac{1}{|C_{c_i}| - 1} \sum_{j \in C_{c_i},\, j \neq i} d(i,j),
\qquad
b_i = \min_{c \neq c_i} \frac{1}{|C_c|} \sum_{j \in C_c} d(i,j),
\qquad
s_i = \frac{b_i - a_i}{\max(a_i,\, b_i)} \in [-1, 1]
$$

Reported value: $\bar s = \frac{1}{N} \sum_i s_i$, via
`sklearn.metrics.silhouette_score` (noise windows excluded).
""",
    "calinski_harabasz": r"""
**Calinski-Harabasz** — ratio of between-cluster to within-cluster dispersion;
higher means denser, better-separated clusters.
""",
    "davies_bouldin": r"""
**Davies-Bouldin** — mean over clusters of the highest similarity to another
cluster; lower means better separation.
""",
    "noise_fraction": r"""
**Noise fraction** — fraction of windows labeled ``-1`` by the clusterer
(HDBSCAN). **Diagnostic only** — zero can mean forced assignment; high values
mean the clustering covers little of the data.
""",
    "mean_transition_entropy": r"""
**Transition entropy** — mean Shannon entropy of each cluster node's outgoing
transition distribution. **Diagnostic only** — low entropy means deterministic
flow (or a degenerate chain); high entropy means heavy branching.
""",
    "swap_rate": r"""
**Swap rate / frame** — adjacent same-episode window pairs with differing
labels, divided by total observation frames (stride-fair). Lower means
temporally coherent labels.
""",
    "mean_segment_length": r"""
**Segment length** — mean length (in windows) of consecutive same-label runs.
**Diagnostic only** — scales with windowing stride.
""",
    "distinct_per_episode": r"""
**Distinct/episode** — distinct cluster labels visited per episode after
run-length collapse. **Diagnostic only** — should roughly match the true
number of task phases.
""",
    "n_clusters": r"""
**Clusters found** — number of non-noise clusters realized by the run
(differs from the configured K for density clusterers). Diagnostic.
""",
}

with st.expander(f"How is **{METRICS[metric_pick]['label']}** computed?", expanded=False):
    st.markdown(
        _METRIC_FORMULAS.get(
            metric_pick,
            f"_(no formula registered for `{metric_pick}`)_",
        )
    )


info = METRICS[metric_pick]
direction = info.get("direction")  # "higher" | "lower" | None
higher_better = direction == "higher"
has_direction = direction is not None
valid = sub.dropna(subset=[metric_pick]).copy()

if valid.empty:
    st.warning(f"No `{metric_pick}` values found.")
    st.stop()


def _frontier_numeric(
    d: pd.DataFrame, axis: str, metric: str, higher: bool
) -> tuple[np.ndarray, np.ndarray]:
    grouped = d.groupby(axis)[metric].agg("max" if higher else "min").sort_index()
    return grouped.index.values, grouped.values


def _hover_cols(d: pd.DataFrame) -> np.ndarray:
    return np.stack(
        [
            d["clustering"].values,
            d["reduction"].values,
            d["k"].values,
            d["w"].values,
            d["s"].values,
            d["run"].values,
        ],
        axis=-1,
    )


_HOVER_TMPL = (
    "<b>%{y:.4f}</b><br>"
    "clustering: %{customdata[0]}<br>"
    "reduction: %{customdata[1]}<br>"
    "K=%{customdata[2]}, W=%{customdata[3]}, S=%{customdata[4]}<br>"
    "run: %{customdata[5]}"
    "<extra></extra>"
)


_show_param_charts = lazy_section(
    f"Per-parameter charts for **{info['label']}** ({len(PARAM_AXES)} axes)",
    key="sa_param_charts",
    default_open=True,
    help="Collapse to skip building these figures entirely.",
)

for axis, axis_label, kind in PARAM_AXES if _show_param_charts else []:
    # Rows where the axis doesn't apply (e.g. clustering.n_clusters for
    # HDBSCAN runs) are dropped for this axis only.
    vax = valid.dropna(subset=[axis])
    if vax[axis].nunique() <= 1:
        continue
    fig = go.Figure()

    if kind == "numeric":
        unique_vals = sorted(vax[axis].unique().tolist())
        if len(unique_vals) <= 20:
            # Discrete axis — box plot per value shows the distribution clearly.
            for v in unique_vals:
                sel = vax[vax[axis] == v]
                fig.add_trace(
                    go.Box(
                        x=[v] * len(sel),
                        y=sel[metric_pick].values,
                        name=str(int(v)) if isinstance(v, float) and v.is_integer() else str(v),
                        boxpoints="all",
                        jitter=0.4,
                        pointpos=0.0,
                        marker=dict(size=4, color="#888", opacity=0.4),
                        line=dict(color="#444"),
                        fillcolor="rgba(150,150,150,0.15)",
                        showlegend=False,
                        customdata=_hover_cols(sel),
                        hovertemplate=_HOVER_TMPL,
                        width=max(
                            0.6, (max(unique_vals) - min(unique_vals)) / len(unique_vals) * 0.6
                        )
                        if len(unique_vals) > 1
                        else 0.6,
                    )
                )
            fig.update_xaxes(
                title=axis_label,
                tickmode="array",
                tickvals=unique_vals,
                ticktext=[
                    str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
                    for v in unique_vals
                ],
            )
        else:
            # Dense continuous axis — scatter with jitter.
            rng = np.random.default_rng(0)
            x_jitter = vax[axis].astype(float).values + rng.uniform(-0.15, 0.15, len(vax))
            fig.add_trace(
                go.Scatter(
                    x=x_jitter,
                    y=vax[metric_pick].values,
                    mode="markers",
                    marker=dict(size=6, opacity=0.4, color="#888"),
                    name="all runs",
                    customdata=_hover_cols(vax),
                    hovertemplate=_HOVER_TMPL,
                )
            )
            fig.update_xaxes(title=axis_label, tickmode="array", tickvals=unique_vals)
        # Frontier overlay: identical for both trace styles above.
        if has_direction:
            xs, ys = _frontier_numeric(vax, axis, metric_pick, higher_better)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines+markers",
                    line=dict(color="crimson", width=2),
                    marker=dict(size=9, color="crimson"),
                    name=f"best {'↑' if higher_better else '↓'}",
                )
            )
    else:
        # Categorical: box per category + best marker on top
        cats = sorted(vax[axis].astype(str).unique().tolist())
        vax = vax.assign(_cat=vax[axis].astype(str))
        for c in cats:
            sel = vax[vax["_cat"] == c]
            fig.add_trace(
                go.Box(
                    y=sel[metric_pick].values,
                    name=str(c),
                    boxpoints="all",
                    jitter=0.4,
                    pointpos=0.0,
                    marker=dict(size=4, color="#888", opacity=0.4),
                    line=dict(color="#444"),
                    fillcolor="rgba(150,150,150,0.15)",
                    showlegend=False,
                    customdata=_hover_cols(sel),
                    hovertemplate=_HOVER_TMPL,
                )
            )
        if has_direction:
            bests = [
                vax[vax["_cat"] == c][metric_pick].agg("max" if higher_better else "min")
                for c in cats
            ]
            fig.add_trace(
                go.Scatter(
                    x=cats,
                    y=bests,
                    mode="lines+markers",
                    line=dict(color="crimson", width=2),
                    marker=dict(size=11, color="crimson", symbol="diamond"),
                    name=f"best {'↑' if higher_better else '↓'}",
                )
            )
        fig.update_xaxes(title=axis_label)

    yaxis_kwargs: dict = {"title": info["label"]}
    if info["fmt"].endswith("%"):
        yaxis_kwargs["tickformat"] = ".0%"
    fig.update_yaxes(**yaxis_kwargs)
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        title=f"{info['label']} vs {axis_label}",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 3. Joint Pareto frontier (configurable axes + highlight) ────────────────

st.subheader("Joint Pareto frontier")
st.caption(
    "Each point is one run. The **Pareto frontier** is the set of runs that "
    "are not dominated by any other — you can't improve one axis without "
    "giving up the other. Pick the two metrics to compare on, and optionally "
    "highlight a subset of points by parameter value."
)

# Axis pickers
_metric_keys = list(METRICS.keys())
_col_x, _col_y = st.columns(2)
with _col_x:
    x_metric = st.selectbox(
        "X axis",
        _metric_keys,
        index=_metric_keys.index("silhouette"),
        format_func=lambda k: METRICS[k]["label"],
        key="joint_x_metric",
    )
with _col_y:
    _default_y = "davies_bouldin"
    y_metric = st.selectbox(
        "Y axis",
        _metric_keys,
        index=_metric_keys.index(_default_y),
        format_func=lambda k: METRICS[k]["label"],
        key="joint_y_metric",
    )

# Highlight picker — by which parameter (or none), and which values
_PARAMS = [
    ("clustering", "Clustering"),
    ("reduction", "Reduction"),
    ("k", "K"),
    ("w", "W"),
    ("s", "S"),
]
_col_hp, _col_hv = st.columns([1, 2])
with _col_hp:
    hl_param_label = st.selectbox(
        "Highlight by",
        ["(none)"] + [p[1] for p in _PARAMS],
        key="joint_hl_param",
    )
hl_param = next((p[0] for p in _PARAMS if p[1] == hl_param_label), None)
with _col_hv:
    if hl_param is None:
        hl_values: list = []
        st.caption("Pick a parameter to enable value highlighting.")
    else:
        all_vals = sorted(sub[hl_param].dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        hl_values = st.multiselect(
            f"Highlight values of `{hl_param}`",
            all_vals,
            default=[],
            key="joint_hl_values",
        )

joint = sub.dropna(subset=[x_metric, y_metric]).copy()
if joint.empty or x_metric == y_metric:
    st.info("No runs with both metrics available, or the two axes are the same metric.")
else:
    # Direction-aware Pareto. Convert each axis so "higher is better" for the
    # dominance check; metrics marked None are treated as "higher is better"
    # by convention (frontier is best-guess only — caveat in caption below).
    def _sign(direction):
        return 1.0 if direction != "lower" else -1.0

    x_sgn = _sign(METRICS[x_metric]["direction"])
    y_sgn = _sign(METRICS[y_metric]["direction"])
    xs_dom = joint[x_metric].values * x_sgn
    ys_dom = joint[y_metric].values * y_sgn
    joint["on_front"] = pareto_front_2d(xs_dom, ys_dom)
    if hl_param is not None and hl_values:
        joint["highlighted"] = joint[hl_param].isin(hl_values)
    else:
        joint["highlighted"] = True  # no highlight = everyone "selected"

    # Color: if highlighting active, highlighted points keep a reduction-based
    # color; everyone else fades to gray. If no highlight, use reduction colors.
    _REDUCTION_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    _reductions = sorted(joint["reduction"].unique().tolist())
    _REDUCTION_COLOR = {
        r: _REDUCTION_COLORS[i % len(_REDUCTION_COLORS)] for i, r in enumerate(_reductions)
    }

    def _point_color(o: str, highlighted: bool) -> str:
        if hl_values and not highlighted:
            return "#cccccc"
        return _REDUCTION_COLOR.get(o, "#888")

    def _point_opacity(highlighted: bool, faint: bool) -> float:
        # faint = "dominated"; highlighted overrides faintness.
        if hl_values:
            if not highlighted:
                return 0.12
            return 1.0
        return 0.25 if faint else 1.0

    fig_joint = go.Figure()

    _x_label = METRICS[x_metric]["label"]
    _y_label = METRICS[y_metric]["label"]
    _hover = (
        f"{_x_label}: %{{x:{METRICS[x_metric]['fmt']}}}<br>"
        f"{_y_label}: %{{y:{METRICS[y_metric]['fmt']}}}<br>"
        "clustering: %{customdata[0]}<br>"
        "reduction: %{customdata[1]}<br>"
        "K=%{customdata[2]}, W=%{customdata[3]}, S=%{customdata[4]}<br>"
        "run: %{customdata[5]}"
        "<extra></extra>"
    )

    def _hover_cd(d: pd.DataFrame) -> np.ndarray:
        return np.stack(
            [
                d["clustering"].values,
                d["reduction"].values,
                d["k"].values,
                d["w"].values,
                d["s"].values,
                d["run"].values,
            ],
            axis=-1,
        )

    # Background: dominated points
    dom = joint[~joint["on_front"]]
    if not dom.empty:
        fig_joint.add_trace(
            go.Scatter(
                x=dom[x_metric],
                y=dom[y_metric],
                mode="markers",
                marker=dict(
                    size=7,
                    color=[
                        _point_color(o, h) for o, h in zip(dom["reduction"], dom["highlighted"])
                    ],
                    opacity=[_point_opacity(h, faint=True) for h in dom["highlighted"]],
                    line=dict(width=0),
                ),
                name="dominated",
                customdata=_hover_cd(dom),
                hovertemplate=_hover,
            )
        )

    # Foreground: frontier staircase + points
    fr = joint[joint["on_front"]].sort_values([x_metric, y_metric])
    if not fr.empty:
        xs_fr = fr[x_metric].tolist()
        ys_fr = fr[y_metric].tolist()
        step_x: list[float] = [xs_fr[0]]
        step_y: list[float] = [ys_fr[0]]
        for i in range(1, len(xs_fr)):
            step_x.extend([xs_fr[i], xs_fr[i]])
            step_y.extend([step_y[-1], ys_fr[i]])
        fig_joint.add_trace(
            go.Scatter(
                x=step_x,
                y=step_y,
                mode="lines",
                line=dict(color="crimson", width=3),
                name="Pareto frontier",
                hoverinfo="skip",
                showlegend=True,
            )
        )
        fig_joint.add_trace(
            go.Scatter(
                x=xs_fr,
                y=ys_fr,
                mode="markers",
                marker=dict(
                    size=13,
                    color=[_point_color(o, h) for o, h in zip(fr["reduction"], fr["highlighted"])],
                    opacity=[_point_opacity(h, faint=False) for h in fr["highlighted"]],
                    line=dict(width=2, color="crimson"),
                    symbol="diamond",
                ),
                name="Pareto frontier (points)",
                customdata=_hover_cd(fr),
                hovertemplate="<b>FRONTIER</b><br>" + _hover,
            )
        )

    # Legend proxies for reduction colors (always shown)
    for red_name, red_col in _REDUCTION_COLOR.items():
        fig_joint.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=red_col),
                name=red_name,
                showlegend=True,
            )
        )

    # Title + axis direction arrows
    def _direction_arrow(direction, axis: str) -> str:
        if direction == "higher":
            return "higher better →" if axis == "x" else "higher better ↑"
        if direction == "lower":
            return "← lower better" if axis == "x" else "lower better ↓"
        return "diagnostic"

    x_title = f"{_x_label} ({_direction_arrow(METRICS[x_metric]['direction'], 'x')})"
    y_title = f"{_y_label} ({_direction_arrow(METRICS[y_metric]['direction'], 'y')})"

    title_suffix = ""
    if hl_param and hl_values:
        title_suffix = (
            f" — highlighting {hl_param}∈" + "{" + ", ".join(str(v) for v in hl_values) + "}"
        )

    fig_joint.update_layout(
        title=f"{_x_label} × {_y_label} — joint Pareto frontier" + title_suffix,
        xaxis_title=x_title,
        yaxis=dict(title=y_title),
        height=480,
        margin=dict(l=50, r=20, t=50, b=50),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )

    if METRICS[x_metric]["direction"] is None or METRICS[y_metric]["direction"] is None:
        st.caption(
            "⚠️ At least one axis is a diagnostic-only metric (no clear best "
            "direction). The Pareto frontier defaults to *higher is better* for "
            "those axes — interpret the frontier accordingly."
        )

    st.plotly_chart(fig_joint, use_container_width=True)

    n_front = int(joint["on_front"].sum())
    n_hl = int(joint["highlighted"].sum()) if hl_values else len(joint)
    if hl_values:
        st.caption(
            f"**{n_front} of {len(joint)}** runs are Pareto-optimal. "
            f"**{n_hl}** match the highlight filter."
        )
    else:
        st.caption(f"**{n_front} of {len(joint)}** runs are Pareto-optimal.")

    with st.expander(f"Pareto-frontier configs ({n_front} rows)"):
        front_cols = ["run", "clustering", "reduction", "k", "w", "s", x_metric, y_metric]
        front_df = (
            joint[joint["on_front"]][front_cols]
            .sort_values(x_metric, ascending=(METRICS[x_metric]["direction"] == "lower"))
            .reset_index(drop=True)
        )
        st.dataframe(front_df, use_container_width=True)


# ── 4. Raw table (collapsed) ─────────────────────────────────────────────────

if lazy_section(f"Raw scores ({len(valid)} rows)", key="sa_raw_table"):
    show_cols = ["run", "clustering", "reduction", "k", "w", "s"] + list(METRICS.keys())
    sort_ascending = direction == "lower"
    st.dataframe(
        valid[show_cols].sort_values(metric_pick, ascending=sort_ascending).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )
