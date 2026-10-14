"""Model selection over all indexed runs: scores, Pareto, best-per-K.

Renders, for the population of run directories the app can see:

- a weight editor for the composite selection score (linear combination of
  population-normalized, direction-corrected metrics) plus a silhouette
  floor that *flags* runs below it ("necessary but not sufficient" — every
  computed run stays in the tables and plots);
- a sortable table of hyperparameters × eval scores (click a column header
  to sort);
- the best remaining hyperparameters for each K — K is the user's
  complexity choice, so it is never optimized over, only conditioned on;
- a two-metric Pareto frontier scatter;
- per-hyperparameter sensitivity tables (main effects + matched pairs).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from behavior_graphs.model_selection.index import (
    hyperparameter_columns,
    index_schema_version,
    varying_columns,
)
from behavior_graphs.model_selection.scoring import (
    DEFAULT_SILHOUETTE_FLOOR,
    DEFAULT_WEIGHTS,
    METRICS,
    best_per_k,
    composite_scores,
    pareto_mask,
)
from behavior_graphs.model_selection.sensitivity import sensitivity_table
from behavior_graphs.viz.streamlit.data import (
    BEST_PER_K_NON_PARAM_KEYS,
    dataset_display_name,
    display_run_dirs,
    metrics_fingerprint,
    paths_token,
    run_task_name,
    runs_index,
)

_log = logging.getLogger(__name__)


def _weight_editor(key_prefix: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Sliders for every directional metric + the silhouette floor."""
    directional = [m for m in METRICS.values() if m.direction is not None]
    st.caption(
        "Composite score = Σ wᵢ · normᵢ, where normᵢ min-max scales metric i "
        "over all runs and flips lower-is-better metrics, so 1 is always the "
        "best run in the population. The silhouette floor flags runs below "
        "it (silhouette is necessary, not sufficient) — flagged runs stay "
        "in every table and plot, they are just marked."
    )
    weights: dict[str, float] = {}
    cols = st.columns(len(directional))
    for col, spec in zip(cols, directional):
        with col:
            weights[spec.key] = st.slider(
                spec.label,
                min_value=0.0,
                max_value=2.0,
                value=float(DEFAULT_WEIGHTS.get(spec.key, 0.0)),
                step=0.05,
                key=f"{key_prefix}_w_{spec.key}",
                help=f"direction: {spec.direction} is better (auto-corrected)",
            )
    floor = st.slider(
        "Silhouette floor (flags runs below it; nothing is discarded)",
        min_value=-0.2,
        max_value=0.5,
        value=float(DEFAULT_SILHOUETTE_FLOOR),
        step=0.01,
        key=f"{key_prefix}_floor",
    )
    return weights, {"silhouette": {"min": floor}}


_MS_COL_TO_PARAM: dict[str, str] = {
    "k": "k",
    "windowing.width": "w",
    "windowing.stride": "s",
    "features.layer": "layer",
    "reduction.n_components": "n_components",
    # Which representation the winning row is on. Without it, applying that
    # row's window/reduction settings would strand the picker on a combination
    # no run has: the winner may well be on another representation. The column
    # is precomputed by flatten_config, so there is nothing to reconstruct.
    "representation": "representation",
}


def _row_to_param_dict(row: pd.Series) -> dict:
    """Translate one best-per-K index row to data.py's short param-key format.

    Returns a dict with keys ``k``, ``w``, ``s``, ``layer``, ``representation``,
    ``reduction``, ``n_components`` (whichever are non-null), plus ``score`` and
    ``path`` if present.
    ``k`` is included so callers that need it (``_apply_best_row``) don't
    have to handle it separately; ``compute_best_per_k`` pops it out before
    returning.

    ``path`` is the winning run itself, and it is here because the
    hyperparameters do not identify it: they are the ones the sidebar offers,
    and a dataset can hold a thousand runs at one K that agree on every one of
    them and differ in the clusterer or the smoothing. A caller that wants *the*
    best run reads ``path``; matching on the rest lands on an arbitrary member
    of the tie. Not a hyperparameter, so it is in
    :data:`~behavior_graphs.viz.streamlit.data.BEST_PER_K_NON_PARAM_KEYS` and the
    selector-state writers skip it.
    """
    params: dict = {}
    for ms_col, data_param in _MS_COL_TO_PARAM.items():
        val = row.get(ms_col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        params[data_param] = int(val) if data_param in ("k", "w", "s", "n_components") else val
    # reduction: "PCAReduction" → "pca", "UMAPReduction" → "umap", etc.
    red = row.get("reduction")
    if red and isinstance(red, str):
        params["reduction"] = red.replace("Reduction", "").lower()
    score = row.get("score")
    if score is not None and not (isinstance(score, float) and pd.isna(score)):
        params["score"] = float(score)
    path = row.get("path")
    if isinstance(path, str) and path:
        params["path"] = path
    return params


def _apply_best_row(row: pd.Series, sidebar_key: str) -> None:
    """Write one best-per-K row's hyperparams into the sidebar selector state.

    Safe to call directly only *before* any sidebar widgets have been rendered
    (e.g. from ``init_best_defaults``).  When triggered from a button click use
    the ``on_click`` callback pattern so Streamlit calls this before the next
    render pass begins.
    """
    for param, val in _row_to_param_dict(row).items():
        if param in BEST_PER_K_NON_PARAM_KEYS:
            continue
        st.session_state[f"{sidebar_key}_{param}"] = val


@st.cache_data(show_spinner=False)
def _sweep_roots_cached(token: str, _run_dir_strs: tuple[str, ...]) -> list[str]:
    """The sweep roots (grandparent of each run dir) behind this run-dir list.

    Only the path arithmetic is cached — deriving 5 000 grandparents is
    milliseconds, and `compute_best_per_k` reaches it on every rerun of every
    page that seeds its selectors. Which CSVs *exist* is deliberately left out:
    that answer decides which branch defines "best per K", so a stale one makes
    the same K resolve to different hyperparameters a minute apart.
    """
    return sorted({str(Path(s).parent.parent) for s in _run_dir_strs})


def _find_sweep_csvs(run_dirs: list[Path]) -> list[Path]:
    strs = tuple(str(d) for d in run_dirs)
    roots = _sweep_roots_cached(paths_token(strs), strs)
    return [Path(r) / "best_per_k.csv" for r in roots if (Path(r) / "best_per_k.csv").exists()]


def _compute_from_csv(csvs: list[Path]) -> dict[str, dict[int, dict]]:
    """Build ``{dataset: {k: params}}`` from precomputed best_per_k.csv files."""
    dfs = []
    for csv_path in csvs:
        try:
            dfs.append(pd.read_csv(csv_path))
        except Exception as e:
            _log.warning("failed to read %s (%s); falling back to index scoring", csv_path, e)
    if not dfs:
        return {}
    df = pd.concat(dfs, ignore_index=True)
    result: dict[str, dict[int, dict]] = {}
    for _, row in df.iterrows():
        dataset = row.get("dataset")
        if not dataset or (isinstance(dataset, float) and pd.isna(dataset)):
            continue
        params = _row_to_param_dict(row)
        k_val = params.pop("k", None)
        if k_val is None:
            continue
        result.setdefault(str(dataset), {})[k_val] = params
    return result


@st.cache_data(show_spinner=False)
def _cached_csv_best_per_k(
    csv_strs: tuple[str, ...],
    fingerprint: tuple[float, ...],  # noqa: ARG001  (cache key: mtimes)
) -> dict[str, dict[int, dict]]:
    return _compute_from_csv([Path(s) for s in csv_strs])


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_index_best_per_k(
    token: str,  # noqa: ARG001  (cache key: which run dirs)
    fingerprint: tuple[int, float],  # noqa: ARG001  (cache key: their metrics.json state)
    schema: str,  # noqa: ARG001  (cache key: the scoring code, as runs_index does)
    _run_dir_strs: tuple[str, ...],
) -> dict[str, dict[int, dict]]:
    """Score the run index per dataset and pick the best config per (task, K).

    Runs are grouped by the *dirname* of their recorded rollout_dir (not the
    raw string) so absolute-recorded legacy runs and relative-recorded new
    runs of the same dataset are scored as one population.

    *fingerprint* and *schema* are not decoration; this memo has to invalidate
    on everything ``runs_index`` does, or it drifts from the table rendered
    beside it. The run-dir list alone is unchanged when a run is re-executed in
    place (hence the fingerprint), and unchanged when the scoring code is edited
    under the bind-mounted live stack (hence the schema hash) — either way the
    sidebar would keep auto-applying a superseded winner.
    """
    df = runs_index([Path(s) for s in _run_dir_strs])
    if df.empty or "rollout_dir" not in df:
        return {}
    df = df.dropna(subset=["rollout_dir"]).reset_index(drop=True)
    df = df.assign(_dataset=df["rollout_dir"].map(lambda r: Path(str(r)).name))
    result: dict[str, dict[int, dict]] = {}
    for task_name, task_df in df.groupby("_dataset"):
        task_df = task_df.reset_index(drop=True)
        scored = composite_scores(task_df, DEFAULT_WEIGHTS, {})
        task_df = task_df.assign(score=scored["score"])
        best = best_per_k(task_df)
        task_result: dict[int, dict] = {}
        for _, row in best.iterrows():
            params = _row_to_param_dict(row)
            k_val = params.pop("k", None)
            if k_val is None:
                continue
            task_result[k_val] = params
        if task_result:
            result[str(task_name)] = task_result
    return result


def clear_best_per_k_cache() -> None:
    """Clear best-per-K caches after the run directory index is refreshed."""
    _sweep_roots_cached.clear()
    _cached_csv_best_per_k.clear()
    _cached_index_best_per_k.clear()


def compute_best_per_k(run_dirs: list[Path] | None = None) -> dict[str, dict[int, dict]]:
    """Return ``{task: {k_val: param_dict}}`` with best-scored hyperparams per task and K.

    *param_dict* uses the short param names that ``data.py``'s
    ``select_run_dir`` uses as widget keys (``w``, ``s``, ``reduction``,
    ``n_components``, ``layer``) plus ``score``.

    Key contract: task keys must match ``select_run_dir``'s task names
    (``data.py::_effective_task``). Both sides derive from the dataset spec
    name — sweep job dirs are ``<sweep>/<spec name>/<hash>`` and rollout dirs
    are ``rollouts/<spec name>`` — so the CSV ``dataset`` column, the rollout
    dirname used here, and the sidebar task name all coincide. A run laid out
    differently simply gets no auto-apply (the K callback no-ops).

    First tries the precomputed ``best_per_k.csv`` in the sweep root
    (grandparent of run dirs); falls back to scoring the run index. Both
    branches are cached with a filesystem fingerprint.
    """
    if run_dirs is None:
        run_dirs = display_run_dirs()

    csvs = _find_sweep_csvs(run_dirs)
    if csvs:
        fingerprint = tuple(c.stat().st_mtime for c in csvs if c.exists())
        result = _cached_csv_best_per_k(tuple(str(c) for c in csvs), fingerprint)
        if result:
            return result

    _strs = tuple(str(d) for d in run_dirs)
    return _cached_index_best_per_k(
        paths_token(_strs), metrics_fingerprint(run_dirs), index_schema_version(), _strs
    )


def init_best_defaults(
    sidebar_key: str, best_per_k: dict[str, dict[int, dict]] | None = None
) -> None:
    """Pre-populate sidebar selectors with the best-scoring run on first load.

    Call this *before* ``select_run_dir`` (i.e. before any sidebar widget with
    the same key prefix is rendered).  Uses a session-state sentinel so the
    auto-selection fires exactly once per browser session; users can freely
    change the selectors afterwards and they won't be reset.

    Seeds the Task selector too (first task, alphabetically, that both has
    best-per-K data and actually exists among the discovered runs), so the
    K/params defaults always describe the task being shown.

    Pass *best_per_k* (from :func:`compute_best_per_k`) to reuse one computed
    dict with the subsequent ``select_run_dir`` call.
    """
    sentinel = f"{sidebar_key}_defaults_applied"
    if st.session_state.get(sentinel):
        return
    st.session_state[sentinel] = True  # set before any early return to avoid loops

    best_k_params = best_per_k if best_per_k is not None else compute_best_per_k()
    if not best_k_params:
        return

    available_tasks = {run_task_name(d) for d in display_run_dirs()}
    tasks = [t for t in sorted(best_k_params) if t in available_tasks and best_k_params[t]]
    if not tasks:
        _log.warning(
            "best-per-K data exists for %s but none match the discovered run "
            "tasks %s — sidebar defaults not applied (task-name contract "
            "violated, see compute_best_per_k)",
            sorted(best_k_params),
            sorted(available_tasks),
        )
        return
    first_task = tasks[0]
    task_params = best_k_params[first_task]
    best_k = max(task_params, key=lambda k: task_params[k].get("score", -1))
    st.session_state[f"{sidebar_key}_task"] = first_task
    st.session_state[f"{sidebar_key}_k"] = best_k
    for param, val in task_params[best_k].items():
        if param == "score":
            continue
        st.session_state[f"{sidebar_key}_{param}"] = val
    st.rerun()


def render_model_selection(
    current_run_path: str | None = None,
    key_prefix: str = "ms",
    sidebar_key: str | None = None,
) -> None:
    run_dirs = display_run_dirs()
    df = runs_index(run_dirs)
    if df.empty:
        st.info("No runs indexed yet — run the pipeline (or a sweep) first.")
        return

    # Scores are normalized over the population being compared — mixing
    # tasks would let an easy task dominate, so filter to one rollout dir
    # (defaulting to the current run's) unless explicitly comparing all.
    datasets = sorted(d for d in df.get("rollout_dir", pd.Series()).dropna().unique())
    if len(datasets) > 1:
        current_ds = None
        if current_run_path is not None:
            match = df.loc[df["path"] == current_run_path, "rollout_dir"].dropna()
            current_ds = match.iloc[0] if not match.empty else None
        options = ["(all datasets)"] + datasets
        default_idx = options.index(current_ds) if current_ds in options else 0
        # Include the current task in the widget key so that switching tasks
        # creates a fresh widget (initialized from default_idx) rather than
        # retaining the previous task's dataset selection in internal widget
        # state. Popping the key from session_state is not sufficient because
        # Streamlit's internal widget state survives a session_state.pop().
        _task_tag = Path(current_run_path).name.split("__")[0] if current_run_path else "none"
        picked = st.selectbox(
            "Dataset (rollout dir) — scores are normalized within this population",
            options,
            index=default_idx,
            format_func=dataset_display_name,  # the "(all datasets)" option passes through
            key=f"{key_prefix}_dataset_{_task_tag}",
        )
        if picked != "(all datasets)":
            df = df[df["rollout_dir"] == picked].reset_index(drop=True)

    weights, constraints = _weight_editor(key_prefix)
    scored = composite_scores(df, weights, constraints)
    df = df.assign(score=scored["score"], feasible=scored["feasible"])

    hyper_cols = varying_columns(df) or hyperparameter_columns(df)
    metric_cols = [m for m in METRICS if df[m].notna().any()]
    show_cols = ["run", "score", "feasible"] + hyper_cols + metric_cols

    n_flagged = int((~df["feasible"]).sum())
    st.caption(
        f"**{len(df)} runs** indexed; **{n_flagged}** below the silhouette "
        f"floor (flagged `feasible=False`, never discarded). Click any "
        f"column header to sort."
    )
    styled = df[show_cols].sort_values("score", ascending=False)
    st.dataframe(
        styled,
        use_container_width=True,
        height=380,
        hide_index=True,
        column_config={
            "score": st.column_config.NumberColumn("score", format="%.3f"),
            "path": None,
        },
    )

    # ── Best hyperparameters per K ───────────────────────────────────────────
    st.markdown("**Recommended hyperparameters per K**")
    _apply_hint = (
        " Use the buttons below the table to jump to that configuration." if sidebar_key else ""
    )
    st.caption(
        "K (cluster count) is a complexity choice, not an optimization "
        "target: pick how many behaviors you want, then read off the "
        "best-scoring configuration found at that K (the `feasible` column "
        "marks the silhouette floor; it does not exclude)." + _apply_hint
    )
    best = best_per_k(df)
    if best.empty:
        st.info("No scored runs with a K to recommend from.")
    else:
        best_cols = ["k", "score", "feasible"] + hyper_cols + metric_cols
        best_cols = [c for c in dict.fromkeys(best_cols) if c in best.columns]
        _best_display = best[best_cols]
        st.dataframe(
            _best_display,
            use_container_width=True,
            hide_index=True,
            column_config={"k": st.column_config.NumberColumn("K", format="%d")},
        )
        if sidebar_key:
            _k_rows = [
                (int(row["k"]), row) for _, row in best.iterrows() if row.get("k") is not None
            ]
            if _k_rows:
                _btn_cols = st.columns(len(_k_rows))
                for _col, (_k_val, _row) in zip(_btn_cols, _k_rows):
                    _icon = "✓" if _row.get("feasible", True) else "⚠"
                    with _col:
                        st.button(
                            f"K={_k_val} {_icon}",
                            key=f"{key_prefix}_apply_{_k_val}",
                            help=f"Apply K={_k_val} (score={_row['score']:.3f})",
                            use_container_width=True,
                            on_click=_apply_best_row,
                            args=(_row, sidebar_key),
                        )

    # ── Pareto frontier ──────────────────────────────────────────────────────
    st.markdown("**Pareto frontier**")
    c1, c2 = st.columns(2)
    metric_keys = list(METRICS)
    with c1:
        x_metric = st.selectbox(
            "X metric",
            metric_keys,
            index=metric_keys.index("silhouette"),
            format_func=lambda k: METRICS[k].label,
            key=f"{key_prefix}_px",
        )
    with c2:
        y_metric = st.selectbox(
            "Y metric",
            metric_keys,
            index=metric_keys.index("davies_bouldin"),
            format_func=lambda k: METRICS[k].label,
            key=f"{key_prefix}_py",
        )
    _render_pareto(df, x_metric, y_metric, current_run_path, hyper_cols)

    # ── Sensitivity analysis ─────────────────────────────────────────────────
    # Lazy: sensitivity_table regroups the whole index per hyperparameter, which
    # is not worth doing behind a closed panel. on_change="rerun" is what makes
    # .open readable — without it the expander tracks no state at all.
    _sens_panel = st.expander("Sensitivity analysis — which knobs matter?", on_change="rerun")
    with _sens_panel:
        st.caption(
            "Per hyperparameter: **eta²** is the variance in the metric "
            "explained by that knob across all runs (confounded if the sweep "
            "is not full-factorial); **mean/max spread** is the "
            "direction-corrected gain available from tuning the knob while "
            "holding every other hyperparameter fixed (matched groups); "
            "**best value** is the setting that wins most matched groups."
        )
        if _sens_panel.open:
            sens_metric = st.selectbox(
                "Metric",
                [m for m in metric_keys if METRICS[m].direction is not None],
                format_func=lambda k: METRICS[k].label,
                key=f"{key_prefix}_sens",
            )
            params = varying_columns(df)
            if not params:
                st.info("Only one configuration indexed — nothing varies yet.")
            else:
                table = sensitivity_table(df, params, sens_metric)
                if table.empty:
                    st.info(f"No `{sens_metric}` values available.")
                else:
                    st.dataframe(table, use_container_width=True, hide_index=True)


def _render_pareto(
    df: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    current_run_path: str | None,
    hyper_cols: list[str],
) -> None:
    if x_metric == y_metric:
        st.info("Pick two different metrics.")
        return
    sub = df.dropna(subset=[x_metric, y_metric]).copy()
    if sub.empty:
        st.info("No runs with both metrics.")
        return
    sub["on_front"] = pareto_mask(sub, [x_metric, y_metric])

    def _hover(d: pd.DataFrame) -> np.ndarray:
        cfg = d[hyper_cols].astype(str).agg(" · ".join, axis=1) if hyper_cols else ""
        return np.stack([d["run"].values, cfg, d["score"].round(3).astype(str)], axis=-1)

    tmpl = "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>score: %{customdata[2]}<extra></extra>"
    fig = go.Figure()
    dom = sub[~sub["on_front"]]
    if not dom.empty:
        fig.add_trace(
            go.Scatter(
                x=dom[x_metric],
                y=dom[y_metric],
                mode="markers",
                marker=dict(size=7, color="#999", opacity=0.35),
                name="dominated",
                customdata=_hover(dom),
                hovertemplate=tmpl,
            )
        )
    front = sub[sub["on_front"]].sort_values(x_metric)
    if not front.empty:
        fig.add_trace(
            go.Scatter(
                x=front[x_metric],
                y=front[y_metric],
                mode="lines+markers",
                line=dict(color="crimson", width=2, shape="hv"),
                marker=dict(size=11, color="crimson", symbol="diamond"),
                name="Pareto frontier",
                customdata=_hover(front),
                hovertemplate=tmpl,
            )
        )
    # Mark (don't drop) frontier points below the silhouette floor.
    flagged = front[~front["feasible"]] if not front.empty else front
    if not flagged.empty:
        fig.add_trace(
            go.Scatter(
                x=flagged[x_metric],
                y=flagged[y_metric],
                mode="markers",
                marker=dict(
                    size=17,
                    color="rgba(0,0,0,0)",
                    symbol="circle-open",
                    line=dict(width=2, color="#888"),
                ),
                name="below silhouette floor",
                customdata=_hover(flagged),
                hovertemplate=tmpl,
            )
        )
    if current_run_path is not None:
        cur = sub[sub["path"] == current_run_path]
        if not cur.empty:
            fig.add_trace(
                go.Scatter(
                    x=cur[x_metric],
                    y=cur[y_metric],
                    mode="markers",
                    marker=dict(
                        size=16, symbol="star", color="#f0c419", line=dict(width=1, color="#333")
                    ),
                    name="current run",
                    customdata=_hover(cur),
                    hovertemplate=tmpl,
                )
            )

    def _axis(metric: str) -> str:
        spec = METRICS[metric]
        arrow = {"higher": "↑ better", "lower": "↓ better", None: "diagnostic"}
        return f"{spec.label} ({arrow[spec.direction]})"

    fig.update_layout(
        xaxis_title=_axis(x_metric),
        yaxis_title=_axis(y_metric),
        height=440,
        margin=dict(l=50, r=20, t=20, b=50),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"{int(sub['on_front'].sum())} of {len(sub)} runs are Pareto-optimal "
        f"for ({METRICS[x_metric].label}, {METRICS[y_metric].label}). "
        "Diagnostic metrics are treated as higher-is-better for dominance."
    )
