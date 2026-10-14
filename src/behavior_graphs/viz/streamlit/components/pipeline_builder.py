"""Interactive pipeline builder: every stage hyperparameter as a widget.

Renders one control per constructor argument of every stage option (read
straight from the config group YAMLs, so new stages/params appear
automatically), then runs the pipeline with the chosen configuration.
Stage outputs are reused from the content-addressed cache, so re-running
with one knob changed only computes the stages downstream of it — graph
creation is cheap, so iterating from the GUI is interactive.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st
import yaml

from behavior_graphs.core.stages import STAGE_GROUPS
from behavior_graphs.pipeline.sweep import format_override_value

#: Params with a known closed set of values get a selectbox.
ENUM_CHOICES: dict[str, list[str]] = {
    "windowing.aggregation": ["mean", "sum", "max"],
    "graph.smoothing": ["none", "median", "sticky"],
    "reduction.normalize": ["none", "standard", "minmax", "robust"],
    "clustering.covariance_type": ["full", "tied", "diag", "spherical"],
}

#: Implementation detail, not science — hidden from the form.
_HIDDEN_PARAMS = {"_target_", "device"}


@st.cache_data(show_spinner=False)
def stage_options(group: str) -> dict[str, dict[str, Any]]:
    """{option_name: flat param dict} for one config group."""
    from behavior_graphs.pipeline.run import CONFIG_PATH

    out: dict[str, dict[str, Any]] = {}
    for path in sorted((Path(CONFIG_PATH) / "pipeline" / group).glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        if "_target_" in cfg:
            out[path.stem] = cfg
    return out


def _param_widget(group: str, option: str, name: str, default: Any, key: str) -> Any:
    """One widget per hyperparameter, typed from its config default."""
    label = name
    full = f"{group}.{name}"
    if full in ENUM_CHOICES:
        choices = ENUM_CHOICES[full]
        idx = choices.index(default) if default in choices else 0
        return st.selectbox(label, choices, index=idx, key=key)
    if isinstance(default, bool):
        return st.checkbox(label, value=default, key=key)
    if isinstance(default, int):
        return int(st.number_input(label, value=int(default), step=1, key=key))
    if isinstance(default, float):
        return float(
            st.number_input(label, value=float(default), step=None, format="%.6g", key=key)
        )
    # Strings and nulls: free text; empty means "keep the config default".
    raw = st.text_input(
        label, value="" if default is None else str(default), key=key, placeholder="null"
    )
    return raw.strip() or None


def render_stage_controls(
    group: str,
    key_prefix: str,
    current: dict[str, Any] | None = None,
    layer_choices: list[str] | None = None,
) -> tuple[str, list[str], dict[str, Any]]:
    """Option picker + per-param widgets for one stage group.

    Returns (option name, hydra overrides, raw param values).
    """
    options = stage_options(group)
    names = list(options)
    cur_target = str((current or {}).get("_target_", ""))
    default_idx = 0
    for i, n in enumerate(names):
        if options[n].get("_target_") == cur_target:
            default_idx = i
            break
    option = st.selectbox(group, names, index=default_idx, key=f"{key_prefix}_{group}")
    params = {
        k: v
        for k, v in options[option].items()
        if k not in _HIDDEN_PARAMS
        and not isinstance(v, dict | list)  # composite cfgs (concat) not editable here
    }
    # Pre-fill from the currently loaded run when it uses the same option.
    if current and options[option].get("_target_") == cur_target:
        params = {k: current.get(k, v) for k, v in params.items()}

    overrides = [f"{group}={option}"]
    values: dict[str, Any] = {}
    for name, default in params.items():
        key = f"{key_prefix}_{group}_{option}_{name}"
        if group == "features" and name == "layer" and layer_choices:
            idx = layer_choices.index(default) if default in layer_choices else 0
            value = st.selectbox(name, layer_choices, index=idx, key=key)
        else:
            value = _param_widget(group, option, name, default, key)
        values[name] = value
        if value is None:
            continue  # keep the option's yaml default (typically null)
        overrides.append(f"{group}.{name}={format_override_value(value)}")
    return option, overrides, values


def parse_k_list(raw: str) -> list[int]:
    """K values from free text: ``4, 8, 12`` and/or ranges ``4:20:2``."""
    out: list[int] = []
    for tok in raw.replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            parts = [int(p) for p in tok.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"range must be start:stop[:step], got {tok!r}")
            step = parts[2] if len(parts) == 3 else 1
            out.extend(range(parts[0], parts[1] + 1, step))
        else:
            out.append(int(tok))
    return sorted(dict.fromkeys(out))


def run_pipeline_from_gui(
    rollout_dir: Path,
    overrides: list[str],
    run_root: Path = Path("runs/gui"),
    name: str | None = None,
) -> Path:
    """Compose a config from overrides and run the (cached) pipeline."""
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory
    from behavior_graphs.pipeline.run import compose_pipeline_config, run_pipeline

    cfg = compose_pipeline_config([f"rollout_dir={rollout_dir}"] + overrides)
    run_dir = run_root / (name or time.strftime("%Y-%m-%d_%H-%M-%S"))
    run_pipeline(cfg, RolloutDirectory(rollout_dir), run_dir)
    return run_dir


def render_pipeline_builder(
    default_rollout_dir: Path | None,
    run_select_key: str = "demo_run",
    key_prefix: str = "pb",
    current_cfg: dict[str, Any] | None = None,
) -> None:
    """The full builder: data source, all stage hyperparameters, run button.

    On success the page's run selector is pointed at the newest run and the
    app reruns, so the new graph is immediately on screen.
    """
    rollout_dir = st.text_input(
        "Rollout directory",
        value=str(default_rollout_dir or ""),
        key=f"{key_prefix}_rollout_dir",
        help="Collected rollouts in the canonical layout.",
    ).strip()

    layer_choices: list[str] | None = None
    if rollout_dir and Path(rollout_dir).is_dir():
        try:
            from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

            layer_choices = TrajectoryDirectory(Path(rollout_dir)).embedding_layers()
        except Exception:
            layer_choices = None

    current_cfg = current_cfg or {}
    all_overrides: list[str] = []
    clustering_values: dict[str, Any] = {}
    cols = st.columns(len(STAGE_GROUPS))
    for col, group in zip(cols, STAGE_GROUPS):
        with col:
            _, overrides, values = render_stage_controls(
                group,
                key_prefix,
                current=current_cfg.get(group),
                layer_choices=layer_choices,
            )
            if group == "clustering":
                clustering_values = values
            all_overrides.extend(overrides)

    # K is the user's complexity choice — allow a list so one click produces
    # one (cached, cheap) run per K, ready for the per-K comparison table.
    k_param = next((p for p in ("n_clusters", "n_components") if p in clustering_values), None)
    k_raw = ""
    if k_param is not None:
        k_raw = st.text_input(
            f"Also run for multiple K ({k_param}, comma-separated; blank = just the value above)",
            value="",
            key=f"{key_prefix}_k_list",
            placeholder="e.g. 4, 8, 12 or 4:20:2",
        )

    if not st.button("▶ Run pipeline", type="primary", key=f"{key_prefix}_go"):
        return
    if not rollout_dir or not Path(rollout_dir).is_dir():
        st.error("Rollout directory does not exist.")
        return

    try:
        ks = parse_k_list(k_raw) if k_raw.strip() else []
    except ValueError:
        st.error(f"Could not parse K list: {k_raw!r}")
        return

    jobs: list[list[str]] = []
    if ks and k_param is not None:
        base = [o for o in all_overrides if not o.startswith(f"clustering.{k_param}=")]
        for k in ks:
            jobs.append(base + [f"clustering.{k_param}={k}"])
    else:
        jobs.append(all_overrides)

    last_run: Path | None = None
    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    progress = st.progress(0.0)
    for i, job in enumerate(jobs):
        name = stamp if len(jobs) == 1 else f"{stamp}_{i:02d}"
        with st.spinner(f"Running pipeline ({i + 1}/{len(jobs)})…"):
            try:
                last_run = run_pipeline_from_gui(Path(rollout_dir), job, name=name)
            except Exception as e:
                st.error(f"Pipeline failed: {type(e).__name__}: {e}")
                return
        progress.progress((i + 1) / len(jobs))

    st.success(f"Done — {len(jobs)} run(s); latest: `{last_run}`")
    # Point the page's run selector at the new run and refresh.
    st.session_state[run_select_key] = str(last_run)
    st.rerun()
