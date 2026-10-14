"""Run a pipeline sweep — hydra entry point.

Pick a grid with the ``grid`` config group and point at data; everything is
a plain hydra override (see ``configs/sweep.yaml`` and
``configs/grid/{quick,full}.yaml``)::

    # run on a named dataset group (configs/data/*.yaml):
    python scripts/run_sweep.py data=robomimic
    python scripts/run_sweep.py data=all grid=full layers=[bottleneck_plan_t0]

    # or specify datasets inline:
    python scripts/run_sweep.py 'datasets={lift: /data/rollouts/lift_mh}'
    python scripts/run_sweep.py \\
        'datasets={lift: /data/rollouts/lift, square: /data/rollouts/square}'

    # sweep multiple embedding layers (runs the full grid once per layer;
    # each dataset joins only the passes whose layer it carries):
    python scripts/run_sweep.py data=all grid=full \\
        'layers=[bottleneck_plan_t0,cond_proj]'

    # inspect the job list without running anything:
    python scripts/run_sweep.py data=mimicgen dry_run=true

The chosen grid config (stages + selection rule) is saved to
``<sweep_dir>/spec.yaml``; the run prints the model-selection summary
(best hyperparameters per K + the most sensitive knobs) at the end.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from behavior_graphs.pipeline.dataset_resolution import (
    available_layers,
    detect_layers,
    selects_layerless_features,
)
from behavior_graphs.pipeline.dataset_resolution import resolve_datasets as _datasets
from behavior_graphs.pipeline.run import CONFIG_PATH, compose_pipeline_config
from behavior_graphs.pipeline.sweep import regenerate_reports
from behavior_graphs.pipeline.sweep import run_sweep as _run_sweep

log = logging.getLogger("run_sweep")


def print_summary(sweep_dir: Path) -> None:
    """Best-per-K and top sensitivity findings, straight to the terminal."""
    import pandas as pd

    best_path = sweep_dir / "best_per_k.csv"
    if best_path.exists():
        best = pd.read_csv(best_path)
        cols = [
            c
            for c in (
                "dataset",
                "k",
                "score",
                "feasible",
                "windowing.width",
                "windowing.stride",
                "reduction",
                "clustering",
                "silhouette",
            )
            if c in best.columns
        ]
        print("\n── Recommended hyperparameters per K (pick the complexity you want) " + "─" * 20)
        print(best[cols].to_string(index=False))

    sens_path = sweep_dir / "sensitivity_silhouette.csv"
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        cols = [
            c
            for c in (
                "param",
                "eta_squared",
                "mean_spread",
                "best_value",
                "wins",
            )
            if c in sens.columns
        ]
        print("\n── Which knobs matter for silhouette (eta² = variance explained) " + "─" * 16)
        print(sens[cols].head(8).to_string(index=False))

    print(
        f"\nFull reports in {sweep_dir}/ (sweep_index.csv, best_per_k.csv, "
        f"sensitivity_*.csv).\nInteractive: streamlit run apps/demo/Home.py "
        f"→ Graph Demo → Model selection."
    )


def _sweep_name(rollouts: dict[str, str]) -> str:
    """Stable name derived from rollout scope for use as the sweep dir basename."""
    return "_".join(sorted(rollouts))


def run_sweep(cfg: DictConfig) -> Path:
    """Build the spec from the resolved config and run the sweep."""
    rollouts = _datasets(cfg)
    spec: dict = OmegaConf.to_container(cfg.grid, resolve=True)  # stages + selection
    spec.setdefault("base_overrides", [])
    spec["datasets"] = rollouts

    # `layers` (list) beats `layer` (scalar); neither means auto-detect.
    layers_cfg = cfg.get("layers")
    extra = list(cfg.get("extra_overrides") or [])
    if layers_cfg:
        requested_layers: list[str | None] = list(layers_cfg)
    elif cfg.get("layer"):
        requested_layers = [cfg.layer]
    elif selects_layerless_features(list(spec["base_overrides"]) + extra, spec.get("stages")):
        requested_layers = [None]  # nothing to detect: this features stage has no layer
    else:
        requested_layers = detect_layers(list(rollouts.values()))

    if cfg.get("sweep_dir"):
        sweep_dir = Path(cfg.sweep_dir)
    else:
        scope = _sweep_name(rollouts)
        sweep_dir = Path(f"multirun/{scope}_{time.strftime('%Y-%m-%d')}")
    limit = cfg.get("limit")
    dry_run = bool(cfg.get("dry_run"))

    # Probe upfront so only (rollout, layer) pairs that exist get queued.
    ds_layers: dict[str, set[str]] = {}
    if any(layer is not None for layer in requested_layers):
        ds_layers = available_layers(list(rollouts.values()))

    for layer in requested_layers:
        if layer is None:
            # No layer override — run all rollouts together as-is.
            layer_datasets = rollouts
        else:
            # Filter to rollouts that have this layer; warn about the rest.
            layer_datasets = {
                name: path for name, path in rollouts.items() if layer in ds_layers.get(path, set())
            }
            skipped = {name: path for name, path in rollouts.items() if name not in layer_datasets}
            if skipped:
                log.warning(
                    "layer '%s' not available in %d dataset(s), skipping: %s",
                    layer,
                    len(skipped),
                    ", ".join(skipped) or list(skipped.values()),
                )
            if not layer_datasets:
                log.warning("layer '%s' not available in any dataset — skipping entirely", layer)
                continue

        log.info("sweeping layer: %s across %d dataset(s)", layer, len(layer_datasets))
        layer_spec = {**spec, "base_overrides": list(spec["base_overrides"])}
        if layer:
            layer_spec["base_overrides"].append(f"features.layer={layer}")
        layer_spec["datasets"] = layer_datasets
        _run_sweep(
            layer_spec,
            sweep_dir=sweep_dir,
            extra_overrides=extra,
            limit=limit,
            dry_run=dry_run,
            allow_stale_data=bool(cfg.get("allow_stale_data")),
            write_reports=False,  # cover the whole dir; regenerated once below
        )

    if not dry_run:
        regenerate_reports(sweep_dir, spec)
        print_summary(sweep_dir)
    return sweep_dir


def compose_sweep_config(overrides: Sequence[str]) -> DictConfig:
    """Compose ``configs/sweep.yaml`` outside ``@hydra.main`` (for tests)."""
    return compose_pipeline_config(overrides, config_name="sweep")


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="sweep")
def main(cfg: DictConfig) -> None:
    run_sweep(cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )
    sys.exit(main())  # type: ignore[func-returns-value]
