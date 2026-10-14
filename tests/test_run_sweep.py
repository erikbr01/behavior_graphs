"""The hydra sweep entry point: config composition + end-to-end run."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.pipeline.sweep import expand_spec, validate_spec

REPO = Path(__file__).resolve().parents[1]

spec_loader = importlib.util.spec_from_file_location("run_sweep", REPO / "scripts" / "run_sweep.py")
run_sweep = importlib.util.module_from_spec(spec_loader)
sys.modules[spec_loader.name] = run_sweep
spec_loader.loader.exec_module(run_sweep)

# High-dim like real policy embeddings, so the default grids (e.g. PCA 32)
# are valid on the synthetic data.
DIM = 40
LAYER = "pi05"


@pytest.fixture(scope="module")
def rollouts(tmp_path_factory):
    rng = np.random.default_rng(3)
    regimes = rng.normal(0.0, 1.0, size=(3, DIM)) * 6.0
    lengths = [int(n) for n in rng.integers(30, 61, size=10)]
    successes = [i % 2 == 0 for i in range(10)]
    chunks = []
    for n in lengths:
        thirds = [n // 3, n // 3, n - 2 * (n // 3)]
        for regime, m in zip(regimes, thirds):
            chunks.append(regime + rng.normal(0.0, 0.5, size=(m, DIM)))
    path = tmp_path_factory.mktemp("rollouts")
    return write_rollout_dir(
        path, lengths, successes, {LAYER: np.concatenate(chunks).astype(np.float32)}
    )


class TestConfig:
    def test_quick_grid_selectable(self):
        cfg = run_sweep.compose_sweep_config(["grid=quick", "datasets={test: /x}"])
        spec = OmegaConf.to_container(cfg.grid, resolve=True)
        validate_spec(spec)
        assert expand_spec(spec)
        assert not any("pipeline/reduction@reduction=sae" in j for j in expand_spec(spec))

    def test_datasets_override(self):
        cfg = run_sweep.compose_sweep_config(["grid=quick", "datasets={lift: /a, square: /b}"])
        assert run_sweep._datasets(cfg) == {"lift": "/a", "square": "/b"}

    def test_no_data_raises(self):
        cfg = run_sweep.compose_sweep_config(["grid=quick"])
        with pytest.raises(ValueError, match="no data given"):
            run_sweep._datasets(cfg)

    @pytest.mark.parametrize("group", ["all", "droid", "visualize", "robomimic"])
    def test_every_shipped_data_group_resolves(self, group, monkeypatch):
        """A group-level key is not a dataset, and must not break the entrypoint.

        The apps read a top-level ``cameras:`` block out of these same files; when
        only their reader learned to skip it, every ``data=`` group here failed
        with "dataset(s) with no rollout dir: cameras" before doing any work.
        """
        monkeypatch.setenv("DATA_PATH", "/x")  # the group files interpolate it
        cfg = run_sweep.compose_sweep_config(["grid=quick", f"data={group}"])

        datasets = run_sweep._datasets(cfg)

        assert datasets, f"data={group} resolved to nothing"
        assert not {"cameras", "defaults"} & set(datasets)
        assert all(v and v != "None" for v in datasets.values())

    def test_detect_layers(self, rollouts):
        assert run_sweep.detect_layers([str(rollouts.path)]) == [LAYER]
        # nothing has embeddings, so there is no layer to override with
        assert run_sweep.detect_layers(["/nonexistent"]) == [None]


def test_dry_run(rollouts, capsys):
    cfg = run_sweep.compose_sweep_config(
        [
            "grid=quick",
            f"datasets={{test: {rollouts.path}}}",
            "dry_run=true",
        ]
    )
    run_sweep.run_sweep(cfg)
    out = capsys.readouterr().out
    assert "pipeline/clustering@clustering=kmeans" in out  # job list printed
    assert "jobs x 1 dataset" in out


def test_end_to_end(rollouts, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)  # cache lands in tmp, not the repo
    cfg = run_sweep.compose_sweep_config(
        [
            "grid=quick",
            f"datasets={{test: {rollouts.path}}}",
            f"sweep_dir={tmp_path / 'sweep'}",
            "limit=4",
        ]
    )
    # Nothing here reads the 2-D coords, and the quick grid's pca rows are 32
    # wide, so the pipeline would otherwise fit the default UMAP for them --
    # ~20s of numba JIT against ~1s for everything this test asserts on. Set
    # after composing: hydra's override grammar will not take a `group@key=`
    # override nested inside a list value.
    cfg.extra_overrides = ["viz/viz_reduction@viz_reduction=pca_2d"]
    sweep_dir = run_sweep.run_sweep(cfg)
    out = capsys.readouterr().out
    assert "Recommended hyperparameters per K" in out

    # The auto-detected layer is baked into the saved spec.
    import yaml

    saved = yaml.safe_load((sweep_dir / "spec.yaml").read_text())
    assert saved["base_overrides"] == [f"features.layer={LAYER}"]

    index = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(index) == 4
    assert index["silhouette"].notna().all()
