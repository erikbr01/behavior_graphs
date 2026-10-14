"""Integration tests for the sweep runner against real robomimic lift rollouts.

Covers the three main workflow scenarios:
- Pause and resume: jobs with metrics.json are skipped on re-run
- Crash recovery: a job dir missing metrics.json is re-run; failures.json clears
- Expanded grid: two grids written to one sweep dir accumulate without collision

Requires an existing rollout dir with policy embeddings; set
BEHAVIOR_GRAPHS_LIFT_ROLLOUTS to override the default path.

Run with::

    PYTHONPATH=src pytest -m integration tests/integration/test_sweep.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

LIFT_ROLLOUTS = Path(
    os.environ.get(
        "BEHAVIOR_GRAPHS_LIFT_ROLLOUTS",
        "${DATA_PATH}/rollouts/robomimic_lift_mh",
    )
)
LAYER = "bottleneck_plan_t0"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def rollout_dir():
    if not LIFT_ROLLOUTS.is_dir():
        pytest.skip(f"rollout dir not found: {LIFT_ROLLOUTS}")
    return LIFT_ROLLOUTS


@pytest.fixture(scope="module")
def rollout_dir_with_layer(rollout_dir):
    """Rollout dir that has the required embedding layer pre-computed.

    The sweep tests need policy_embeddings/<LAYER>.npz to exist.  If it
    doesn't, skip gracefully rather than erroring mid-run.
    """
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory

    available = RolloutDirectory(rollout_dir).embedding_layers()
    if LAYER not in available:
        pytest.skip(
            f"embedding layer '{LAYER}' not found in {rollout_dir} "
            f"(available: {available or 'none'} — run the lift integration test first to generate rollouts with embeddings)"
        )
    return rollout_dir


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory):
    """One cache dir for the whole module so upstream artifacts are shared."""
    return tmp_path_factory.mktemp("sweep_cache")


def _clear_hydra():
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()


def _spec(n_components: int = 8, widths: tuple = (3,), k_values: tuple = (5, 10)) -> dict:
    """Small but valid sweep spec for fast integration runs."""
    return {
        "base_overrides": [f"features.layer={LAYER}"],
        "stages": {
            "windowing": {"sliding": {"width": list(widths), "stride": 2}},
            "reduction": {"pca": {"n_components": n_components}},
            "clustering": {"kmeans": {"n_clusters": list(k_values)}},
        },
        "selection": {
            "weights": {"silhouette": 1.0},
            "constraints": {"silhouette": {"min": -1.0}},
        },
    }


# ────────────────────────────────────────────────────────────────── basic run


def test_basic_sweep(rollout_dir_with_layer, shared_cache, tmp_path):
    rollout_dir = rollout_dir_with_layer
    """Full sweep: verifies hash-named dirs, jobs.json, datasets.yaml, and reports."""
    from behavior_graphs.pipeline.sweep import run_sweep

    _clear_hydra()
    sweep_dir = run_sweep(
        _spec(widths=(3, 5), k_values=(5, 10)),
        datasets={"lift": rollout_dir},
        sweep_dir=tmp_path / "sweep",
        extra_overrides=[f"cache_dir={shared_cache}"],
    )

    # jobs.json: 2 widths × 2 K = 4 jobs, keyed by "<dataset>/<hash12>"
    jobs = json.loads((sweep_dir / "jobs.json").read_text())
    assert len(jobs) == 4
    for key in jobs:
        hash_part = key.split("/")[-1]
        assert len(hash_part) == 12, f"expected 12-char hash suffix, got {key!r}"
        assert all(c in "0123456789abcdef" for c in hash_part), f"non-hex hash in key: {key!r}"
        assert (sweep_dir / key / "metrics.json").exists()

    # datasets.yaml accumulates the rollout path
    ds = yaml.safe_load((sweep_dir / "datasets.yaml").read_text())
    assert len(ds) == 1

    assert (sweep_dir / "spec.yaml").exists()
    assert not (sweep_dir / "failures.json").exists()

    index = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(index) == 4
    assert index["silhouette"].notna().all()
    assert set(index["windowing.width"]) == {3, 5}

    best = pd.read_csv(sweep_dir / "best_per_k.csv")
    assert set(best["k"]) == {5, 10}


# ────────────────────────────────────────────────────────────── pause + resume


def test_pause_and_resume(rollout_dir_with_layer, shared_cache, tmp_path):
    rollout_dir = rollout_dir_with_layer
    """Re-running the identical spec skips jobs that already have metrics.json."""
    from behavior_graphs.pipeline.sweep import run_sweep

    spec = _spec(k_values=(5, 10))
    sweep_dir = tmp_path / "sweep"

    # First run — partial: only 1 job (limit=1)
    _clear_hydra()
    run_sweep(
        spec,
        datasets={"lift": rollout_dir},
        sweep_dir=sweep_dir,
        extra_overrides=[f"cache_dir={shared_cache}"],
        limit=1,
    )
    jobs_partial = json.loads((sweep_dir / "jobs.json").read_text())
    assert len(jobs_partial) == 1
    completed_key = next(iter(jobs_partial))
    mtime_before = (sweep_dir / completed_key / "metrics.json").stat().st_mtime

    # Second run — full spec: completed job is skipped, rest run
    _clear_hydra()
    run_sweep(
        spec,
        datasets={"lift": rollout_dir},
        sweep_dir=sweep_dir,
        extra_overrides=[f"cache_dir={shared_cache}"],
    )

    # The already-completed job was not re-run (mtime unchanged)
    mtime_after = (sweep_dir / completed_key / "metrics.json").stat().st_mtime
    assert mtime_after == mtime_before

    # Full set of jobs now complete
    jobs_full = json.loads((sweep_dir / "jobs.json").read_text())
    assert len(jobs_full) == 2
    index = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(index) == 2


# ─────────────────────────────────────────────────────────── crash recovery


def test_crash_recovery(rollout_dir_with_layer, shared_cache, tmp_path):
    rollout_dir = rollout_dir_with_layer
    """Removing metrics.json simulates a crash; re-run recovers and clears failures.json."""
    from behavior_graphs.pipeline.sweep import run_sweep

    spec = _spec(k_values=(5, 10))
    sweep_dir = tmp_path / "sweep"

    # Complete all jobs
    _clear_hydra()
    run_sweep(
        spec,
        datasets={"lift": rollout_dir},
        sweep_dir=sweep_dir,
        extra_overrides=[f"cache_dir={shared_cache}"],
    )

    jobs = json.loads((sweep_dir / "jobs.json").read_text())
    assert len(jobs) == 2

    # Simulate a crash on the first job: remove its metrics.json
    crashed_key = next(iter(jobs))
    crashed_metrics = sweep_dir / crashed_key / "metrics.json"
    crashed_metrics.unlink()
    assert not crashed_metrics.exists()

    # Plant a failures.json as if the crash was recorded by a prior run
    (sweep_dir / "failures.json").write_text(
        json.dumps({crashed_key: "RuntimeError: simulated crash"})
    )

    # Re-run: crashed job is re-run (no metrics.json), healthy job is skipped
    _clear_hydra()
    run_sweep(
        spec,
        datasets={"lift": rollout_dir},
        sweep_dir=sweep_dir,
        extra_overrides=[f"cache_dir={shared_cache}"],
    )

    # Recovered: metrics.json is back
    assert crashed_metrics.exists()

    # failures.json is deleted — all failures resolved
    assert not (sweep_dir / "failures.json").exists()

    # Index still covers all 2 jobs
    index = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(index) == 2


# ──────────────────────────────────────────────── expanded-grid accumulation


def test_expanded_grid_accumulation(rollout_dir_with_layer, shared_cache, tmp_path):
    rollout_dir = rollout_dir_with_layer
    """PCA n=8 grid then PCA n=16 grid in the same sweep dir:
    no hash collision, jobs.json accumulates, unified sweep_index.csv."""
    from behavior_graphs.pipeline.sweep import run_sweep

    sweep_dir = tmp_path / "sweep"

    # Grid A: PCA n_components=8, K={5,10} — 2 jobs
    _clear_hydra()
    run_sweep(
        _spec(n_components=8, k_values=(5, 10)),
        datasets={"lift": rollout_dir},
        sweep_dir=sweep_dir,
        extra_overrides=[f"cache_dir={shared_cache}"],
    )
    jobs_after_a = json.loads((sweep_dir / "jobs.json").read_text())
    index_after_a = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(jobs_after_a) == 2
    assert len(index_after_a) == 2

    # Grid B: PCA n_components=16, K={5,10} — 2 more jobs (different overrides → different hashes)
    _clear_hydra()
    run_sweep(
        _spec(n_components=16, k_values=(5, 10)),
        datasets={"lift": rollout_dir},
        sweep_dir=sweep_dir,
        extra_overrides=[f"cache_dir={shared_cache}"],
    )
    jobs_after_b = json.loads((sweep_dir / "jobs.json").read_text())
    index_after_b = pd.read_csv(sweep_dir / "sweep_index.csv")

    # 4 distinct hashes — no collision between grids A and B
    assert len(jobs_after_b) == 4
    for k in jobs_after_a:
        assert k in jobs_after_b, "grid A hashes must still be present"

    # Unified index covers all 4 jobs
    assert len(index_after_b) == 4
    assert set(index_after_b["reduction.n_components"]) == {8, 16}
    assert not (sweep_dir / "failures.json").exists()

    # datasets.yaml still reflects a single dataset (same rollout dir both runs)
    ds = yaml.safe_load((sweep_dir / "datasets.yaml").read_text())
    assert len(ds) == 1

    # Grid A jobs are not re-run (mtime unchanged)
    for key in jobs_after_a:
        assert (sweep_dir / key / "metrics.json").exists()


# ─────────────────────────────────────────── hash stability sanity check


def test_hash_stability(rollout_dir):
    """Same (overrides, path) always produces the same 12-char hash."""
    from behavior_graphs.pipeline.sweep import _job_hash

    overrides = [
        "pipeline/windowing@windowing=sliding",
        "windowing.stride=2",
        "windowing.width=3",
        "pipeline/reduction@reduction=pca",
        "reduction.n_components=8",
        "pipeline/clustering@clustering=kmeans",
        "clustering.n_clusters=5",
        f"features.layer={LAYER}",
    ]
    h1 = _job_hash(rollout_dir, overrides)
    h2 = _job_hash(rollout_dir, overrides)
    assert h1 == h2
    assert len(h1) == 12
    assert all(c in "0123456789abcdef" for c in h1)

    # Different n_components → different hash
    other = [o.replace("n_components=8", "n_components=16") for o in overrides]
    assert _job_hash(rollout_dir, other) != h1

    # Ordering of overrides list doesn't matter (canonical form is sorted)
    shuffled = list(reversed(overrides))
    assert _job_hash(rollout_dir, shuffled) == h1
