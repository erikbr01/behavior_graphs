"""Sweep runner: spec expansion, validation, and a tiny end-to-end ablation."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.pipeline.sweep import (
    expand_spec,
    expand_values,
    resolve_datasets,
    run_sweep,
    validate_spec,
)

DIM = 8
LAYER = "pi05"


def _gen(target, *args):
    return {"_target_": f"behavior_graphs.pipeline.grid.{target}", "_args_": list(args)}


class TestExpandValues:
    def test_scalar_and_list(self):
        assert expand_values(5) == [5]
        assert expand_values([1, 2]) == [1, 2]
        assert expand_values("mean") == ["mean"]

    def test_target_generators(self):
        assert expand_values(_gen("irange", 4, 20, 4)) == [4, 8, 12, 16, 20]
        assert expand_values(_gen("linspace", 0.0, 0.5, 3)) == [0.0, 0.25, 0.5]
        assert expand_values(_gen("logspace", 1e-4, 1e-1, 4)) == [
            0.0001,
            0.001,
            0.01,
            0.1,
        ]

    def test_mapping_without_target_rejected(self):
        with pytest.raises(ValueError, match="_target_ generator"):
            expand_values({"range": [1, 2]})


class TestValidateSpec:
    def test_valid_spec_passes(self):
        validate_spec(
            {
                "stages": {
                    "clustering": {"kmeans": {"n_clusters": [3]}},
                    "windowing": {"sliding": {"width": _gen("irange", 1, 5)}},
                }
            }
        )

    def test_unknown_option_lists_available(self):
        with pytest.raises(ValueError, match="no option 'kmeens'.*kmeans"):
            validate_spec({"stages": {"clustering": {"kmeens": {}}}})

    def test_unknown_param_lists_valid(self):
        with pytest.raises(ValueError, match=r"unknown parameter\(s\) \['n_cluster'\].*n_clusters"):
            validate_spec({"stages": {"clustering": {"kmeans": {"n_cluster": [3]}}}})

    def test_unknown_selection_metric(self):
        with pytest.raises(ValueError, match="selection.weights: unknown metric"):
            validate_spec({"selection": {"weights": {"silhuette": 1.0}}})

    def test_unknown_group(self):
        with pytest.raises(ValueError, match="unknown stage group"):
            validate_spec({"stages": {"clusterer": {}}})


class TestResolveDatasets:
    def test_kwarg_overrides_spec(self, tmp_path):
        out = resolve_datasets({"datasets": {"a": "/x"}}, datasets={"b": tmp_path})
        assert out == {"b": tmp_path}

    def test_spec_datasets(self):
        out = resolve_datasets({"datasets": {"a": "/x", "b": "/y"}})
        assert set(out) == {"a", "b"}

    def test_missing_data_raises(self):
        with pytest.raises(ValueError, match="no data given"):
            resolve_datasets({})


class TestExpandSpec:
    def test_param_grids_stay_inside_their_option(self):
        jobs = expand_spec(
            {
                "stages": {
                    "clustering": {
                        "kmeans": {"n_clusters": [3, 5]},
                        "hdbscan": {"min_cluster_size": [10]},
                    },
                }
            }
        )
        assert jobs == [
            ["pipeline/clustering@clustering=kmeans", "clustering.n_clusters=3"],
            ["pipeline/clustering@clustering=kmeans", "clustering.n_clusters=5"],
            ["pipeline/clustering@clustering=hdbscan", "clustering.min_cluster_size=10"],
        ]

    def test_cross_product_upstream_first(self):
        jobs = expand_spec(
            {
                "stages": {
                    "windowing": {"sliding": {"width": [3, 5]}},
                    "clustering": {"kmeans": {"n_clusters": [3, 5]}},
                }
            }
        )
        assert len(jobs) == 4
        # Downstream (clustering) varies fastest -> upstream cache stays hot.
        assert jobs[0][:2] == ["pipeline/windowing@windowing=sliding", "windowing.width=3"]
        assert jobs[1][:2] == ["pipeline/windowing@windowing=sliding", "windowing.width=3"]
        assert jobs[1][2:] == ["pipeline/clustering@clustering=kmeans", "clustering.n_clusters=5"]

    def test_base_overrides_and_scalars(self):
        jobs = expand_spec(
            {
                "base_overrides": ["features.layer=pi05"],
                "stages": {
                    "graph": {
                        "transition": {
                            "smoothing": ["none"],
                            "degree_one_prune": True,
                        }
                    }
                },
            }
        )
        assert jobs == [
            [
                "features.layer=pi05",
                "pipeline/graph@graph=transition",
                "graph.degree_one_prune=true",
                "graph.smoothing=none",
            ]
        ]

    def test_unknown_group_rejected(self):
        with pytest.raises(ValueError, match="unknown stage group"):
            expand_spec({"stages": {"clusterer": {}}})


@pytest.fixture(scope="module")
def rollouts(tmp_path_factory):
    rng = np.random.default_rng(1)
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


def test_end_to_end_sweep(rollouts, tmp_path):
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    spec = {
        "datasets": {"test": str(rollouts.path)},
        "stages": {
            "windowing": {"sliding": {"width": [3, 5], "stride": [2]}},
            "reduction": {"pca": {"n_components": [2]}},
            "clustering": {"kmeans": {"n_clusters": _gen("irange", 3, 4)}},
        },
        "selection": {
            "weights": {"silhouette": 1.0},
            "constraints": {"silhouette": {"min": -1.0}},
        },
    }

    sweep_dir = run_sweep(
        spec,
        sweep_dir=tmp_path / "sweep",
        extra_overrides=[f"cache_dir={tmp_path / 'cache'}"],
    )

    assert (sweep_dir / "spec.yaml").exists()
    jobs = json.loads((sweep_dir / "jobs.json").read_text())
    assert len(jobs) == 4
    assert not (sweep_dir / "failures.json").exists()

    index = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(index) == 4
    assert index["silhouette"].notna().all()
    assert index["score"].notna().all()
    assert set(index["windowing.width"]) == {3, 5}

    best = pd.read_csv(sweep_dir / "best_per_k.csv")
    assert sorted(best["k"]) == [3, 4]

    # Cache reuse: the second width-3 job (different K) must reuse upstream.
    # Find it via jobs.json rather than a sequential index.
    w3k4_key = next(
        k
        for k, ov in jobs.items()
        if any("windowing.width=3" in o for o in ov)
        and any("clustering.n_clusters=4" in o for o in ov)
    )
    mx1 = json.loads((sweep_dir / w3k4_key / "metrics.json").read_text())
    assert mx1["cache_hits"] == {
        "features": True,
        "windows": True,
        "reduced": True,
        "labels": False,
    }

    # Re-running the sweep skips completed jobs and keeps reports intact.
    run_sweep(
        spec,
        sweep_dir=tmp_path / "sweep",
        extra_overrides=[f"cache_dir={tmp_path / 'cache'}"],
    )
    assert len(pd.read_csv(sweep_dir / "sweep_index.csv")) == 4


def test_multi_dataset_sweep(rollouts, tmp_path):
    """`datasets:` sweeps tasks too: per-dataset job dirs, scores, best-per-K."""
    rng = np.random.default_rng(7)
    lengths = [40] * 6
    emb = rng.normal(size=(sum(lengths), DIM)).astype(np.float32)
    other = write_rollout_dir(tmp_path / "other_rollouts", lengths, [True, False] * 3, {LAYER: emb})

    spec = {
        "datasets": {
            "taskA": str(rollouts.path),
            "taskB": str(other.path),
        },
        "stages": {
            "reduction": {"pca": {"n_components": 2}},
            "clustering": {"kmeans": {"n_clusters": _gen("irange", 3, 4)}},
        },
        "selection": {"weights": {"silhouette": 1.0}, "constraints": {"silhouette": {"min": -1.0}}},
    }

    sweep_dir = run_sweep(
        spec,
        sweep_dir=tmp_path / "sweep_multi",
        extra_overrides=[f"cache_dir={tmp_path / 'cache'}"],
    )

    # Each dataset gets its own hash-named subdirs.
    taskA_done = list((sweep_dir / "taskA").glob("*/metrics.json"))
    taskB_done = list((sweep_dir / "taskB").glob("*/metrics.json"))
    assert len(taskA_done) == 2
    assert len(taskB_done) == 2

    index = pd.read_csv(sweep_dir / "sweep_index.csv")
    assert len(index) == 4
    assert set(index["dataset"]) == {"taskA", "taskB"}
    # Scores normalized within each dataset: every dataset has its own best.
    assert index.groupby("dataset")["score"].max().tolist() == [1.0, 1.0]

    best = pd.read_csv(sweep_dir / "best_per_k.csv")
    assert sorted(best["k"]) == [3, 3, 4, 4]  # one block per dataset


def _merge_worker(args):
    """Top-level so multiprocessing can pickle it."""
    from behavior_graphs.pipeline.sweep import _merge_jobs

    jobs_path, start, count = args
    _merge_jobs(
        Path(jobs_path), {f"ds/{i:05d}": [f"override={i}"] for i in range(start, start + count)}
    )


def test_concurrent_sweeps_do_not_lose_each_others_job_keys(tmp_path):
    """One sweep dir, several sweep processes -- how a multi-dataset sweep is run.

    The atomic write alone prevented a torn jobs.json but not a lost update:
    each process read, merged and replaced the file, so later writers dropped
    earlier ones' keys. In sweeps/sweep_all that cost 144 keys per dataset on
    four of twelve datasets, and since the reports read jobs.json in preference
    to walking the tree, those runs disappeared from every CSV.
    """
    import multiprocessing as mp

    jobs_path = tmp_path / "jobs.json"
    per_worker, workers = 150, 6
    tasks = [(str(jobs_path), w * per_worker, per_worker) for w in range(workers)]
    # `spawn` bootstraps each child by re-importing whatever
    # `sys.modules["__main__"].__file__` points at. Streamlit's `AppTest` sets
    # that to the script it renders, and `AppTest.from_string` renders a *temp*
    # file that is deleted afterwards -- so once any Streamlit test has run in
    # this process, every child here dies on a stale path, `Pool` respawns them
    # forever, and the run hangs. That is exactly how one full-suite run sat for
    # two days. A serial run always shares the process, so it always hit it;
    # under `-n auto` it depends on which worker got this file.
    #
    # Clearing both attributes makes multiprocessing skip the main fixup
    # entirely (see multiprocessing/spawn.py::get_preparation_data), which is
    # what we want: `_merge_worker` is importable on its own.
    # `__spec__` must still *exist* (spawn.py reads it unguarded), so blank it
    # rather than delete it; `__file__` is the one that has to go.
    main = sys.modules["__main__"]
    had_file = hasattr(main, "__file__")
    saved_file = getattr(main, "__file__", None)
    saved_spec = getattr(main, "__spec__", None)
    if had_file:
        del main.__file__
    main.__spec__ = None
    try:
        # .get(timeout=) rather than .map(): a bare pool.map blocks forever if a
        # worker never reports back, so any problem in here fails the test
        # instead of hanging the run. The test takes ~1 s; this is a margin.
        with mp.get_context("spawn").Pool(workers) as pool:
            pool.map_async(_merge_worker, tasks).get(timeout=120)
    finally:
        if had_file:
            main.__file__ = saved_file
        main.__spec__ = saved_spec

    jobs = json.loads(jobs_path.read_text())
    assert len(jobs) == per_worker * workers, f"lost {per_worker * workers - len(jobs)} keys"
