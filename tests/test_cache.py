"""Stage cache: roundtrips, key chaining, and pipeline-level reuse."""

import numpy as np
import pytest

from behavior_graphs.core.types import FeatureSet, WindowInfo, WindowSet, episodes_from_lengths
from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.pipeline.cache import StageCache, data_fingerprint, stage_key
from behavior_graphs.pipeline.run import compose_pipeline_config, run_pipeline

DIM = 8
LAYER = "pi05"


@pytest.fixture(scope="module")
def rollouts(tmp_path_factory):
    rng = np.random.default_rng(0)
    regimes = rng.normal(0.0, 1.0, size=(3, DIM)) * 6.0
    lengths = [int(n) for n in rng.integers(30, 61, size=10)]
    successes = [i % 2 == 0 for i in range(10)]
    chunks = []
    for n in lengths:
        thirds = [n // 3, n // 3, n - 2 * (n // 3)]
        for regime, m in zip(regimes, thirds):
            chunks.append(regime + rng.normal(0.0, 0.5, size=(m, DIM)))
    embeddings = np.concatenate(chunks).astype(np.float32)
    path = tmp_path_factory.mktemp("rollouts")
    return write_rollout_dir(path, lengths, successes, {LAYER: embeddings})


class TestStageKey:
    def test_deterministic_and_order_sensitive(self):
        a = stage_key({"x": 1}, {"y": 2})
        assert a == stage_key({"x": 1}, {"y": 2})
        assert a != stage_key({"y": 2}, {"x": 1})
        assert a != stage_key({"x": 1}, {"y": 3})

    def test_dict_key_order_irrelevant(self):
        assert stage_key({"a": 1, "b": 2}) == stage_key({"b": 2, "a": 1})


class TestRoundtrip:
    def test_features(self, tmp_path):
        cache = StageCache(tmp_path)
        fs = FeatureSet(
            features=np.arange(12, dtype=np.float32).reshape(6, 2),
            episodes=episodes_from_lengths([4, 2], [True, False]),
        )
        cache.put_features("k1", fs)
        out = cache.get_features("k1")
        np.testing.assert_array_equal(out.features, fs.features)
        assert [ep.to_dict() for ep in out.episodes] == [ep.to_dict() for ep in fs.episodes]
        assert cache.get_features("nope") is None

    def test_windows(self, tmp_path):
        cache = StageCache(tmp_path)
        ws = WindowSet(
            features=np.ones((3, 2), dtype=np.float32),
            windows=[WindowInfo(0, 0, 2), WindowInfo(0, 2, 4), WindowInfo(1, 0, 2)],
            episodes=episodes_from_lengths([4, 2]),
        )
        cache.put_windows("k1", ws)
        out = cache.get_windows("k1")
        np.testing.assert_array_equal(out.features, ws.features)
        assert [w.to_dict() for w in out.windows] == [w.to_dict() for w in ws.windows]

    def test_array(self, tmp_path):
        cache = StageCache(tmp_path)
        arr = np.random.default_rng(0).normal(size=(5, 3))
        cache.put_array("labels", "k1", arr)
        np.testing.assert_array_equal(cache.get_array("labels", "k1"), arr)

    def test_put_is_idempotent(self, tmp_path):
        cache = StageCache(tmp_path)
        cache.put_array("labels", "k1", np.zeros(3))
        cache.put_array("labels", "k1", np.ones(3))  # ignored: entry exists
        np.testing.assert_array_equal(cache.get_array("labels", "k1"), np.zeros(3))


def test_data_fingerprint_changes_on_rewrite(rollouts, tmp_path):
    fp1 = data_fingerprint(rollouts.path)
    meta = rollouts.path / "episodes" / "metadata.yaml"
    text = meta.read_text()
    meta.write_text(text + "\n# touched\n")
    fp2 = data_fingerprint(rollouts.path)
    meta.write_text(text)
    assert fp1 != fp2


_compose = compose_pipeline_config


class TestPipelineReuse:
    def _overrides(self, rollouts, cache_dir, **extra):
        ov = [
            f"rollout_dir={rollouts.path}",
            f"cache_dir={cache_dir}",
            "pipeline/reduction@reduction=pca",
            "reduction.n_components=2",
            "clustering.n_clusters=3",
        ]
        ov += [f"{k}={v}" for k, v in extra.items()]
        return ov

    def test_second_run_hits_every_stage(self, rollouts, tmp_path):
        cache_dir = tmp_path / "cache"
        cfg = _compose(self._overrides(rollouts, cache_dir))
        r1 = run_pipeline(cfg, rollouts, tmp_path / "run1")
        assert r1.metrics["cache_hits"] == {
            "features": False,
            "windows": False,
            "reduced": False,
            "labels": False,
        }
        r2 = run_pipeline(cfg, rollouts, tmp_path / "run2")
        assert r2.metrics["cache_hits"] == {
            "features": True,
            "windows": True,
            "reduced": True,
            "labels": True,
        }
        np.testing.assert_array_equal(r1.labels, r2.labels)
        np.testing.assert_array_equal(r1.reduced, r2.reduced)
        assert r1.metrics["cluster_quality"] == r2.metrics["cluster_quality"]

    def test_downstream_change_reuses_upstream(self, rollouts, tmp_path):
        cache_dir = tmp_path / "cache"
        cfg = _compose(self._overrides(rollouts, cache_dir))
        run_pipeline(cfg, rollouts, tmp_path / "run1")
        cfg2 = _compose(self._overrides(rollouts, cache_dir, **{"clustering.n_clusters": 4}))
        r2 = run_pipeline(cfg2, rollouts, tmp_path / "run2")
        assert r2.metrics["cache_hits"] == {
            "features": True,
            "windows": True,
            "reduced": True,
            "labels": False,
        }
        assert len(np.unique(r2.labels[r2.labels >= 0])) == 4

    def test_disabled_cache(self, rollouts, tmp_path):
        cfg = _compose(
            [
                f"rollout_dir={rollouts.path}",
                "cache_dir=null",
                "pipeline/reduction@reduction=pca",
                "reduction.n_components=2",
                "clustering.n_clusters=3",
            ]
        )
        r = run_pipeline(cfg, rollouts, tmp_path / "run1")
        assert "cache_hits" not in r.metrics


def test_objective_in_metrics(rollouts, tmp_path):
    from omegaconf import OmegaConf

    cfg = _compose(
        [
            f"rollout_dir={rollouts.path}",
            f"cache_dir={tmp_path / 'cache'}",
            "pipeline/reduction@reduction=pca",
            "reduction.n_components=2",
            "clustering.n_clusters=3",
        ]
    )
    cfg.objective = OmegaConf.create(
        {
            "_target_": "behavior_graphs.model_selection.scoring.Objective",
            "weights": {"cluster_quality.silhouette": 1.0},
            "constraints": {"cluster_quality.silhouette": {"min": -1.0}},
        }
    )
    r = run_pipeline(cfg, rollouts, tmp_path / "run_obj")
    obj = r.metrics["objective"]
    assert obj["feasible"] is True
    assert obj["score"] == pytest.approx(r.metrics["cluster_quality"]["silhouette"])
