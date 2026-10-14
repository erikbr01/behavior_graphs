"""Synthetic end-to-end and unit tests for the pipeline stages."""

import numpy as np
import pytest

from behavior_graphs.core.graph import START_NODE_ID, TERMINAL_NODE_IDS, BehaviorGraph
from behavior_graphs.core.result import PipelineResult
from behavior_graphs.core.stages import GraphConstruction
from behavior_graphs.core.types import FeatureSet, episodes_from_lengths
from behavior_graphs.envs.trajectory_dir import write_rollout_dir
from behavior_graphs.stages import (
    ClusterQualityEvaluator,
    GraphStatsEvaluator,
    IdentityReduction,
    KMeansClustering,
    PCAReduction,
    PolicyEmbeddings,
    SlidingWindow,
    TransitionGraphBuilder,
    ValueEvaluator,
    make_normalizer,
)

DIM = 8
N_EPISODES = 12
LAYER = "pi05"


@pytest.fixture(scope="module")
def rollouts(tmp_path_factory):
    """12 episodes, 30-60 steps, 8-dim embeddings from 3 regimes, half successes."""
    rng = np.random.default_rng(0)
    regimes = rng.normal(0.0, 1.0, size=(3, DIM)) * 6.0
    lengths = [int(n) for n in rng.integers(30, 61, size=N_EPISODES)]
    successes = [i % 2 == 0 for i in range(N_EPISODES)]
    chunks = []
    for n in lengths:
        thirds = [n // 3, n // 3, n - 2 * (n // 3)]
        for regime, m in zip(regimes, thirds):
            chunks.append(regime + rng.normal(0.0, 0.5, size=(m, DIM)))
    embeddings = np.concatenate(chunks).astype(np.float32)
    path = tmp_path_factory.mktemp("rollouts")
    return write_rollout_dir(path, lengths, successes, {LAYER: embeddings})


@pytest.fixture(scope="module")
def pipeline_result(rollouts):
    """Run the full pipeline once and share the result across tests."""
    result = PipelineResult()
    result.features = PolicyEmbeddings(layer=LAYER).extract(rollouts)
    result.windows = SlidingWindow(width=5, stride=2, aggregation="mean").transform(result.features)
    result.reduced = PCAReduction(n_components=2).fit_transform(result.windows.features)
    result.labels = KMeansClustering(n_clusters=3).fit_predict(result.reduced)
    result.graph = TransitionGraphBuilder(smoothing="median", smoothing_window=3).build(
        result.labels, result.windows
    )
    return result


class TestEndToEnd:
    def test_graph_structure(self, pipeline_result):
        graph = pipeline_result.graph
        assert START_NODE_ID in graph.nodes
        assert graph.terminal_node_ids <= TERMINAL_NODE_IDS
        assert len(graph.terminal_node_ids) >= 1
        assert len(graph.cluster_nodes) == 3
        assert graph.num_episodes == N_EPISODES
        for src, targets in graph.transition_probs.items():
            assert sum(targets.values()) == pytest.approx(1.0), f"node {src}"

    def test_compute_values_finite(self, pipeline_result):
        values = pipeline_result.graph.compute_values()
        assert isinstance(values, dict)
        assert all(np.isfinite(v) for v in values.values())
        assert set(values) >= set(pipeline_result.graph.cluster_nodes)

    def test_graph_stats_evaluator(self, pipeline_result):
        metrics = GraphStatsEvaluator().evaluate(pipeline_result)
        assert metrics["num_cluster_nodes"] == 3
        assert metrics["num_episodes"] == N_EPISODES
        assert metrics["success_rate"] == pytest.approx(0.5)
        assert metrics["num_edges"] > 0
        assert metrics["mean_out_degree"] >= 1.0
        assert metrics["mean_transition_entropy"] >= 0.0
        assert 1 <= len(metrics["top_paths"]) <= 5
        for entry in metrics["top_paths"]:
            assert entry["path"][0] == START_NODE_ID
            assert 0.0 < entry["probability"] <= 1.0

    def test_cluster_quality_evaluator(self, pipeline_result):
        metrics = ClusterQualityEvaluator().evaluate(pipeline_result)
        assert metrics["n_clusters"] == 3
        assert metrics["noise_fraction"] == 0.0
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz"):
            assert np.isfinite(metrics[key])

    def test_value_evaluator(self, pipeline_result):
        metrics = ValueEvaluator(gamma=0.95).evaluate(pipeline_result)
        assert all(np.isfinite(v) for v in metrics["node_values"].values())
        assert 0.0 <= metrics["frac_negative_advantage"] <= 1.0
        assert metrics["mean_abs_advantage"] >= 0.0


class TestSlidingWindow:
    def make_features(self, lengths):
        episodes = episodes_from_lengths(lengths, [None] * len(lengths))
        total = sum(lengths)
        features = np.arange(total, dtype=np.float32).reshape(-1, 1)
        return FeatureSet(features=features, episodes=episodes)

    def test_windows_never_cross_episodes(self):
        features = self.make_features([7, 5, 9])
        ws = SlidingWindow(width=4, stride=2).transform(features)
        for w in ws.windows:
            n = features.episodes[w.episode_idx].num_samples
            assert 0 <= w.start < w.end <= n

    def test_short_episode_emits_single_full_window(self):
        features = self.make_features([3, 10])
        ws = SlidingWindow(width=5, stride=2).transform(features)
        ep0 = [w for w in ws.windows if w.episode_idx == 0]
        assert len(ep0) == 1
        assert (ep0[0].start, ep0[0].end) == (0, 3)

    def test_degenerate_is_per_timestep(self):
        features = self.make_features([4, 3])
        ws = SlidingWindow(width=1, stride=1).transform(features)
        assert ws.num_windows == 7
        assert np.allclose(ws.features, features.features)
        assert all(w.end - w.start == 1 for w in ws.windows)

    def test_aggregations(self):
        features = self.make_features([4])
        for aggregation, expected in [("mean", 1.5), ("sum", 6.0), ("max", 3.0)]:
            ws = SlidingWindow(width=4, stride=1, aggregation=aggregation).transform(features)
            assert ws.features[0, 0] == pytest.approx(expected)

    def test_window_count(self):
        features = self.make_features([10])
        ws = SlidingWindow(width=4, stride=3).transform(features)
        # starts at 0, 3, 6 (9 + 4 > 10)
        assert [(w.start, w.end) for w in ws.windows] == [(0, 4), (3, 7), (6, 10)]

    def test_invalid_aggregation(self):
        with pytest.raises(ValueError):
            SlidingWindow(aggregation="median")


class TestReduction:
    def test_make_normalizer(self):
        assert make_normalizer("none") is None
        for method in ("standard", "minmax", "robust"):
            scaler = make_normalizer(method)
            assert hasattr(scaler, "fit")
        with pytest.raises(ValueError):
            make_normalizer("l3")

    def test_identity_with_normalize(self):
        x = np.random.default_rng(0).normal(3.0, 2.0, size=(50, 4)).astype(np.float32)
        out = IdentityReduction(normalize="standard").fit_transform(x)
        assert out.shape == x.shape
        assert np.allclose(out.mean(axis=0), 0.0, atol=1e-4)

    def test_identity_passthrough(self):
        x = np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        assert np.allclose(IdentityReduction().fit_transform(x), x)

    def test_pca_fit_transform_contract(self):
        rng = np.random.default_rng(0)
        train = rng.normal(size=(60, 8)).astype(np.float32)
        test = rng.normal(size=(20, 8)).astype(np.float32)
        reducer = PCAReduction(n_components=3, normalize="standard").fit(train)
        assert reducer.transform(train).shape == (60, 3)
        assert reducer.transform(test).shape == (20, 3)

    def test_pca_clamps_components(self):
        x = np.random.default_rng(0).normal(size=(5, 4)).astype(np.float32)
        out = PCAReduction(n_components=32).fit_transform(x)
        assert out.shape == (5, 4)

    def test_umap_reduction(self):
        pytest.importorskip("umap")
        from behavior_graphs.stages.reduction import UMAPReduction

        x = np.random.default_rng(0).normal(size=(60, 6)).astype(np.float32)
        reducer = UMAPReduction(n_components=2, n_neighbors=5).fit(x)
        out = reducer.transform(x)
        assert out.shape == (60, 2)
        assert np.isfinite(out).all()


class TestClustering:
    def test_kmeans_labels(self):
        rng = np.random.default_rng(0)
        x = np.concatenate([rng.normal(c, 0.1, size=(30, 2)) for c in (-5.0, 0.0, 5.0)]).astype(
            np.float32
        )
        labels = KMeansClustering(n_clusters=3).fit_predict(x)
        assert labels.shape == (90,)
        assert len(np.unique(labels)) == 3


class TestPolicyEmbeddings:
    def test_extract(self, rollouts):
        features = PolicyEmbeddings(layer=LAYER).extract(rollouts)
        assert features.dim == DIM
        assert features.num_samples == sum(ep.num_samples for ep in features.episodes)
        assert len(features.episodes) == N_EPISODES

    def test_missing_layer(self, rollouts):
        with pytest.raises(FileNotFoundError):
            PolicyEmbeddings(layer="missing").extract(rollouts)


#: Filled by :class:`RecordingGraphBuilder`; module level because hydra has to
#: locate the builder by import path, so it cannot be nested in a test.
GRAPH_STAGE_SAW: dict = {}


class RecordingGraphBuilder(GraphConstruction):
    """A graph stage that records the features it was handed."""

    def build(self, labels, windows):
        GRAPH_STAGE_SAW["features"] = np.array(windows.features, copy=True)
        GRAPH_STAGE_SAW["n_windows"] = len(windows.windows)
        return BehaviorGraph.from_window_labels(labels, windows)


class TestGraphStageFeatureSpace:
    """The graph stage must smooth in the space the clustering was fitted in.

    ``run.py`` clusters ``result.reduced`` but a ``WindowSet`` carries the raw
    features, and the graph stage's optional smoothing scores each window's
    distance to a cluster centroid. Handing it raw features measures in a space
    UMAP did not preserve, where the decoder overrides labels instead of
    correcting them (13.7% of windows relabeled at λ=1 across sweep_all, median
    trajectory tree +56%; in the clustering space, 0.4% and -11%).

    Only observable from outside the stage, so it is pinned at the call site.
    """

    def test_the_pipeline_hands_the_builder_the_clustering_space(self, tmp_path):
        from behavior_graphs.envs.trajectory_dir import RolloutDirectory
        from behavior_graphs.pipeline.run import compose_pipeline_config, run_pipeline

        GRAPH_STAGE_SAW.clear()
        rng = np.random.default_rng(0)
        rollouts = write_rollout_dir(
            tmp_path / "rollouts",
            [30, 30, 30],
            [True, False, True],
            {"pi05": rng.normal(size=(90, 16)).astype(np.float32)},
        )
        cfg = compose_pipeline_config(
            [
                f"rollout_dir={rollouts.path}",
                "features.layer=pi05",
                "clustering.n_clusters=3",
                "reduction.n_components=4",
                "cache_dir=null",
            ]
        )
        cfg.graph = {"_target_": f"{__name__}.RecordingGraphBuilder"}

        result = run_pipeline(cfg, RolloutDirectory(rollouts.path), tmp_path / "run")

        assert GRAPH_STAGE_SAW["n_windows"] == result.windows.num_windows
        # 4 reduced components, not the 16 raw feature dims.
        assert GRAPH_STAGE_SAW["features"].shape == result.reduced.shape
        assert np.allclose(GRAPH_STAGE_SAW["features"], result.reduced)
        assert result.windows.features.shape[1] == 16, "and the raw windows are untouched"
