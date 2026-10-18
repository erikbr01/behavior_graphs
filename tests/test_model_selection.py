"""Model selection + sensitivity analysis over a synthetic runs index."""

import numpy as np
import pandas as pd
import pytest

from behavior_graphs.model_selection.index import flatten_config, unified_k, varying_columns
from behavior_graphs.model_selection.scoring import (
    CompositeSelection,
    best_per_k,
    composite_scores,
    metric_value,
    normalize_metric_columns,
    pareto_mask,
    score_metrics,
)
from behavior_graphs.model_selection.sensitivity import eta_squared, sensitivity_table


@pytest.fixture()
def df():
    """8 runs: 2 widths × 2 K × 2 reductions, silhouette favors width=5."""
    rows = []
    for width in (3, 5):
        for k in (5, 10):
            for red in ("PCAReduction", "UMAPReduction"):
                sil = 0.1 + (0.2 if width == 5 else 0.0) + (0.05 if red == "PCAReduction" else 0.0)
                rows.append(
                    {
                        "run": f"w{width}_k{k}_{red[:4]}",
                        "path": f"/runs/w{width}_k{k}_{red[:4]}",
                        "clustering": "KMeansClustering",
                        "reduction": red,
                        "windowing.width": width,
                        "k": k,
                        "silhouette": sil,
                        "davies_bouldin": 1.0 - sil,
                        "boundary_gain": 0.3 + (0.1 if k == 10 else 0.0),
                        "swap_rate": 0.1,
                    }
                )
    return pd.DataFrame(rows)


class TestScoreMetrics:
    METRICS_JSON = {"cluster_quality": {"silhouette": 0.4, "davies_bouldin": 0.8}}

    def test_weighted_sum(self):
        out = score_metrics(
            self.METRICS_JSON,
            {
                "weights": {
                    "cluster_quality.silhouette": 1.0,
                    "cluster_quality.davies_bouldin": -0.5,
                }
            },
        )
        assert out["score"] == pytest.approx(0.4 - 0.4)
        assert out["feasible"] is True

    def test_constraint_violation_flags_but_keeps_score(self):
        out = score_metrics(
            self.METRICS_JSON,
            {
                "weights": {"cluster_quality.silhouette": 1.0},
                "constraints": {"cluster_quality.silhouette": {"min": 0.5}},
            },
        )
        # The run is already computed: constraints flag, never discard.
        assert out["score"] == pytest.approx(0.4)
        assert out["feasible"] is False
        assert out["violated"] == ["cluster_quality.silhouette"]

    def test_missing_metric_contributes_nothing(self):
        out = score_metrics({}, {"weights": {"cluster_quality.silhouette": 1.0}})
        assert out["score"] == 0.0
        assert out["terms"] == {"cluster_quality.silhouette": None}

    def test_metric_value_paths(self):
        assert metric_value(self.METRICS_JSON, "cluster_quality.silhouette") == 0.4
        assert metric_value(self.METRICS_JSON, "missing.path") is None
        assert metric_value({"a": {"b": "str"}}, "a.b") is None


class TestNormalization:
    def test_direction_corrected(self, df):
        norm = normalize_metric_columns(df, ["silhouette", "davies_bouldin"])
        # Best silhouette row must also be best (1.0) on davies_bouldin here,
        # since db = 1 - silhouette and lower db is better.
        best = df["silhouette"].idxmax()
        assert norm.loc[best, "silhouette"] == pytest.approx(1.0)
        assert norm.loc[best, "davies_bouldin"] == pytest.approx(1.0)

    def test_constant_column_is_half(self, df):
        norm = normalize_metric_columns(df, ["swap_rate"])
        assert (norm["swap_rate"] == 0.5).all()


class TestCompositeScores:
    def test_score_in_unit_interval(self, df):
        out = composite_scores(df, {"silhouette": 1.0, "boundary_gain": 0.5})
        assert out["score"].between(0, 1).all()

    def test_constraints_flag_infeasible(self, df):
        out = composite_scores(df, {"silhouette": 1.0}, {"silhouette": {"min": 0.2}})
        assert (out["feasible"] == (df["silhouette"] >= 0.2)).all()

    def test_best_per_k_one_row_per_k(self, df):
        scored = composite_scores(df, {"silhouette": 1.0})
        d = df.assign(score=scored["score"], feasible=scored["feasible"])
        best = best_per_k(d)
        assert list(best["k"]) == [5, 10]
        # Width 5 + PCA dominates silhouette at every K.
        assert (best["windowing.width"] == 5).all()
        assert (best["reduction"] == "PCAReduction").all()

    def test_best_per_k_keeps_flagged_runs(self, df):
        # An unreachable floor flags every run, but recommendations still
        # come out — nothing computed is ever discarded.
        scored = composite_scores(df, {"silhouette": 1.0}, {"silhouette": {"min": 10.0}})
        d = df.assign(score=scored["score"], feasible=scored["feasible"])
        assert not d["feasible"].any()
        best = best_per_k(d)
        assert list(best["k"]) == [5, 10]
        assert not best["feasible"].any()


class TestPareto:
    def test_single_dominator(self):
        d = pd.DataFrame(
            {
                "silhouette": [0.1, 0.3, 0.2],
                "davies_bouldin": [1.0, 0.5, 0.8],
            }
        )
        assert pareto_mask(d, ["silhouette", "davies_bouldin"]).tolist() == [
            False,
            True,
            False,
        ]

    def test_tradeoff_keeps_both(self):
        d = pd.DataFrame(
            {
                "silhouette": [0.1, 0.3],
                "boundary_gain": [0.9, 0.2],
            }
        )
        assert pareto_mask(d, ["silhouette", "boundary_gain"]).all()

    def test_nan_rows_excluded(self):
        d = pd.DataFrame(
            {
                "silhouette": [0.1, np.nan],
                "boundary_gain": [0.9, 0.95],
            }
        )
        assert pareto_mask(d, ["silhouette", "boundary_gain"]).tolist() == [True, False]


class TestSensitivity:
    def test_width_explains_silhouette(self, df):
        table = sensitivity_table(df, ["windowing.width", "reduction", "k"], "silhouette")
        table = table.set_index("param")
        # Width contributes ±0.1, reduction ±0.025, k nothing.
        assert table.loc["windowing.width", "eta_squared"] > table.loc["reduction", "eta_squared"]
        assert table.loc["windowing.width", "best_value"] == "5"
        assert table.loc["windowing.width", "mean_spread"] == pytest.approx(0.2)
        assert "k" not in table.index or table.loc["k", "eta_squared"] == pytest.approx(
            0.0, abs=1e-9
        )

    def test_lower_is_better_direction(self, df):
        table = sensitivity_table(df, ["windowing.width"], "davies_bouldin")
        assert table.iloc[0]["best_value"] == "5"  # lower db at width 5

    def test_eta_squared_bounds(self, df):
        v = eta_squared(df, "windowing.width", "silhouette")
        assert 0.0 <= v <= 1.0


class TestIndexHelpers:
    CFG = {
        "features": {
            "_target_": "behavior_graphs.stages.features.policy_embedding.PolicyEmbeddings",
            "layer": "pi05",
        },
        "windowing": {
            "_target_": "behavior_graphs.stages.windowing.sliding.SlidingWindow",
            "width": 5,
            "stride": 2,
        },
        "reduction": {
            "_target_": "behavior_graphs.stages.reduction.pca.PCAReduction",
            "n_components": 32,
        },
        "clustering": {
            "_target_": "behavior_graphs.stages.clustering.kmeans.KMeansClustering",
            "n_clusters": 10,
        },
        "graph": {
            "_target_": "behavior_graphs.stages.graph.transition.TransitionGraphBuilder",
            "smoothing": "none",
        },
    }

    def test_flatten_config(self):
        row = flatten_config(self.CFG)
        assert row["clustering"] == "KMeansClustering"
        assert row["clustering.n_clusters"] == 10
        assert row["windowing.width"] == 5
        assert row["k"] == 10

    def test_unified_k_gmm(self):
        cfg = {"clustering": {"n_components": 7}}
        assert unified_k(cfg) == 7
        assert unified_k({"clustering": {"min_cluster_size": 15}}) is None

    def test_varying_columns(self, df):
        assert "windowing.width" in varying_columns(df)
        assert "clustering" not in varying_columns(df)


def test_pareto_front_2d_matches_pairwise_definition():
    """The skyline sweep must agree with the O(n²) dominance definition."""
    import numpy as np

    from behavior_graphs.model_selection.scoring import pareto_front_2d

    rng = np.random.default_rng(0)

    def naive(xs, ys):
        n = len(xs)
        out = np.ones(n, dtype=bool)
        for i in range(n):
            if np.isnan(xs[i]) or np.isnan(ys[i]):
                out[i] = False
                continue
            for j in range(n):
                if i == j or np.isnan(xs[j]) or np.isnan(ys[j]):
                    continue
                if xs[j] >= xs[i] and ys[j] >= ys[i] and (xs[j] > xs[i] or ys[j] > ys[i]):
                    out[i] = False
                    break
        return out

    for _ in range(30):
        n = int(rng.integers(1, 40))
        # Coarse grid so ties and duplicate points are common.
        xs = rng.integers(0, 5, n).astype(float)
        ys = rng.integers(0, 5, n).astype(float)
        assert (pareto_front_2d(xs, ys) == naive(xs, ys)).all()

    # NaNs are never on the front.
    xs = np.array([1.0, np.nan, 3.0])
    ys = np.array([1.0, 5.0, np.nan])
    assert pareto_front_2d(xs, ys).tolist() == [True, False, False]
    assert pareto_front_2d(np.array([]), np.array([])).tolist() == []


class TestTargetWrappers:
    """The _target_-instantiable selection/objective wrappers (hydra pattern)."""

    def test_composite_selection_matches_function(self, df):
        sel = CompositeSelection(
            weights={"silhouette": 1.0, "boundary_gain": 0.5},
            constraints={"silhouette": {"min": 0.2}},
        )
        got = sel.score(df)
        want = composite_scores(
            df, {"silhouette": 1.0, "boundary_gain": 0.5}, {"silhouette": {"min": 0.2}}
        )
        pd.testing.assert_frame_equal(got, want)

    def test_composite_selection_instantiate(self, df):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        sel = instantiate(
            OmegaConf.create(
                {
                    "_target_": "behavior_graphs.model_selection.scoring.CompositeSelection",
                    "weights": {"silhouette": 1.0},
                    "constraints": {"silhouette": {"min": -1.0}},
                }
            )
        )
        assert isinstance(sel, CompositeSelection)
        assert sel.score(df)["feasible"].all()

    def test_composite_selection_defaults(self):
        sel = CompositeSelection()
        assert sel.weights["silhouette"] == 1.0
        assert sel.constraints["silhouette"]["min"] == 0.05
