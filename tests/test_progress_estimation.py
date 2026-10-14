"""Config-instantiation + CPU end-to-end tests for the progress_estimation
benchmark (methods / metrics / backbone / split configs, and the BenchmarkRunner).

The config-instantiation tests mirror tests/test_configs.py: every method/metric
config must build its object WITHOUT a GPU or heavy VLM deps (VLM methods import
torch/robometer lazily in ``fit``, never at construction)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from behavior_graphs.progress_estimation import success_auroc
from behavior_graphs.progress_estimation.io import EpisodeSet
from behavior_graphs.progress_estimation.methods import (
    FrameIndex,
    GraphBackbone,
    GraphStage,
    ProgressMethod,
)
from behavior_graphs.progress_estimation.methods.mlp import _stack_window
from behavior_graphs.progress_estimation.metrics import (
    MAE,
    VOC,
    ProgressMetric,
    SuccessAUROC,
)
from behavior_graphs.progress_estimation.runner import (
    AlignedSplit,
    BenchmarkRunner,
    KFoldSplit,
    SplitSpec,
)
from behavior_graphs.stages import KMeansClustering

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "progress_estimation"


@pytest.mark.parametrize("path", sorted((CONFIGS / "method").glob("*.yaml")), ids=lambda p: p.stem)
def test_method_configs_instantiate(path):
    obj = instantiate(OmegaConf.load(path))
    assert isinstance(obj, ProgressMethod)
    assert obj.name
    assert obj.modality in ("embedding", "video")


@pytest.mark.parametrize("path", sorted((CONFIGS / "metric").glob("*.yaml")), ids=lambda p: p.stem)
def test_metric_configs_instantiate(path):
    obj = instantiate(OmegaConf.load(path))
    assert isinstance(obj, ProgressMetric)
    assert obj.name


@pytest.mark.parametrize(
    "path", sorted((CONFIGS / "backbone").glob("*.yaml")), ids=lambda p: p.stem
)
def test_backbone_configs_instantiate(path):
    target = OmegaConf.load(path).get("reduction", {}).get("_target_", "")
    if "umap" in target:
        pytest.importorskip("umap")
    obj = instantiate(OmegaConf.load(path))
    assert isinstance(obj, GraphBackbone)
    assert obj.reduction is not None and obj.clustering is not None


@pytest.mark.parametrize("path", sorted((CONFIGS / "split").glob("*.yaml")), ids=lambda p: p.stem)
def test_split_configs_instantiate(path):
    assert isinstance(instantiate(OmegaConf.load(path)), SplitSpec)


def _synthetic_set(n=40, dim=6, seed=0):
    """In-memory EpisodeSet: monotone-drift embeddings so stage progress is real."""
    rng = np.random.default_rng(seed)
    flags = [(i % 5) != 0 for i in range(n)]  # 80% success
    embs = []
    for _ in range(n):
        T = int(rng.integers(20, 40))
        t = np.linspace(0, 1, T)[:, None]
        embs.append((t * rng.normal(size=(1, dim)) + 0.1 * rng.normal(size=(T, dim))).astype(float))
    ds = EpisodeSet("synth", "/nonexistent", flags)
    ds._embeddings = embs  # inject; no disk read
    return ds


def test_runner_graph_end_to_end_cpu():
    ds = _synthetic_set()
    bb = GraphBackbone(clustering=KMeansClustering(n_clusters=4, random_state=0))
    methods = [GraphStage(backbone=bb), FrameIndex(backbone=bb)]
    metrics = [VOC(), MAE(), SuccessAUROC()]
    summary = BenchmarkRunner(metrics, KFoldSplit(n_folds=2, seed=0)).run(ds, methods)

    assert set(summary) == {"graph_stage", "frame_index"}
    for name in summary:
        for m in ("voc", "mae", "auroc"):
            assert m in summary[name]
    # frame index is the literal window order -> perfect VOC
    assert summary["frame_index"]["voc"][0] == pytest.approx(1.0, abs=1e-6)
    # graph-stage tracks the injected drift -> clearly positive VOC
    assert summary["graph_stage"]["voc"][0] > 0.3


def test_shared_backbone_fits_once_per_fold():
    ds = _synthetic_set()
    bb = GraphBackbone(clustering=KMeansClustering(n_clusters=4, random_state=0))
    calls = {"n": 0}
    orig_fit = bb.fit

    def counting_fit(train):
        calls["n"] += 1
        return orig_fit(train)

    bb.fit = counting_fit
    methods = [GraphStage(backbone=bb), FrameIndex(backbone=bb)]
    BenchmarkRunner([VOC()], KFoldSplit(n_folds=2, seed=0)).run(ds, methods)
    # 2 folds x 1 backbone fit each (graph_stage triggers it; frame_index reuses)
    assert calls["n"] == 2


def test_aligned_split_partitions_by_task_block():
    split = AlignedSplit(eval_per_task=2, n_tasks=5)
    (tr, te) = split.folds([True] * 20)[0]
    assert sorted(tr + te) == list(range(20))
    assert set(tr).isdisjoint(te)
    assert len(te) == 10  # 2 per task x 5 tasks (block size 4)


def test_aligned_split_refuses_to_leave_nothing_to_train_on():
    # The shipped defaults do this to any dataset under 1000 episodes, and the
    # empty train list used to surface as `np.concatenate([])` inside a
    # backbone, several minutes into the fit.
    with pytest.raises(ValueError, match="nothing to train on"):
        AlignedSplit().folds([True] * 100)
    with pytest.raises(ValueError, match="nothing to train on"):
        AlignedSplit(eval_per_task=4, n_tasks=5).folds([True] * 20)  # block size 4


def test_the_per_fold_line_survives_a_metric_set_without_voc():
    # `metric/auroc.yaml` selects one metric on its own, and the progress line
    # used to format a missing `voc` as None -- after the fold had been fit.
    ds = _synthetic_set(n=20)
    bb = GraphBackbone(clustering=KMeansClustering(n_clusters=3, random_state=0))
    summary = BenchmarkRunner([SuccessAUROC()], KFoldSplit(n_folds=2, seed=0)).run(
        ds, [GraphStage(backbone=bb)]
    )
    assert set(summary["graph_stage"]) == {"auroc"}


def test_a_window_wider_than_the_episode_pads_rather_than_raising():
    # window_width is a config knob and the zoo spans 6-to-708-step episodes, so
    # `width > T + 1` is reachable; it used to broadcast a negative slice and
    # raise mid-fit rather than degrading to an all-pad column.
    ep = np.arange(12, dtype=float).reshape(3, 4)
    out = _stack_window(ep, 8)
    assert out.shape == (3, 4 * 8)
    assert np.array_equal(out[:, :4], ep)  # j=0 is the episode itself
    assert np.array_equal(out[:, -4:], np.repeat(ep[:1], 3, axis=0))  # j>=T is all pad


def test_an_unscored_episode_leaves_the_auroc_rather_than_losing_it():
    # NaN is False for both `>` and `==`, so an episode with no prediction used
    # to score zero against every counterpart while still sitting in the
    # denominator -- a silent downward bias the size of the decode-failure rate.
    flags = np.array([True, True, False, False])
    assert success_auroc(np.array([0.9, np.nan, 0.2, 0.1]), flags) == pytest.approx(1.0)
    assert success_auroc(np.array([0.9, 0.8, 0.2, 0.1]), flags) == pytest.approx(1.0)
    # Nothing left on one side is undefined, not zero.
    assert np.isnan(success_auroc(np.array([np.nan, np.nan, 0.2, 0.1]), flags))


def test_booktabs_table_matches_the_displayed_frame():
    """The LaTeX export must reproduce the on-screen table, not re-derive it.

    Guards the three translations the renderer owns (markdown bold, the ± sign,
    the em-dash placeholder for a missing cell) plus the two things that break
    silently in a paper build: an unescaped ``_``/``%`` in a method name, and the
    U+00B7 middot the UI uses to tag baseline rows, which plain pdflatex cannot
    typeset.
    """
    import pandas as pd

    from behavior_graphs.progress_estimation.report import booktabs_table

    frame = pd.DataFrame(
        {
            "graph_stage (Graph)": {"voc": "**0.998±0.001**", "mae": "—"},
            "mlp_100% (MLP)": {"voc": "0.640", "mae": "**0.088**"},
            "frame_index (Time) · baseline": {"voc": "1.000", "mae": "0.000"},
        }
    ).T
    tex = booktabs_table(
        frame,
        caption="Caption.",
        label="tab:t",
        higher_better={"voc": True, "mae": False},
        baseline_suffix=" · baseline",
    )

    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert "\\begin{tabular}{lrr}" in tex  # one l + one r per metric
    assert "voc $\\uparrow$" in tex and "mae $\\downarrow$" in tex
    assert "\\textbf{0.998$\\pm$0.001}" in tex  # bold + ± survive together
    assert "\\textbf{0.088}" in tex  # lower-is-better winner bolded
    assert "& -- \\\\" in tex  # missing cell, not an em-dash
    assert "mlp\\_100\\%" in tex  # LaTeX specials escaped
    assert "·" not in tex  # middot never reaches the .tex
    assert "frame\\_index (Time) \\emph{(baseline)}" in tex


def test_booktabs_table_handles_empty_and_exotic_names():
    """An empty selection and LaTeX-hostile characters must not emit broken .tex."""
    import pandas as pd

    from behavior_graphs.progress_estimation.report import booktabs_table

    empty = booktabs_table(
        pd.DataFrame({}).T,
        caption="c",
        label="l",
        higher_better={},
    )
    assert "\\begin{tabular}" not in empty

    frame = pd.DataFrame({"a~b^c (X)": {"voc": "0.5"}}).T
    tex = booktabs_table(frame, caption="c", label="l", higher_better={"voc": True})
    assert "\\textasciitilde{}" in tex and "\\textasciicircum{}" in tex
    assert "~" not in tex and "^" not in tex


def test_metadata_task_strings_only_fill_episodes_with_no_instruction():
    """The metadata fallback must not restate an already-answered question.

    The three perturbation suites were scored with an EMPTY task prompt, which
    is what drove the VLM baselines to negative VOC there. Their rollout
    metadata records the prompt the policy actually got, so it is read as a last
    resort. Ranking it last is the whole point: promoting it would change the
    question every already-scored dataset was asked, so the suite list and an
    explicit instruction both still win.
    """
    from behavior_graphs.progress_estimation.io import (
        metadata_instructions,
        resolve_instructions,
    )

    episodes = [
        {"success": True, "prompt": "put the mug on the shelf"},
        {"success": False, "task_name": "pick_up_the_black_bowl"},
        {"success": True},
    ]
    from_meta = metadata_instructions(episodes)
    # `task_name` is de-underscored; an episode recording nothing stays absent.
    assert from_meta == {0: "put the mug on the shelf", 1: "pick up the black bowl"}

    # Nothing else known: the fallback is what the method gets asked about.
    filled = resolve_instructions(3, from_metadata=from_meta)
    assert filled[0] == "put the mug on the shelf"
    assert filled[1] == "pick up the black bowl"
    assert filled[2] == ""  # no source at all, still empty

    # A suite list outranks it, per episode.
    suited = resolve_instructions(3, suite="libero_object", from_metadata=from_meta)
    assert suited[0] == "pick up the alphabet soup and place it in the basket"

    # So does one explicit instruction for the whole dataset.
    single = resolve_instructions(3, instruction="explicit", from_metadata=from_meta)
    assert set(single.values()) == {"explicit"}

    # And with no metadata the behaviour is exactly what it was before.
    assert resolve_instructions(2) == {0: "", 1: ""}


def test_sweep_table_survives_a_dataset_where_no_k_scored():
    """`best_k: None` must not kill the table after every dataset has been fitted.

    sweep_progress_allds writes `best_k = None` when no K produced a finite stage
    VOC, while still writing `per_k`. That passes the "per_k not in r" guard, and
    `pk.get(None, pk.get("None"))` then yields None, so `kd[STAGE]` raised
    TypeError during report writing -- i.e. after all the GPU work was done.
    """
    from behavior_graphs.progress_estimation.report import sweep_table_md

    results = {
        "ok_ds": {
            "family": "sim",
            "n_succ": 40,
            "n_fail": 10,
            "mean_len": 120.0,
            "best_k": 16,
            "per_k": {
                "16": {
                    "graph_direct:stage": {"voc": 0.9, "pearson": 0.9, "mae": 0.1, "auroc": 0.8},
                    "graph_direct:absorb": {"voc": 0.5, "pearson": 0.5, "mae": 0.3, "auroc": 0.7},
                }
            },
        },
        "unscorable_ds": {
            "family": "sim",
            "n_succ": 0,
            "n_fail": 50,
            "mean_len": 90.0,
            "best_k": None,
            "per_k": {
                "16": {
                    "graph_direct:stage": {
                        "voc": float("nan"),
                        "pearson": float("nan"),
                        "mae": float("nan"),
                        "auroc": float("nan"),
                    },
                    "graph_direct:absorb": {
                        "voc": float("nan"),
                        "pearson": float("nan"),
                        "mae": float("nan"),
                        "auroc": float("nan"),
                    },
                }
            },
        },
    }
    text = sweep_table_md(results)
    assert "ok_ds" in text and "unscorable_ds" in text
    # The unscorable dataset is reported as such rather than crashing or being dropped.
    assert "no finite stage VOC at any K" in text
