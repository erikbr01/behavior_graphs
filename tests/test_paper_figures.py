"""Every ablation figure must build and export from cells alone.

The point of keeping the builders free of disk and Streamlit is that they can be
pinned here: a figure that raises only when the app renders it is a figure nobody
notices is broken. These use synthetic cells, so they run without a sweep on
disk and cover the shapes real data does not currently have -- a single
representation, a missing metric, an HDBSCAN-only arm.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from behavior_graphs.ablation_study.cells import Cell
from behavior_graphs.ablation_study.config import DatasetConfig, EvalSpec
from behavior_graphs.viz import paper_figures as pf


def _cfg(**kw) -> DatasetConfig:
    base = dict(
        name="test",
        task="t",
        root=pytest.importorskip("pathlib").Path("/nonexistent"),
        sweep_dir=pytest.importorskip("pathlib").Path("/nonexistent"),
        representations=("state", "rgrl_dit", "dinov3"),
        evaluation=EvalSpec(),
    )
    base.update(kw)
    return DatasetConfig(**base)


def _cell(layer: str, k: int, score: float, **kw) -> Cell:
    metrics = {
        "segmental_f1@0.5": score,
        "segmental_uniform_f1@0.5": 0.4,
        "segmental_f1@0.1": score + 0.1,
        "segmental_f1@0.25": score + 0.05,
        "segmental_f1@0.75": max(0.0, score - 0.1),
        "density": kw.pop("density", 1.0),
    }
    metrics.update(kw.pop("metrics", {}))
    fields = dict(
        dir=f"/tmp/{layer}{k}{score}",
        layer=layer,
        reduction="pca",
        n_components=10,
        clusterer="kmeans",
        k=k,
        width=4,
        stride=4,
        smoothing="none",
        min_segment="1",
        score=score,
        density=metrics["density"],
        gain=score - 0.4,
        metrics=metrics,
    )
    fields.update(kw)
    return Cell(**fields)


def _rows() -> list[Cell]:
    out = []
    for layer, peak in (("state", 0.60), ("rgrl_dit", 0.55), ("dinov3", 0.48)):
        for k in (4, 8, 16, 32):
            out.append(_cell(layer, k, peak - abs(k - 8) * 0.01, density=1.0 + (k - 8) * 0.02))
    return out


@pytest.mark.parametrize("name", sorted(pf.FIGURES))
def test_every_figure_builds_and_exports(name):
    fig = pf.build(name, _rows(), _cfg())
    try:
        for fmt in ("pdf", "svg", "png"):
            blob = pf.export(fig, fmt)
            assert blob, f"{name} exported empty {fmt}"
        # A PDF is recognisable from its first bytes; a silently empty canvas is
        # not, which is why the size floor is here too.
        assert pf.export(fig, "pdf").startswith(b"%PDF")
        assert len(pf.export(fig, "png")) > 5_000
    finally:
        plt.close(fig)


@pytest.mark.parametrize("name", sorted(pf.FIGURES))
def test_every_figure_survives_one_representation(name):
    """A dataset mid-sweep has arms with no cells yet."""
    rows = [c for c in _rows() if c.layer == "state"]
    fig = pf.build(name, rows, _cfg())
    try:
        assert pf.export(fig, "pdf").startswith(b"%PDF")
    finally:
        plt.close(fig)


def test_k_curve_excludes_hdbscan():
    """HDBSCAN's dial is a minimum cluster size, not a K; 200 is not a big K."""
    rows = _rows() + [_cell("state", 200, 0.2, clusterer="hdbscan")]
    fig = pf.k_curve(rows, _cfg())
    try:
        xs = [x for line in fig.axes[0].get_lines() for x in line.get_xdata()]
        assert 200 not in xs, "an HDBSCAN min_cluster_size was plotted as a cluster count"
    finally:
        plt.close(fig)


def test_k_curve_handles_an_arm_with_only_hdbscan():
    rows = [_cell("state", 50, 0.3, clusterer="hdbscan")]
    fig = pf.k_curve(rows, _cfg())
    try:
        assert pf.export(fig, "pdf").startswith(b"%PDF")
    finally:
        plt.close(fig)


def test_iou_sweep_tolerates_a_missing_threshold():
    rows = [_cell("state", 8, 0.5, metrics={"segmental_f1@0.75": None})]
    rows[0].metrics.pop("segmental_f1@0.75")
    fig = pf.iou_sweep(rows, _cfg())
    try:
        assert pf.export(fig, "pdf").startswith(b"%PDF")
    finally:
        plt.close(fig)


def test_export_rejects_an_unsupported_format():
    fig = pf.build("k_curve", _rows(), _cfg())
    try:
        with pytest.raises(ValueError, match="unsupported format"):
            pf.export(fig, "pgf")
    finally:
        plt.close(fig)


def test_paper_style_does_not_leak():
    """The rc context must restore whatever the caller had."""
    before = plt.rcParams["font.size"]
    with pf.paper_style():
        assert plt.rcParams["font.size"] == pf.PAPER_RC["font.size"]
    assert plt.rcParams["font.size"] == before


def test_unknown_figure_names_itself():
    with pytest.raises(KeyError, match="unknown figure"):
        pf.build("nope", _rows(), _cfg())


def test_labels_fall_back_for_an_unseen_layer():
    assert pf.rep_label("state") == "Proprioception"
    assert pf.rep_label("brand_new") == "brand_new"
    assert pf.rep_color("brand_new") == pf._FALLBACK
