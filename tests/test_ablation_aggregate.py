"""Pooling the per-task ablation rows into one table.

The slow half -- selecting each level on one fold and re-scoring it on another --
is `report/latex.rows_for` and is covered where it lives. What is testable here
is everything the cache buys: that a pooled mean only ever covers datasets it can
name, that the win count says something the mean does not, and that two
incompatible caches are refused rather than averaged.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from behavior_graphs.ablation_study.config import DatasetConfig, EvalSpec
from behavior_graphs.ablation_study.report import aggregate


def _cfg(tmp_path, name: str, criterion: str = "segmental_f1@0.75") -> DatasetConfig:
    sweep = tmp_path / f"sweep_{name}"
    sweep.mkdir(parents=True, exist_ok=True)
    return DatasetConfig(
        name=name,
        task=f"task_{name}",
        root=tmp_path / f"root_{name}",
        sweep_dir=sweep,
        evaluation=dataclasses.replace(EvalSpec(), criterion=criterion),
    )


def _write(cfg: DatasetConfig, entries, *, spread=None, version=None) -> None:
    """A cache file by hand, so no test needs a swept dataset."""
    payload = {
        "cache_version": aggregate.CACHE_VERSION if version is None else version,
        "dataset": cfg.name,
        "task": cfg.task,
        "criterion": cfg.evaluation.criterion,
        "written_at": "2026-09-15T12:00:00",
        "entries": [{"axis": a, "level": lv, "f1": f} for a, lv, f in entries],
        "spread": spread or {},
        "meta": {},
    }
    aggregate.cache_path(cfg).write_text(json.dumps(payload))


# ── Reading caches ───────────────────────────────────────────────────────────


def test_a_dataset_with_no_cache_is_named_not_dropped(tmp_path):
    a = _cfg(tmp_path, "a")
    b = _cfg(tmp_path, "b")
    _write(a, [("reduction", "umap", 0.5)])
    agg = aggregate.pooled([a, b])
    assert agg.datasets == ["a"]
    assert agg.missing == ["b"]


def test_a_cache_from_an_older_shape_is_refused(tmp_path):
    a = _cfg(tmp_path, "a")
    _write(a, [("reduction", "umap", 0.5)], version=aggregate.CACHE_VERSION - 1)
    agg = aggregate.pooled([a])
    assert agg.missing == ["a"] and not agg.rows


def test_an_unreadable_cache_counts_as_missing(tmp_path):
    a = _cfg(tmp_path, "a")
    aggregate.cache_path(a).write_text("{not json")
    assert aggregate.pooled([a]).missing == ["a"]


def test_nothing_cached_at_all_is_empty_rather_than_an_error(tmp_path):
    agg = aggregate.pooled([_cfg(tmp_path, "a")])
    assert agg.rows == [] and agg.datasets == []


# ── Pooling ──────────────────────────────────────────────────────────────────


def test_the_mean_is_over_the_datasets_that_have_the_level(tmp_path):
    a, b = _cfg(tmp_path, "a"), _cfg(tmp_path, "b")
    _write(a, [("reduction", "umap", 0.4), ("reduction", "pca", 0.2)])
    _write(b, [("reduction", "umap", 0.6)])
    rows = {r["level"]: r for r in aggregate.pooled([a, b]).rows}
    assert rows["umap"]["mean"] == pytest.approx(0.5)
    assert rows["umap"]["n"] == 2
    # pca exists on one dataset only, and says so rather than being averaged
    # against a zero for the other.
    assert rows["pca"]["mean"] == pytest.approx(0.2)
    assert rows["pca"]["n"] == 1


def test_the_win_count_separates_two_options_with_the_same_mean(tmp_path):
    # The case the docstring is about: `steady` wins narrowly on both tasks,
    # `spiky` wins hugely on one and loses on the other, and the means are equal.
    a, b = _cfg(tmp_path, "a"), _cfg(tmp_path, "b")
    _write(a, [("reduction", "steady", 0.50), ("reduction", "spiky", 0.90)])
    _write(b, [("reduction", "steady", 0.50), ("reduction", "spiky", 0.10)])
    rows = {r["level"]: r for r in aggregate.pooled([a, b]).rows}
    assert rows["steady"]["mean"] == pytest.approx(rows["spiky"]["mean"])
    assert rows["steady"]["wins"] == 1
    assert rows["spiky"]["wins"] == 1
    # ...and the spread is what shows it, since the means cannot.
    assert rows["spiky"]["sd"] > rows["steady"]["sd"]


def test_wins_are_counted_per_axis_not_over_the_whole_table(tmp_path):
    a = _cfg(tmp_path, "a")
    _write(
        a,
        [
            ("reduction", "umap", 0.4),
            ("reduction", "pca", 0.2),
            ("clusterer", "kmeans", 0.9),
            ("clusterer", "gmm", 0.1),
        ],
    )
    rows = {r["level"]: r["wins"] for r in aggregate.pooled([a]).rows}
    # The best reduction wins its own axis even though a clusterer level scored
    # higher anywhere in the table.
    assert rows == {"umap": 1, "pca": 0, "kmeans": 1, "gmm": 0}


def test_a_single_dataset_has_no_spread(tmp_path):
    a = _cfg(tmp_path, "a")
    _write(a, [("reduction", "umap", 0.4)])
    (row,) = aggregate.pooled([a]).rows
    assert row["sd"] != row["sd"]  # NaN: one value is not a spread of zero


def test_axis_order_follows_the_per_dataset_tables(tmp_path):
    a = _cfg(tmp_path, "a")
    _write(a, [("layer", "x", 0.1), ("reduction", "umap", 0.2), ("k", "8", 0.3)])
    assert aggregate.pooled([a]).axes == ["layer", "reduction", "k"]


def test_two_criteria_are_refused_rather_than_averaged(tmp_path):
    a = _cfg(tmp_path, "a", criterion="segmental_f1@0.75")
    b = _cfg(tmp_path, "b", criterion="segmental_f1@0.50")
    _write(a, [("reduction", "umap", 0.4)])
    _write(b, [("reduction", "umap", 0.9)])
    with pytest.raises(ValueError, match="criterion"):
        aggregate.pooled([a, b])


def test_the_axis_spread_averages_each_dataset_s_own_range(tmp_path):
    a, b = _cfg(tmp_path, "a"), _cfg(tmp_path, "b")
    _write(a, [("reduction", "umap", 0.4)], spread={"reduction": 0.2})
    _write(b, [("reduction", "umap", 0.6)], spread={"reduction": 0.4})
    assert aggregate.pooled([a, b]).spread_of("reduction") == pytest.approx(0.3)


def test_an_axis_with_no_recorded_spread_is_nan_not_zero(tmp_path):
    a = _cfg(tmp_path, "a")
    _write(a, [("reduction", "umap", 0.4)], spread={})
    value = aggregate.pooled([a]).spread_of("reduction")
    assert value != value


# ── Output shapes ────────────────────────────────────────────────────────────


def test_the_table_is_a_comment_when_there_is_nothing_to_show(tmp_path):
    out = aggregate.table(aggregate.pooled([_cfg(tmp_path, "a")]))
    assert out.startswith("%") and "cache_rows" in out


def test_the_table_names_the_tasks_it_pooled(tmp_path):
    a, b = _cfg(tmp_path, "a"), _cfg(tmp_path, "b")
    _write(a, [("reduction", "umap", 0.4)])
    _write(b, [("reduction", "umap", 0.6)])
    tex = aggregate.table(aggregate.pooled([a, b]))
    assert "2 tasks" in tex and "a, b" in tex
    assert "\\begin{tabular}" in tex and "\\bottomrule" in tex


def test_a_given_caption_and_label_win(tmp_path):
    a = _cfg(tmp_path, "a")
    _write(a, [("reduction", "umap", 0.4)])
    tex = aggregate.table(aggregate.pooled([a]), caption="Mine.", label="tab:mine")
    assert "Mine." in tex and "tab:mine" in tex
    assert "pooled over" not in tex


def test_dataframe_rows_carry_the_win_count_as_a_fraction(tmp_path):
    a, b = _cfg(tmp_path, "a"), _cfg(tmp_path, "b")
    _write(a, [("reduction", "umap", 0.4)])
    _write(b, [("reduction", "umap", 0.6)])
    (row,) = aggregate.dataframe_rows(aggregate.pooled([a, b]))
    assert row["best on"] == "2/2"
    assert row["F1"] == pytest.approx(0.5)


def test_the_text_report_says_what_is_missing(tmp_path, monkeypatch):
    a, b = _cfg(tmp_path, "a"), _cfg(tmp_path, "b")
    _write(a, [("reduction", "umap", 0.4)], spread={"reduction": 0.1})
    monkeypatch.setattr("behavior_graphs.ablation_study.config.available", lambda: ["a", "b"])
    monkeypatch.setattr(
        "behavior_graphs.ablation_study.config.load", lambda name: a if name == "a" else b
    )
    out = aggregate.run(a)
    assert "pooled over 1 datasets" in out
    assert "NOT cached" in out and "b" in out
