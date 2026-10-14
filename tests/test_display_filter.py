"""Tests for the app's dataset display filter.

The demo app discovers runs by scanning the sweep dir, so every dataset that
has ever been swept shows up in the selector. ``BEHAVIOR_GRAPHS_DISPLAY_DATA``
names a ``configs/data/`` group to restrict that list to.
"""

from __future__ import annotations

import logging

import pytest

from behavior_graphs.viz.streamlit.data import _data_group_names, display_dataset_names


def _group(tmp_path, name: str, body: str):
    d = tmp_path / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.yaml").write_text(body)
    return tmp_path


def test_group_names_are_the_dataset_keys(tmp_path):
    cfgs = _group(tmp_path, "small", "lift: /data/rollouts/lift\ncan: /data/rollouts/can\n")
    assert _data_group_names("small", cfgs) == {"lift", "can"}


def test_mapping_form_entries_are_still_dataset_names(tmp_path):
    """A dataset carrying `display_name` must not drop out of the allowlist."""
    cfgs = _group(
        tmp_path,
        "small",
        "lift:\n  path: /data/rollouts/lift\n  display_name: Lift\ncan: /data/rollouts/can\n",
    )
    assert _data_group_names("small", cfgs) == {"lift", "can"}


def test_entry_without_a_path_is_listed_and_logged(tmp_path, caplog):
    """Dropping it would hide the dataset; raising would take down the page."""
    cfgs = _group(tmp_path, "small", "lift:\n  display_name: Lift\ncan: /data/rollouts/can\n")
    with caplog.at_level(logging.WARNING):
        assert _data_group_names("small", cfgs) == {"lift", "can"}
    assert "'lift'" in caplog.text and "rollout dir" in caplog.text


def test_defaults_are_composed(tmp_path):
    """A group's `defaults:` list pulls in the parent groups' datasets."""
    cfgs = _group(tmp_path, "parent", "lift: /data/rollouts/lift\n")
    _group(cfgs, "child", "defaults:\n  - parent\n  - _self_\ncan: /data/rollouts/can\n")
    assert _data_group_names("child", cfgs) == {"lift", "can"}


def test_self_is_not_treated_as_a_group(tmp_path):
    cfgs = _group(tmp_path, "only", "defaults:\n  - _self_\nlift: /data/rollouts/lift\n")
    assert _data_group_names("only", cfgs) == {"lift"}


def test_unresolved_interpolations_do_not_matter(tmp_path):
    """Only keys are read, so ``${oc.env:DATA_PATH}`` needs no resolution."""
    cfgs = _group(tmp_path, "env", "lift: ${oc.env:DATA_PATH}/rollouts/lift\n")
    assert _data_group_names("env", cfgs) == {"lift"}


def test_defaults_cycle_terminates(tmp_path):
    """A → B → A would otherwise recurse until the stack blows."""
    cfgs = _group(tmp_path, "a", "defaults:\n  - b\nlift: /data/rollouts/lift\n")
    _group(cfgs, "b", "defaults:\n  - a\ncan: /data/rollouts/can\n")
    assert _data_group_names("a", cfgs) == {"lift", "can"}


def test_missing_group_is_an_error(tmp_path):
    """Better a loud failure than silently showing every dataset."""
    with pytest.raises(FileNotFoundError):
        _data_group_names("nope", tmp_path)


def test_unset_env_means_no_filtering(monkeypatch):
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    assert display_dataset_names() is None


def test_blank_env_means_no_filtering(monkeypatch):
    """An empty compose variable must not filter everything away."""
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "  ")
    assert display_dataset_names() is None


def test_configs_dir_prefers_the_packaged_configs_over_cwd(tmp_path, monkeypatch):
    """Resolution must not depend on where Streamlit was launched from."""
    from behavior_graphs.viz.streamlit.data import configs_dir

    monkeypatch.chdir(tmp_path)
    resolved = configs_dir()
    assert resolved.is_dir() and (resolved / "data" / "visualize.yaml").is_file()


def test_cwd_is_the_fallback_when_no_packaged_configs(tmp_path, monkeypatch):
    _group(tmp_path / "configs", "shown", "lift: /data/rollouts/lift\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.data.Path.cwd", lambda: tmp_path, raising=False
    )
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "shown")
    # packaged configs/ exists in this checkout, so assert via the helper directly
    assert _data_group_names("shown", tmp_path / "configs") == {"lift"}


def test_repo_visualize_group_covers_the_canonical_droid_set():
    """The shipped groups compose: `visualize` is a superset of `droid`."""
    from pathlib import Path

    cfgs = Path(__file__).resolve().parents[1] / "configs"
    assert _data_group_names("droid", cfgs) <= _data_group_names("visualize", cfgs)
    assert _data_group_names("all", cfgs) <= _data_group_names("visualize", cfgs)


def test_bad_group_is_loud_rather_than_falling_back(monkeypatch):
    """Falling back to unfiltered is indistinguishable from a working filter.

    A deployment quietly serving a dataset list nobody asked for is worse than
    a page that says why it cannot resolve the group.
    """
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "typo-not-a-group")
    with pytest.raises(FileNotFoundError, match="no such data config group"):
        display_dataset_names()


def test_group_naming_no_datasets_is_loud(monkeypatch, tmp_path):
    """An allowlist of nothing would hide every run in the app."""
    _group(tmp_path / "configs", "empty", "defaults:\n  - _self_\n")
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.data.configs_dir", lambda: tmp_path / "configs"
    )
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "empty")
    with pytest.raises(ValueError, match="resolves to no datasets"):
        display_dataset_names()


def test_group_is_found_from_outside_the_checkout(tmp_path, monkeypatch):
    """Launching Streamlit from another cwd must still resolve the group."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "visualize")
    assert (display_dataset_names() or set()) >= {"cooking_baseline_dit_200"}


def test_filter_key_changes_with_the_group(monkeypatch):
    """The key feeds the cached index, so it must track the allowlist."""
    from behavior_graphs.viz.streamlit.data import display_filter_key

    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    unfiltered = display_filter_key()
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "droid")
    droid = display_filter_key()
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "visualize")
    assert unfiltered == "" and droid and droid != display_filter_key()


def test_display_run_dirs_applies_the_allowlist(tmp_path, monkeypatch):
    from behavior_graphs.viz.streamlit.data import display_run_dirs

    for task in ("keep_me", "hide_me"):
        d = tmp_path / task / "abc123"
        d.mkdir(parents=True)
        (d / "windows.json").write_text("{}")
        (d / "labels.npy").write_bytes(b"")
    cfgs = tmp_path / "configs" / "data"
    cfgs.mkdir(parents=True)
    (cfgs / "shown.yaml").write_text("keep_me: /data/rollouts/keep_me\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.data.configs_dir", lambda: tmp_path / "configs"
    )
    monkeypatch.setenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", "shown")
    names = {p.parent.name for p in display_run_dirs([tmp_path])}
    assert names == {"keep_me"}


def test_every_viz_discovery_site_uses_the_filtered_helper():
    """Findings only bite where the filter isn't applied — keep all sites on it."""
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    for rel in (
        "src/behavior_graphs/viz/streamlit/components/selection_panel.py",
        "apps/demo/_pages/sweep_analysis.py",
    ):
        assert "discover_run_dirs" not in (root / rel).read_text(), rel


def test_the_picker_can_tell_two_sae_runs_apart_by_temporal_coef(tmp_path, monkeypatch):
    """They differ in nothing else, so without this the picker offers one option.

    ``_run_index`` is the picker's whole view of a run: a param missing from a row
    cannot be selected on, and the first matching run silently wins.
    """
    from behavior_graphs.viz.streamlit.data import PARAM_LABELS, _run_index

    for name, coef in (("plain", 0), ("slow", 1)):
        d = tmp_path / "set_table" / name
        d.mkdir(parents=True)
        (d / "windows.json").write_text("{}")
        (d / "labels.npy").write_bytes(b"")
        (d / "config.yaml").write_text(
            "clustering: {n_clusters: 5}\nwindowing: {width: 5, stride: 2}\n"
            "reduction:\n  _target_: behavior_graphs.stages.reduction.sae.SAEReduction\n"
            f"  n_components: 400\n  temporal_coef: {coef}\n"
        )
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    _run_index.clear()
    rows = _run_index(str(tmp_path))

    assert {r["temporal_coef"] for r in rows} == {0, 1}
    assert "temporal_coef" in dict(PARAM_LABELS)


def test_the_picker_carries_every_param_a_reducer_can_differ_by(tmp_path, monkeypatch):
    """One list, checked against the configs, rather than one bug fix per knob.

    Each of these was added after runs that differed only in it turned out to be
    unselectable — the picker showed whichever it found first. Anything a grid
    can vary and the app cannot select on repeats that.
    """
    from behavior_graphs.viz.streamlit.data import PARAM_LABELS, QUERY_PARAM_NAMES, _run_index

    variants = {
        "plain": "",
        "slow": "  temporal_coef: 1\n",
        "contrastive": "  contrastive_coef: 1\n",
        "beta2": "  beta: 2.0\n",
    }
    clusterers = {
        "kmeans": "behavior_graphs.stages.clustering.kmeans.KMeansClustering",
        "hsmm": "behavior_graphs.stages.clustering.hsmm.HSMMClustering",
    }
    for name, extra in variants.items():
        d = tmp_path / "set_table" / name
        d.mkdir(parents=True)
        (d / "windows.json").write_text("{}")
        (d / "labels.npy").write_bytes(b"")
        (d / "config.yaml").write_text(
            "clustering: {n_clusters: 5}\nwindowing: {width: 5, stride: 2}\n"
            "reduction:\n  _target_: behavior_graphs.stages.reduction.vae.VAEReduction\n"
            "  n_components: 100\n" + extra
        )
    for name, target in clusterers.items():
        d = tmp_path / "set_table" / f"clu_{name}"
        d.mkdir(parents=True)
        (d / "windows.json").write_text("{}")
        (d / "labels.npy").write_bytes(b"")
        (d / "config.yaml").write_text(
            f"clustering:\n  _target_: {target}\n  n_clusters: 5\n"
            "windowing: {width: 5, stride: 2}\n"
            "reduction:\n  _target_: behavior_graphs.stages.reduction.vae.VAEReduction\n"
            "  n_components: 100\n"
        )
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    _run_index.clear()
    rows = _run_index(str(tmp_path))

    for param in ("temporal_coef", "contrastive_coef", "beta", "clustering"):
        assert param in dict(PARAM_LABELS), param
        assert param in QUERY_PARAM_NAMES, param
    # Four runs, four distinct selections: no two collapse onto one option.
    keys = {
        tuple(r.get(p) for p in ("temporal_coef", "contrastive_coef", "beta", "clustering"))
        for r in rows
    }
    assert len(keys) == len(variants) + len(clusterers), keys


def _run(tmp_path, name: str, graph: str = "", clustering: str = "kmeans"):
    d = tmp_path / "set_table" / name
    d.mkdir(parents=True)
    (d / "windows.json").write_text("{}")
    (d / "labels.npy").write_bytes(b"")
    (d / "config.yaml").write_text(
        "clustering: {n_clusters: 5}\nwindowing: {width: 5, stride: 2}\n"
        "reduction:\n  _target_: behavior_graphs.stages.reduction.umap.UMAPReduction\n"
        "  n_components: 10\n" + graph
    )
    return d


def test_the_picker_can_tell_a_smoothed_run_from_its_twin(tmp_path, monkeypatch):
    """`grid=smoothing` re-runs every standard cell with the sticky decoder on.

    The two differ in nothing the picker read before, so the smoothed twin was
    unreachable — and which of the two you got depended on directory order.
    """
    from behavior_graphs.viz.streamlit.data import PARAM_LABELS, QUERY_PARAM_NAMES, _run_index

    _run(tmp_path, "plain")
    _run(tmp_path, "sticky2", "graph:\n  smoothing: sticky\n  sticky_lambda: 2.0\n")
    _run(tmp_path, "sticky10", "graph:\n  smoothing: sticky\n  sticky_lambda: 10.0\n")
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    _run_index.clear()
    rows = _run_index(str(tmp_path))

    for param in ("smoothing", "sticky_lambda"):
        assert param in dict(PARAM_LABELS), param
        assert param in QUERY_PARAM_NAMES, param
    assert {(r["smoothing"], r["sticky_lambda"]) for r in rows} == {
        ("none", None),
        ("sticky", 2.0),
        ("sticky", 10.0),
    }


def test_a_run_swept_before_the_smoothing_axis_is_the_unsmoothed_arm(tmp_path, monkeypatch):
    """Its config has no `graph:` block at all, but it is not unknown-smoothing.

    Reading that as None would drop it from the selector's value set, and with
    the whole pre-smoothing sweep on one side of the comparison the axis would
    look constant and never be offered.
    """
    from behavior_graphs.viz.streamlit.data import _run_index

    _run(tmp_path, "old")
    _run(tmp_path, "new", "graph:\n  smoothing: none\n  sticky_lambda: 2.0\n")
    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    _run_index.clear()

    rows = _run_index(str(tmp_path))

    # Same arm, and neither carries a lambda it never used.
    assert {r["smoothing"] for r in rows} == {"none"}
    assert {r["sticky_lambda"] for r in rows} == {None}


def test_run_discovery_is_not_re_walked_on_a_timer():
    """A TTL here is a stall that lands on one widget interaction a minute.

    The walk reads a directory per run dir (6-8s at 43k runs), and expiring the
    metrics fingerprint alongside it also re-keys the runs index, so the cost is
    the walk *plus* a full re-index. `Refresh runs` (#78) is the intended way to
    pick up new runs, and it clears both.
    """
    from behavior_graphs.viz.streamlit.data import (
        _discover_run_dirs_cached,
        _metrics_fingerprint_cached,
    )

    for fn in (_discover_run_dirs_cached, _metrics_fingerprint_cached):
        assert fn._info.ttl is None, fn._info.func.__name__


def test_refresh_clears_every_cache_that_lost_its_ttl(monkeypatch):
    """Without a TTL, a cache the refresh button forgets can never be rebuilt."""
    import behavior_graphs.viz.streamlit.data as data

    cleared: set[str] = set()

    class _Stub:
        def __init__(self, name: str):
            self.name = name

        def clear(self):
            cleared.add(self.name)

    for name in (
        "_discover_run_dirs_cached",
        "_display_run_dirs_cached",
        "_metrics_fingerprint_cached",
        "_runs_index_cached",
        "_varying_params",
    ):
        monkeypatch.setattr(data, name, _Stub(name))
    monkeypatch.setattr(data._run_index, "clear", lambda: cleared.add("_run_index"))
    monkeypatch.setattr(
        "behavior_graphs.viz.streamlit.components.selection_panel.clear_best_per_k_cache",
        lambda: cleared.add("best_per_k"),
    )

    data.refresh_run_index()

    assert cleared == {
        "_discover_run_dirs_cached",
        "_display_run_dirs_cached",
        "_metrics_fingerprint_cached",
        "_runs_index_cached",
        "_varying_params",
        "_run_index",
        "best_per_k",
    }


def test_the_run_list_is_a_snapshot_until_refreshed(tmp_path, monkeypatch):
    """The other half of dropping the TTL, stated as behaviour.

    Discovery and the allowlist filter together cost seconds on a large sweep
    root, so they are memoized for the life of the session. That means a run
    written while the page is open is invisible until "Refresh runs" — which is
    the trade this makes, and the reason that button has to clear both.
    """
    from behavior_graphs.viz.streamlit.data import display_run_dirs, refresh_run_index

    def _write(name: str):
        d = tmp_path / "task" / name
        d.mkdir(parents=True)
        (d / "windows.json").write_text("{}")
        (d / "labels.npy").write_bytes(b"")

    monkeypatch.delenv("BEHAVIOR_GRAPHS_DISPLAY_DATA", raising=False)
    _write("first")
    refresh_run_index()
    assert {p.name for p in display_run_dirs([tmp_path])} == {"first"}

    _write("second")
    assert {p.name for p in display_run_dirs([tmp_path])} == {"first"}

    refresh_run_index()
    assert {p.name for p in display_run_dirs([tmp_path])} == {"first", "second"}
