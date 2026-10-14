"""Tests for run-artifact rollout_dir recording and resolution.

Two halves of one convention: :meth:`RunArtifacts.save_metrics` records
``rollout_dir`` relative to the data root, and :func:`resolve_rollout_dir`
joins it back -- the path every reader (viz apps, MCP/chat tools, the VLM
labeler) goes through.
"""

from __future__ import annotations

import json
from pathlib import Path

from behavior_graphs.pipeline.artifacts import RunArtifacts, resolve_rollout_dir


def _saved(run_dir: Path, metrics: dict, data_path=None) -> dict:
    RunArtifacts(run_dir).save_metrics(metrics, data_path=data_path)
    return json.loads((run_dir / "metrics.json").read_text())


def test_save_metrics_relativizes_rollout_dir_under_the_data_root(tmp_path):
    """The recorded value is portable, whatever the caller passed in."""
    root = tmp_path / "data"
    rollouts = root / "rollouts" / "set_table"
    rollouts.mkdir(parents=True)
    saved = _saved(tmp_path / "run", {"rollout_dir": str(rollouts)}, data_path=root)
    assert saved["rollout_dir"] == "rollouts/set_table"


def test_save_metrics_leaves_a_relative_rollout_dir_alone(tmp_path):
    """Idempotent: a load/save round trip does not re-relativize."""
    root = tmp_path / "data"
    saved = _saved(tmp_path / "run", {"rollout_dir": "rollouts/set_table"}, data_path=root)
    assert saved["rollout_dir"] == "rollouts/set_table"


def test_save_metrics_keeps_a_rollout_dir_outside_the_data_root(tmp_path):
    """Not expressible portably ('..' differs per machine), so kept absolute."""
    outside = tmp_path / "elsewhere" / "set_table"
    outside.mkdir(parents=True)
    saved = _saved(tmp_path / "run", {"rollout_dir": str(outside)}, data_path=tmp_path / "data")
    assert saved["rollout_dir"] == str(outside)


def test_save_metrics_without_a_data_root_records_as_given(tmp_path):
    """No root supplied, so there is nothing to be relative to."""
    rollouts = tmp_path / "rollouts" / "set_table"
    saved = _saved(tmp_path / "run", {"rollout_dir": str(rollouts)})
    assert saved["rollout_dir"] == str(rollouts)


def test_save_metrics_ignores_the_environment(tmp_path, monkeypatch):
    """The data root arrives as an argument, never from the environment."""
    root = tmp_path / "data"
    (root / "rollouts" / "set_table").mkdir(parents=True)
    monkeypatch.setenv("DATA_PATH", str(root))
    recorded = str(root / "rollouts" / "set_table")
    assert _saved(tmp_path / "run", {"rollout_dir": recorded})["rollout_dir"] == recorded


def test_save_metrics_without_a_rollout_dir_is_untouched(tmp_path):
    saved = _saved(tmp_path / "run", {"silhouette": 0.4}, data_path=tmp_path)
    assert saved == {"silhouette": 0.4}


def test_relative_rollout_dir_joins_data_root(tmp_path):
    (tmp_path / "rollouts" / "set_table").mkdir(parents=True)
    resolved = resolve_rollout_dir("rollouts/set_table", tmp_path)
    assert resolved == tmp_path / "rollouts" / "set_table"


def test_empty_rollout_dir_is_none():
    assert resolve_rollout_dir(None, "/data") is None
    assert resolve_rollout_dir("", "/data") is None


def test_existing_absolute_rollout_dir_passes_through(tmp_path):
    """A pre-convention absolute path that exists is used as recorded."""
    recorded = tmp_path / "elsewhere" / "rollouts" / "set_table"
    recorded.mkdir(parents=True)
    root = tmp_path / "data"
    (root / "rollouts" / "set_table").mkdir(parents=True)
    assert resolve_rollout_dir(str(recorded), root) == recorded


def test_foreign_absolute_rollout_dir_is_not_rebased(tmp_path):
    """A path from another machine is reported as recorded, not re-rooted.

    Absolute values are never re-anchored onto *data_path*, even when a local
    counterpart exists under ``rollouts/``: the recorded path is what the run
    claims, and guessing otherwise cannot distinguish a foreign path from a
    real local one. Normalizing at write time is the fix (see
    :meth:`RunArtifacts.save_metrics`).
    """
    root = tmp_path / "data"
    (root / "rollouts" / "set_table").mkdir(parents=True)
    recorded = tmp_path / "other-host" / "data" / "rollouts" / "set_table"
    assert resolve_rollout_dir(str(recorded), root) == recorded


def test_missing_absolute_rollout_dir_reports_the_recorded_path(tmp_path):
    """With no local counterpart, callers see the path the run recorded."""
    recorded = tmp_path / "other-host" / "data" / "rollouts" / "nonexistent"
    assert resolve_rollout_dir(str(recorded), tmp_path) == Path(recorded)


def test_absolute_path_without_rollouts_component_passes_through(tmp_path):
    """Nothing to anchor on: the recorded path is returned unchanged."""
    recorded = "/scratch/set_table"
    assert resolve_rollout_dir(recorded, tmp_path) == Path(recorded)
