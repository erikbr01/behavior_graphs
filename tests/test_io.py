"""Tests for the shared filesystem/path helpers."""

from __future__ import annotations

from behavior_graphs.core.io import resolve_data_path


def test_resolve_data_path_reads_the_environment(tmp_path, monkeypatch):
    """The one sanctioned read, for entrypoints to call."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path))
    assert resolve_data_path() == str(tmp_path)


def test_blank_or_unset_data_path_is_none(monkeypatch):
    monkeypatch.setenv("DATA_PATH", "  ")
    assert resolve_data_path() is None
    monkeypatch.delenv("DATA_PATH")
    assert resolve_data_path() is None
