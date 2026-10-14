"""Run-dir artifacts must be readable by users other than the writer.

The apps run in Docker. When that container is root (the default before
APP_UID/APP_GID landed in deploy/.env), anything it writes into the
bind-mounted sweeps tree is root-owned — and ``mkstemp``, which every atomic
write goes through, creates ``0600`` regardless of umask, so the host user
could not even read it. 27 run dirs ended up with annotations that
``load_node_labels`` had to swallow a ``PermissionError`` for.

Ownership is a deployment concern (deploy/docker-compose.yml ``user:``); the
mode is ours, and this pins it.
"""

from __future__ import annotations

import stat
from pathlib import Path

from behavior_graphs.core.graph_analysis import save_graph_analysis
from behavior_graphs.core.io import atomic_write_json, atomic_write_text
from behavior_graphs.core.node_labels import make_label_entry, save_node_labels


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_atomic_writes_are_group_and_world_readable(tmp_path: Path) -> None:
    atomic_write_text(tmp_path / "a.txt", "hello")
    atomic_write_json(tmp_path / "b.json", {"k": 1})

    for name in ("a.txt", "b.json"):
        mode = _mode(tmp_path / name)
        assert mode & stat.S_IRGRP, f"{name} is not group-readable ({mode:o})"
        assert mode & stat.S_IROTH, f"{name} is not world-readable ({mode:o})"


def test_run_dir_artifacts_inherit_that_mode(tmp_path: Path) -> None:
    """The two writers that land next to graph.json in a shared sweeps tree."""
    save_node_labels(tmp_path, {1: make_label_entry("reaching for handle")})
    save_graph_analysis(tmp_path, {"task_summary": "picks up the lid"})

    for name in ("node_labels.json", "graph_analysis.json"):
        mode = _mode(tmp_path / name)
        assert mode & stat.S_IROTH, f"{name} is not world-readable ({mode:o})"


def test_overwriting_keeps_the_mode(tmp_path: Path) -> None:
    """``Path.replace`` carries the temp file's mode onto the target."""
    save_node_labels(tmp_path, {1: make_label_entry("first")})
    save_node_labels(tmp_path, {1: make_label_entry("second"), 2: make_label_entry("more")})

    assert _mode(tmp_path / "node_labels.json") & stat.S_IROTH
