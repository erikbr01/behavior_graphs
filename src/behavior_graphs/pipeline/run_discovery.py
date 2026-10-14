"""Locating pipeline run directories on disk.

Intentionally kept dependency-free (only stdlib ``os`` and ``pathlib``) so
that Streamlit apps, the MCP server, and other lightweight consumers can
discover runs without importing the pipeline executor (``run.py``), which
pulls in Hydra, NumPy, and the full stage graph.

Environment variables:
    BEHAVIOR_GRAPHS_RUN_DIR    Pin one run directory (skips any selector).
    BEHAVIOR_GRAPHS_RUNS_ROOT  Colon-separated roots to scan for run dirs
                             (default: ``./runs:./multirun``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

ENV_RUN_DIR = "BEHAVIOR_GRAPHS_RUN_DIR"
ENV_RUNS_ROOT = "BEHAVIOR_GRAPHS_RUNS_ROOT"
DEFAULT_ROOTS = ("runs", "multirun")


def is_run_dir(path: Path) -> bool:
    """A pipeline run dir always has window metadata + cluster labels.

    The window metadata may be *shared* rather than local — a sweep stores one
    copy per distinct window set and leaves each run a pointer to it (see
    :mod:`behavior_graphs.pipeline.shared_store`), so testing for a local
    ``windows.json`` alone would make every deduped run invisible to the apps.
    Labels stay local always: they are what makes a run a distinct run.
    """
    if not (path / "labels.npy").exists():
        return False
    if (path / "windows.json").exists():
        return True
    from behavior_graphs.pipeline.shared_store import read_pointers

    return "windows" in read_pointers(path)


def runs_roots() -> list[Path]:
    raw = os.environ.get(ENV_RUNS_ROOT, "").strip()
    roots = [Path(p) for p in raw.split(":") if p] if raw else [Path(r) for r in DEFAULT_ROOTS]
    return [r for r in roots if r.is_dir()]


def discover_all_job_dirs(sweep_dir: Path) -> list[Path]:
    """All completed job dirs (containing metrics.json) anywhere under sweep_dir.

    Prefers the ``jobs.json`` key list (O(jobs) stat calls) over walking the
    whole tree — job dirs are full artifact trees, so an ``rglob`` visits
    every features/array file of every run. Falls back to the tree walk for
    sweep dirs without a readable ``jobs.json``.
    """
    import json

    jobs_path = sweep_dir / "jobs.json"
    if jobs_path.exists():
        try:
            keys = json.loads(jobs_path.read_text())
        except Exception:
            keys = None
        if keys:
            return sorted(sweep_dir / k for k in keys if (sweep_dir / k / "metrics.json").exists())
    return sorted(p.parent for p in sweep_dir.rglob("metrics.json"))


def discover_run_dirs(roots: list[Path] | None = None, max_depth: int = 4) -> list[Path]:
    """All run dirs under the roots (hydra single runs and multirun jobs)."""
    found: list[Path] = []

    def _walk(d: Path, depth: int) -> None:
        if is_run_dir(d):
            found.append(d)
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir())
        except OSError as exc:
            # Every run beneath an unreadable dir disappears from the app and
            # from the sweep reports, which looks identical to "not swept yet".
            logging.getLogger(__name__).warning("cannot list %s (%s); skipping", d, exc)
            return
        for c in children:
            _walk(c, depth + 1)

    for root in roots if roots is not None else runs_roots():
        _walk(root, 0)
    return sorted(set(found))
