"""Small filesystem and path helpers shared across the package."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

#: Keys of a ``configs/data/<group>.yaml`` that are *not* dataset names.
#:
#: ``defaults:`` is Hydra's include list; ``cameras:`` is the group-wide display
#: geometry and default view the apps read (see
#: :func:`behavior_graphs.viz.streamlit.data.camera_transforms` and
#: :func:`~behavior_graphs.viz.streamlit.data.primary_camera`). Two independent
#: readers walk these files — the sweep's dataset resolution
#: (``scripts/run_sweep.py``) and the app's group reader — so the set lives here
#: rather than in either of them: when only one learned to skip ``cameras:``, the
#: other read it as a dataset with no rollout dir and every ``data=`` group
#: failed to launch. Anything added here must be added *here*, not in a caller.
DATA_GROUP_NON_DATASET_KEYS = frozenset({"defaults", "cameras"})


def resolve_data_path() -> str | None:
    """Read ``$DATA_PATH``, the repo-wide data root, for an entrypoint to
    pass down explicitly.

    The one sanctioned read: a run's recorded ``rollout_dir`` is stored
    relative to this root (see
    :meth:`behavior_graphs.pipeline.artifacts.RunArtifacts.save_metrics`), and
    library code takes it as an argument rather than reaching for the
    environment mid-call.
    """
    return os.environ.get("DATA_PATH", "").strip() or None


#: Mode for files written by :func:`atomic_write_text`.
#:
#: ``mkstemp`` (under ``NamedTemporaryFile``) always creates ``0600`` and
#: ignores the umask, and ``Path.replace`` carries that mode onto the final
#: file — so without this every atomically-written artifact is readable only by
#: the writing user. That bit the apps: the demo container runs as root and
#: writes ``node_labels.json`` into the shared sweeps tree, leaving 27 run dirs
#: with annotations the host user could neither read nor copy. Run artifacts are
#: shared, non-secret data; write them like the rest of the run dir.
_ARTIFACT_MODE = 0o644


def atomic_write_text(path: Path, text: str) -> None:
    """Write *text* to *path* via a same-directory temp file + atomic replace.

    The result is world-readable (:data:`_ARTIFACT_MODE`) regardless of which
    user — or container — did the writing.
    """
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False) as f:
        f.write(text)
        tmp = Path(f.name)
    try:
        tmp.chmod(_ARTIFACT_MODE)
    except OSError:  # e.g. a filesystem that does not carry modes
        pass
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> None:
    """Atomically write *obj* as JSON to *path*."""
    atomic_write_text(path, json.dumps(obj, indent=indent))
