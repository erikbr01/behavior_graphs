"""VLM-assigned labels for behavior graph nodes.

Labels are stored as ``<run_dir>/node_labels.json`` alongside ``graph.json``
and ``node_values.json``.  The format is:

    {
        "<node_id>": {
            "label": "reaching for handle",
            "description": "Robot arm extending toward the door handle …",
            "source": "claude-sonnet-4-6",
            "timestamp": "2026-07-01T12:34:56Z"
        },
        ...
    }

Node IDs are stored as strings in JSON (like ``graph.json``) and converted to
``int`` on load.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from behavior_graphs.core.io import atomic_write_json

_log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_node_labels(run_dir: Path) -> dict[int, dict]:
    """Load ``node_labels.json`` from *run_dir*, returning ``{}`` if unreadable.

    Unreadable, not just absent: the app writes this file as whatever user the
    container runs as, so a run annotated inside Docker has a root-owned file
    that a host-side process cannot open. Missing annotations degrade a graph to
    ``Behavior 3`` labels; an exception here takes the whole page down.
    """
    path = Path(run_dir) / "node_labels.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        _log.warning("could not read %s — showing the graph without annotations", path)
        return {}
    return {int(k): v for k, v in raw.items()}


def save_node_labels(run_dir: Path, labels: dict[int, dict]) -> None:
    """Atomically write *labels* to ``<run_dir>/node_labels.json``."""
    atomic_write_json(Path(run_dir) / "node_labels.json", {str(k): v for k, v in labels.items()})


def make_label_entry(
    label: str,
    description: str = "",
    source: str = "unknown",
    reasoning: str = "",
    confidence: float | None = None,
) -> dict:
    """Build a label entry dict with a fresh ``timestamp``."""
    entry: dict = {
        "label": label,
        "description": description,
        "source": source,
        "timestamp": _now_iso(),
    }
    if reasoning:
        entry["reasoning"] = reasoning
    if confidence is not None:
        entry["confidence"] = confidence
    return entry


def upsert_node_labels(run_dir: Path, entries: dict[int, dict]) -> None:
    """Merge many label entries into ``node_labels.json`` in one read/write.

    Existing labels for other nodes are preserved.
    """
    current = load_node_labels(run_dir)
    current.update(entries)
    save_node_labels(run_dir, current)


def upsert_node_label(
    run_dir: Path,
    node_id: int,
    label: str,
    description: str = "",
    source: str = "unknown",
    reasoning: str = "",
    confidence: float | None = None,
) -> None:
    """Merge a single node label into ``node_labels.json``.

    Existing labels for other nodes are preserved.  The ``timestamp`` field is
    always refreshed on upsert.
    """
    entry = make_label_entry(label, description, source, reasoning, confidence)
    upsert_node_labels(run_dir, {node_id: entry})
