"""Storage and loading for VLM-generated graph analysis reports.

Analysis reports are written to ``<run_dir>/graph_analysis.json`` by the
VLMLabeler and loaded by the demo app for display.

Schema::

    {
        "task_summary": "...",
        "success_modes": [
            {"path_str": "B2 → B3 → SUCCESS", "probability": 0.24,
             "node_ids": [2, 3, -1], "description": "..."},
            ...
        ],
        "failure_modes": [
            {"description": "...", "key_node_ids": [0]},
            ...
        ],
        "similar_pairs": [
            {"node_a": 0, "node_b": 2, "distinction": "..."},
            ...
        ],
        "overall_assessment": "...",
        "model": "gemini-3.5-flash",
        "timestamp": "2026-07-03T..."
    }
"""

from __future__ import annotations

import json
from pathlib import Path

from behavior_graphs.core.io import atomic_write_json

_FILENAME = "graph_analysis.json"


def load_graph_analysis(run_dir: Path) -> dict | None:
    """Load graph_analysis.json from run_dir; return None if absent."""
    path = Path(run_dir) / _FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_graph_analysis(run_dir: Path, analysis: dict) -> None:
    """Atomically write analysis to graph_analysis.json in run_dir."""
    atomic_write_json(Path(run_dir) / _FILENAME, analysis)
