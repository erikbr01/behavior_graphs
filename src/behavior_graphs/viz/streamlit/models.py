"""VLM model choices and labeler defaults for the Streamlit pages.

Lives in the UI layer rather than :mod:`behavior_graphs.vlm` on purpose: the
options list is a *presentation* concern read from ``configs/``, whereas
``behavior_graphs.vlm`` is the backend-adapter package and, per
:func:`behavior_graphs.core.io.resolve_data_path`, library code takes its
inputs as arguments instead of discovering them from the environment.

The options list is shared; a page supplies its own default — the labeling
modal takes the labeling stage's config — so tuning one doesn't move the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from behavior_graphs.core.yaml_io import load_yaml
from behavior_graphs.viz.streamlit.data import configs_dir

# Mirrors VLMLabeler's own signature defaults, used only when the config file
# is missing entirely — the UI should still work in a bare checkout.
LABELING_FALLBACKS: dict[str, Any] = {
    "model": "gemini-3.5-flash",
    "frames_per_episode": 5,
    "max_episodes_per_node": 4,
    "thinking_level": "high",
    "tile_px": 224,
}


def _read_config(path: Path) -> dict:
    """Parse a config file, distinguishing "absent" from "broken".

    Not :func:`~behavior_graphs.core.yaml_io.load_yaml_file`, which returns the
    default on *any* failure: a mis-indented ``models.yaml`` then looks exactly
    like a missing ``options:`` list, and the UI tells the operator to add a
    key that is plainly already there. A parse error is a real error.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc
    try:
        return load_yaml(text) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"{path} is not valid YAML: {exc}") from exc


def model_choices(default: str | None = None) -> tuple[list[str], str | None]:
    """``(options, default)`` from ``configs/vlm/models.yaml``.

    *default* overrides the file's own ``default:``; either way a default
    missing from ``options`` is prepended so it stays selectable.
    """
    cfg = _read_config(configs_dir() / "vlm" / "models.yaml")
    options = list(cfg.get("options") or [])
    chosen = default or cfg.get("default")
    if chosen and chosen not in options:
        options.insert(0, chosen)
    return options, chosen


def labeling_defaults() -> dict[str, Any]:
    """The node labeler's own defaults from ``configs/pipeline/labeling/vlm.yaml``.

    Shared with ``scripts/label_nodes.py`` and the pipeline stage so the
    annotation modal opens on the same settings a CLI run would use.
    """
    cfg = _read_config(configs_dir() / "pipeline" / "labeling" / "vlm.yaml")
    return {key: cfg.get(key, fallback) for key, fallback in LABELING_FALLBACKS.items()}
