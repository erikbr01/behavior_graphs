"""YAML loading that uses libyaml when it is available.

``yaml.safe_load`` falls back to PyYAML's pure-Python parser, which is ~8×
slower than the libyaml-backed ``CSafeLoader``. That is irrelevant for a
single config file and very relevant for the model-selection index, which
parses one ``config.yaml`` per run — thousands of them on a sweep root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

try:  # libyaml bindings; present in every wheel built against it
    _Loader = yaml.CSafeLoader  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - pure-Python PyYAML build
    _Loader = yaml.SafeLoader


def load_yaml(text: str) -> Any:
    """``yaml.safe_load`` with the fastest available loader."""
    return yaml.load(text, Loader=_Loader)


def load_yaml_file(path: Path, default: Any | None = None) -> Any:
    """Parse a YAML file, returning ``default`` on any read/parse failure."""
    try:
        return load_yaml(Path(path).read_text()) or default
    except (OSError, yaml.YAMLError):
        return default
