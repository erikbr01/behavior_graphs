"""Flatten pipeline run directories into one DataFrame for analysis.

Every run dir is self-describing (``config.yaml`` + ``metrics.json``), so a
sweep is just a set of run dirs. One row per run:

- hyperparameter columns: a short class name per stage group (``clustering``,
  ``reduction``, ...) plus ``<group>.<param>`` for every constructor argument
  (``clustering.n_clusters``, ``windowing.width``, ...);
- ``k``: the unified model-complexity axis (``n_clusters`` for k-means / KNN,
  ``n_components`` for GMM; NaN for density clusterers where K is emergent);
- one column per registry metric (see
  :data:`behavior_graphs.model_selection.scoring.METRICS`).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavior_graphs.core.stages import STAGE_GROUPS
from behavior_graphs.core.yaml_io import load_yaml
from behavior_graphs.model_selection import scoring
from behavior_graphs.model_selection.scoring import METRICS, metric_value

#: Per-group config keys that are implementation detail, not science.
_SKIP_PARAMS = {"_target_", "device"}


def short_target(stage_cfg: dict[str, Any] | None, default: str = "?") -> str:
    tgt = str((stage_cfg or {}).get("_target_", default))
    return tgt.rsplit(".", 1)[-1].removesuffix("Stage").removesuffix("Reducer")


#: Short names for the feature extractors whose class name alone does not read
#: as a representation. ``StateFeatures``/``ConcatFeatures`` fall out of the
#: generic rule below, so only the irregular one is listed.
_EXTRACTOR_LABELS = {"StateActionFeatures": "state+action"}


def representation_label(features: dict[str, Any] | None) -> str | None:
    """What a run's features *are*, in one sidebar-sized string.

    The representation is the thing a behavior graph is a graph *of*, and it
    used to be identifiable by one column: every run's features came from a
    policy, so ``features.layer`` named it. A run built on a frozen vision
    backbone over the episode video has no layer at all, so the axis needs a
    name that spans both -- the layer for ``PolicyEmbeddings`` (``cond_proj``),
    the backbone and pooling for ``VisionBackboneFeatures``
    (``dinov2-base patch2x2``, ``dinov2-base mean``), since the model and
    whether the patch grid was kept or averaged away are what distinguish two
    runs over the same video. Views are appended when a run concatenates more
    than one camera, because that doubles the feature width and shows up
    nowhere else.

    Lives here, beside :func:`unified_k` -- which is the same shape of thing,
    one user-facing axis across implementation families -- and is called by
    :func:`flatten_config` on the *raw* config, so the column is computed
    before pandas can widen ``features.grid`` to a float or
    :func:`flatten_config` can JSON-dump the camera list. Both consumers (the
    runs index and the app's picker) then read one precomputed value instead
    of reconstructing it from typed columns.

    Returns None for a config this cannot read, which keeps such a run out of
    the picker's value set instead of filing it under a wrong name.
    """
    features = features or {}
    if not features.get("_target_"):
        return None
    cls = short_target(features)
    if cls == "PolicyEmbeddings":
        return str(features.get("layer") or "") or "policy"
    if cls == "VisionBackboneFeatures":
        model = str(features.get("model_name") or "?").rsplit("/", 1)[-1]
        pooling = str(features.get("pooling") or "grid")
        if pooling == "grid":
            side = int(features.get("grid", 2))
            # grid 1 *is* mean pooling; "patch1x1" would offer a distinction
            # that does not exist.
            detail = "mean" if side == 1 else f"patch{side}x{side}"
        else:
            detail = pooling
        cameras = features.get("cameras") or []
        views = f" x{len(cameras)}views" if len(cameras) > 1 else ""
        return f"{model} {detail}{views}"
    if cls in _EXTRACTOR_LABELS:
        return _EXTRACTOR_LABELS[cls]
    return cls.replace("Features", "").lower() or None


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Hyperparameter columns for one resolved pipeline config."""
    row: dict[str, Any] = {}
    for group in STAGE_GROUPS:
        sub = cfg.get(group) or {}
        row[group] = short_target(sub)
        for key, value in sub.items():
            if key in _SKIP_PARAMS:
                continue
            if isinstance(value, dict | list):
                value = json.dumps(value, sort_keys=True)
            row[f"{group}.{key}"] = value
    row["k"] = unified_k(cfg)
    # From the raw config, so the label never sees a pandas-widened float or a
    # JSON-dumped list -- see representation_label.
    row["representation"] = representation_label(cfg.get("features"))
    return row


def unified_k(cfg: dict[str, Any]) -> int | None:
    """The user-facing complexity choice K, across clusterer families."""
    clu = cfg.get("clustering") or {}
    for key in ("n_clusters", "n_components"):
        if clu.get(key) is not None:
            return int(clu[key])
    return None


def load_run_row(run_dir: Path) -> dict[str, Any]:
    """Config + metric columns for one run dir (missing files -> NaNs)."""
    run_dir = Path(run_dir)
    try:
        cfg = load_yaml((run_dir / "config.yaml").read_text()) or {}
    except Exception:
        cfg = {}
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text())
    except Exception:
        metrics = {}
    metrics = dict(metrics)

    row = {
        "run": run_dir.name,
        "path": str(run_dir),
        "rollout_dir": metrics.get("rollout_dir"),
    }
    row.update(flatten_config(cfg))
    for key, spec in METRICS.items():
        row[key] = metric_value(metrics, spec.path)
    obj = metrics.get("objective") or {}
    if isinstance(obj.get("score"), int | float):
        row["objective"] = float(obj["score"])
    return row


def load_runs_index(run_dirs: Iterable[Path]) -> pd.DataFrame:
    """One row per run dir; metric columns always present (NaN when absent)."""
    rows: list[dict[str, Any]] = [load_run_row(d) for d in run_dirs]
    df = pd.DataFrame(rows)
    for key in METRICS:
        if key not in df.columns:
            df[key] = np.nan
    if "k" in df.columns:
        df["k"] = pd.to_numeric(df["k"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def index_schema_version() -> str:
    """Hash of the code that decides the runs-index schema.

    Streamlit's cache key covers only the *calling* function's own source, and
    the deployed stack bind-mounts ``src/`` so an edit hot-reloads the script
    without dropping caches. Adding a metric to :data:`scoring.METRICS` or
    renaming a column here would therefore replay a DataFrame built by the old
    code, and the page would die on a KeyError for a column the cached frame
    never had. Pass this as an argument to any cache over the index.
    """
    h = hashlib.blake2b(digest_size=8)
    for mod in (sys.modules[__name__], scoring):
        try:
            h.update(inspect.getsource(mod).encode())
        except (OSError, TypeError):  # pragma: no cover - source not available
            h.update(b"?")
    return h.hexdigest()


def hyperparameter_columns(df: pd.DataFrame) -> list[str]:
    """All hyperparameter columns present in the index, in stable order."""
    cols: list[str] = []
    for group in STAGE_GROUPS:
        if group in df.columns:
            cols.append(group)
        cols.extend(sorted(c for c in df.columns if c.startswith(f"{group}.")))
    return cols


def varying_columns(df: pd.DataFrame, cols: list[str] | None = None) -> list[str]:
    """The subset of hyperparameter columns that actually vary in this sweep."""
    cols = cols if cols is not None else hyperparameter_columns(df)
    return [c for c in cols if df[c].astype(str).nunique(dropna=False) > 1]
