"""Shared dataset-map + embedding-layer resolution for sweep-style scripts.

Both ``scripts/run_sweep.py`` and ``scripts/outcome_classifier/run_outcome_classifier_sweep.py``
pick their dataset set the same way -- a ``configs/data/*.yaml`` group or an
inline ``datasets={...}`` map -- and both need to know what embedding
layer(s) each rollout dir has on disk. This is the one place that logic
lives, so the two scripts can't drift out of sync.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from behavior_graphs.core.io import DATA_GROUP_NON_DATASET_KEYS

log = logging.getLogger(__name__)


def resolve_datasets(cfg: DictConfig) -> dict[str, str]:
    # Inline datasets= override takes precedence over data= config group.
    for key in ("datasets", "data"):
        if cfg.get(key):
            # An entry is the rollout dir, or a mapping carrying it plus extras
            # (`display_name`, `cameras`) that only the app reads. Group-level
            # keys are not datasets and have no rollout dir of their own.
            dirs = {
                str(k): v.get("path") if isinstance(v, DictConfig | dict) else v
                for k, v in dict(cfg[key]).items()
                if str(k) not in DATA_GROUP_NON_DATASET_KEYS
            }
            if empty := [k for k, v in dirs.items() if not v]:
                # Otherwise `str(None)` sweeps a task dir named "None".
                raise ValueError(f"dataset(s) with no rollout dir: {', '.join(empty)}")  # noqa: TRY003
            return {k: str(v) for k, v in dirs.items()}
    raise ValueError("no data given: use data=<name> or datasets='{name: dir, ...}'")


def available_layers(dataset_paths: Sequence[str]) -> dict[str, set[str]]:
    """Return {path: set_of_available_layers} for each dataset path."""
    from behavior_graphs.envs.trajectory_dir import TrajectoryDirectory

    result: dict[str, set[str]] = {}
    for p in dataset_paths:
        if not Path(p).is_dir():
            log.warning("dataset dir %s does not exist; it has no layers to sweep", p)
            result[p] = set()
            continue
        try:
            result[p] = set(TrajectoryDirectory(Path(p)).embedding_layers())
        except Exception:
            # Silently drops the dataset from every layer pass, so say so.
            log.exception("could not read embedding layers of %s; treating as none", p)
            result[p] = set()
    return result


def detect_layers(dataset_paths: Sequence[str]) -> list[str | None]:
    """Layers to sweep when the caller named none: one pass per layer.

    Datasets with no ``policy_embeddings/`` are ignored so they can't empty the
    intersection. A common layer wins; otherwise sweep every layer present
    (``data=all`` mixes six) and let the per-layer filter below place each
    dataset — no extra jobs, since a dataset only appears in the passes whose
    layer it carries. ``[None]`` means nothing has embeddings.
    """
    per_dataset = available_layers(dataset_paths)
    with_layers = {p: ls for p, ls in per_dataset.items() if ls}
    if not with_layers:
        return [None]
    common = set.intersection(*with_layers.values())
    if common:
        return [sorted(common)[0]]
    union = sorted({ly for ls in with_layers.values() for ly in ls})
    log.warning(
        "datasets share no common embedding layer; sweeping each of %s and "
        "skipping datasets that lack one. Pass layers=[...] to choose.",
        union,
    )
    return list(union)


def selects_layerless_features(overrides: Sequence[str], grid_stages: Any = None) -> bool:
    """Whether a features stage that takes no ``layer`` is in play.

    ``features.layer=`` is a hard compose error against e.g. ``features=state``,
    and it lands in ``base_overrides`` outside the per-job try/except, so it
    aborts the whole run. Reads the grid's ``features`` stage group as well as
    the overrides — a grid can sweep feature extractors too.
    """
    for name in (grid_stages or {}).get("features") or {}:
        if name != "policy_embedding":
            return True
    for o in overrides:
        key, _, value = o.partition("=")
        if key.split("@")[0] in ("features", "pipeline/features") and value:
            return value != "policy_embedding"
    return False


def resolve_layer(rollout_dir: Path, requested: str | None) -> str | None:
    """Embedding layer to use for one dataset.

    Returns ``requested`` if set (validated against what's on disk); with no
    override, auto-detects only when the dataset has exactly one embedding
    layer -- ambiguous cases must be disambiguated explicitly via ``layer=``
    since guessing wrong would silently featurize the wrong signal.

    Deliberately a different policy from :func:`detect_layers` (which finds
    one layer common across *all* datasets in a sweep and skips mismatches):
    here every dataset is independent, so ambiguity is an error rather than
    something to route around. Prefer :func:`resolve_layer_if_applicable`
    over calling this directly -- a feature stage with no ``layer`` field at
    all (e.g. raw-state features) should never reach here in the first
    place, not receive a meaningless resolved value.
    """
    layers = available_layers([str(rollout_dir)])[str(rollout_dir)]
    if requested:
        if requested not in layers:
            raise ValueError(f"layer '{requested}' not in {rollout_dir} (available: {layers})")
        return requested
    if len(layers) == 1:
        return next(iter(layers))
    raise ValueError(
        f"{rollout_dir} has {len(layers)} embedding layer(s) {sorted(layers)}; pass "
        "layer=<name> to disambiguate"
    )


def resolve_layer_if_applicable(
    features_cfg: Any, rollout_dir: Path, requested: str | None
) -> str | None:
    """:func:`resolve_layer` if ``features_cfg`` has a ``layer`` field, else ``None``.

    Shared by every ``scripts/outcome_classifier/`` script that fits one
    dataset at a time (``run_outcome_classifier_sweep.py``,
    ``seen_vs_unseen_report.py``) -- a feature stage with no ``layer`` field
    (e.g. raw-state features) has nothing to resolve/validate.
    """
    if "layer" not in features_cfg:
        return None
    return resolve_layer(rollout_dir, requested)
