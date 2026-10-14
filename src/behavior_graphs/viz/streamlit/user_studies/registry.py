"""Discovers study-variant configs and pairs each with its renderer.

Mirrors ``user_study/sessions.py``'s directory-of-yamls convention: variant
metadata (title, description, which existing study session it reuses, its
groups/pages and their setup) lives in ``configs/viz/study_variants/*.yaml``
so new copy/ordering tweaks don't require touching code. The Python renderer
for each variant id is wired explicitly in ``_render_dispatch`` below — a
yaml with no matching entry there is skipped (its renderer isn't built yet).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from behavior_graphs.viz.streamlit.user_studies.types import (
    StudyGroupSpec,
    StudyStageSpec,
    StudyVariantSpec,
)

REPO_ROOT = Path.cwd()


def variants_dir() -> Path:
    """Directory holding study-variant yamls ($STUDY_VARIANTS_DIR override)."""
    return Path(
        os.environ.get("STUDY_VARIANTS_DIR", "").strip()
        or (REPO_ROOT / "configs" / "viz" / "study_variants")
    )


def _render_dispatch() -> dict:
    """variant id -> render(variant, group) callable, one entry per implemented variant."""
    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
        render_three_stage,
    )

    return {
        "three_stage_v1": render_three_stage,
    }


def list_variants() -> list[StudyVariantSpec]:
    """All variants with both a config yaml and a wired renderer, in yaml filename order."""
    directory = variants_dir()
    dispatch = _render_dispatch()
    files = sorted(directory.glob("*.yaml")) if directory.is_dir() else []

    variants: list[StudyVariantSpec] = []
    for f in files:
        cfg = yaml.safe_load(f.read_text())
        variant_id = cfg["id"]
        render_fn = dispatch.get(variant_id)
        if render_fn is None:
            continue
        groups = [
            StudyGroupSpec(id=g["id"], label=g["label"], order=g["order"]) for g in cfg["groups"]
        ]
        stages = [
            StudyStageSpec(
                task=s["task"],
                label=s["label"],
                session_name=s["session"],
                question_set=s.get("question_set"),
                graph_params=s.get("graph", {}),
            )
            for s in cfg.get("stages", [])
        ]
        variants.append(
            StudyVariantSpec(
                id=variant_id,
                title=cfg["title"],
                description=cfg.get("description", ""),
                # A multi-task variant has one session per stage rather than one
                # for the variant; its `session_name` is the first stage's, so a
                # caller that reads only that field still names a real session
                # this variant uses instead of "".
                session_name=cfg.get("session") or (stages[0].session_name if stages else ""),
                groups=groups,
                render=render_fn,
                graph_params=cfg.get("graph", {}),
                question_set=cfg.get("question_set"),
                stages=stages,
            )
        )
    return variants


def get_variant(variant_id: str) -> StudyVariantSpec | None:
    return next((v for v in list_variants() if v.id == variant_id), None)
