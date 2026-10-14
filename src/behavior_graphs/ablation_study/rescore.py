"""Re-score every completed sweep job under the current metric, without re-running it.

Everything the metric needs is already on disk: each job dir carries
``labels.npy`` and ``config.yaml``. Measured at ~0.8s a job, so well under an
hour for a 5,176-job grid against ~20h to recompute the jobs themselves. That is
the point of having fixed the metric rather than the pipeline: changing what is
measured is cheap, changing what is computed is not.

Two costs dominate and both are handled by :class:`~behavior_graphs.ablation_study.scoring.Scorer`
plus the loop order here:

*   Windowing. A grid spans a couple of dozen distinct ``(layer, width, stride)``
    combinations, so the ``Scorer``'s memoisation turns 5,176 extractions into 24.
*   Segmental extents, which are tolerance-independent. They are asked for at
    the primary tolerance only, which halves the per-job work.

This replaces ``scripts/ablation_study/rescore_sweep.py``, which duplicated the
scoring logic and defaulted ``--tolerance`` to bottles' 0.5273. A dataset whose
segments are 2.2x shorter would have been scored under bottles' tolerance
without complaint; here the tolerance comes from the config, which derives it
from the dataset's own fit half when it is left null.
"""

from __future__ import annotations

import json
import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import yaml

from behavior_graphs.ablation_study.config import DatasetConfig
from behavior_graphs.ablation_study.scoring import Scorer
from behavior_graphs.envs.trajectory_dir import RolloutDirectory
from behavior_graphs.pipeline.folds import resolve_split
from behavior_graphs.stages.evaluation._boundary_metrics import (
    build_ground_truth,
    label_agreement,
    score_boundaries,
)
from behavior_graphs.stages.evaluation.subtask_boundaries import predicted_boundaries

#: Both ground-truth sets are recorded. ``task`` is what the study reports;
#: ``all`` includes the non-task segments and is kept so the choice stays
#: auditable rather than baked into the artefact.
GT_SETS = ("task", "all")

#: The loose tolerance is the primary one doubled. Reported as a robustness
#: check on the boundary metric, which is the only metric a tolerance touches.
LOOSE_MULTIPLE = 2.0


def job_dirs(cfg: DatasetConfig) -> list[pathlib.Path]:
    """Every completed job of this dataset's grid, and nothing else.

    Scoped to ``cfg.sweep_dir / cfg.key`` rather than globbed as ``*/*``. The
    sweep directory holds more than the grid: ``pre_clean_policy*`` quarantines
    the arms of a policy that trained on the evaluation episodes, and ``select``
    holds a confirmation run. Under bottles that is 3,448 leaky jobs and 128
    confirmation jobs against the grid's 1,979, so a wildcard would fold the
    very cells that were set aside as evidence back into the reported numbers.

    A pass under a second key therefore has to name it in the config, which is
    the intent: an extra key changes what is reported and should not be picked
    up by whatever happens to be on disk.
    """
    return list((cfg.sweep_dir / cfg.key).glob("*/labels.npy"))


def _layer_name(features_cfg: dict[str, Any]) -> str | None:
    """The name this study calls a job's representation.

    Mirrors the names in a dataset's `representations`, because everything
    downstream -- the tables, `eta_squared`, the per-option argmax -- groups on
    this string. A concat is named for the whole recipe rather than for the
    embedding it contains, so `state_dinov3` does not silently pool with the
    `dinov3` arm it shares an npz with.
    """
    target = features_cfg.get("_target_") or ""
    if "StateFeatures" in target:
        return "state"
    if "ConcatFeatures" in target:
        nested = [(e.get("_target_") or "") for e in (features_cfg.get("extractors") or [])]
        has_state = any("StateFeatures" in t for t in nested)
        layers = [
            e.get("layer")
            for e in (features_cfg.get("extractors") or [])
            if "PolicyEmbeddings" in (e.get("_target_") or "")
        ]
        parts = (["state"] if has_state else []) + [str(x) for x in layers if x]
        return "_".join(parts) or None
    return features_cfg.get("layer")


def _dials(cfg_yaml: dict[str, Any]) -> dict[str, Any]:
    """The nine swept axes, read back out of a job's own resolved config."""
    fe = cfg_yaml.get("features") or {}
    win = cfg_yaml.get("windowing") or {}
    red = cfg_yaml.get("reduction") or {}
    clu = cfg_yaml.get("clustering") or {}
    gr = cfg_yaml.get("graph") or {}
    return {
        # Two extractors have no `layer` field: `state` reads the raw joint
        # vector, and `state_dinov3` is a ConcatFeatures whose layer lives on a
        # nested extractor. Both were previously read as `None`, which reached
        # the scorer as a layer name and raised
        # `FileNotFoundError: layer 'None' not found` -- one task's rescore
        # died and the chain dropped it from the tables.
        "layer": _layer_name(fe),
        "width": int(win["width"]),
        "stride": int(win["stride"]),
        "reduction": red.get("_target_", "?").split(".")[-2],
        "n_components": red.get("n_components"),
        "clusterer": clu.get("_target_", "?").split(".")[-2],
        # One column for three differently-named knobs: k-means' cluster count,
        # spectral's component count, HDBSCAN's minimum cluster size. They are
        # not the same quantity, which is why `Cell.k_label` tags the last one.
        "k": clu.get("n_clusters") or clu.get("n_components") or clu.get("min_cluster_size"),
        "smoothing": gr.get("smoothing"),
        "sticky_lambda": gr.get("sticky_lambda"),
        # The core *pins* these three rather than sweeping them, so they were
        # not recorded -- and a cell varying one of them therefore had exactly
        # the dials of a core cell and passed `is_core_dials`. That is how 204
        # `n_neighbors` cells entered `rescored_core.json` and won
        # `core.select`'s newest-wins dedup, being the newest directories.
        # Recorded now so `core.PINNED_DIALS` can reject them. `None` is the
        # honest value where a dial does not apply: PCA has no `n_neighbors`,
        # k-means no `min_duration`.
        "n_neighbors": red.get("n_neighbors"),
        "min_duration": clu.get("min_duration"),
        "episode_norm": cfg_yaml.get("episode_norm") or "none",
    }


def run(
    cfg: DatasetConfig,
    out: pathlib.Path | None = None,
    only: Callable[[dict[str, Any]], bool] | None = None,
) -> pathlib.Path:
    """Write ``rescored.json``: one record per completed job, dials plus metrics.

    ``only`` filters by a job's dials, which is what makes a subset affordable:
    the full bottles grid is 5,374 jobs at ~0.9s each, while the 180-job core
    design takes under three minutes. Pass a separate ``out`` when filtering, or
    the partial result replaces the whole artefact.
    """
    out = out or cfg.rescored
    rollouts = RolloutDirectory(cfg.root)
    n_episodes = len(rollouts.episodes())
    ground_truth = {gs: build_ground_truth(cfg.root, gt_set=gs) for gs in GT_SETS}

    # One Scorer, used only for its memoised windows: the two ground-truth sets
    # share one set of features, and instantiating a Scorer per set would
    # extract everything twice.
    scorer = Scorer(cfg, gt_set=GT_SETS[0])

    jobs = sorted(job_dirs(cfg))

    # One tolerance for the whole artefact, pinned by the Scorer to the train
    # split. Not to the episodes each job happens to score: that would make
    # every fold a different metric.
    tolerances = (scorer.tolerance, round(scorer.tolerance * LOOSE_MULTIPLE, 4))
    print(
        f"{cfg.name}: {len(jobs)} jobs; tolerance {tolerances[0]} / {tolerances[1]}; "
        f"density band {cfg.evaluation.density_band}",
        flush=True,
    )

    records: list[dict[str, Any]] = []
    skipped = 0
    t0 = time.time()
    for n, lp in enumerate(jobs, 1):
        d = lp.parent
        job = yaml.safe_load((d / "config.yaml").read_text()) or {}
        dials = _dials(job)
        if only is not None and not only(dials):
            continue
        labels = np.load(lp)
        windows = scorer.windows(dials["layer"], dials["width"], dials["stride"])
        if labels.shape[0] != windows.num_windows:
            # A stale job whose feature file has since changed length. Counted
            # and reported rather than silently dropped, because a nonzero count
            # means some arm of the grid needs re-running, not re-scoring.
            print(f"  SKIP {d.name}: {labels.shape[0]} labels vs {windows.num_windows} windows")
            skipped += 1
            continue

        split = resolve_split(job.get("split"), n_episodes)
        scored = list(split.score) if split else sorted(ground_truth["task"])

        rec: dict[str, Any] = {
            "dir": str(d),
            **dials,
            "split": split.label if split else "none",
            "metrics": {},
        }
        for gs, gt_all in ground_truth.items():
            gt = {i: gt_all[i] for i in scored if i in gt_all}
            eps = sorted(gt)
            preds = predicted_boundaries(labels, windows, {i: g["fps"] for i, g in gt.items()})
            block: dict[str, Any] = {}
            for ti, t in enumerate(tolerances):
                for ms in cfg.evaluation.min_segments:
                    sc = score_boundaries(
                        preds,
                        gt,
                        eps,
                        t,
                        ms,
                        cfg.evaluation.density_band,
                        with_segmental=(ti == 0),
                    )
                    tag = ("" if ti == 0 else "loose") + ("" if ms == 0.0 else f"ms{ms:g}")
                    block[tag or "primary"] = {
                        k: v for k, v in sc.items() if k not in ("tolerance", "min_segment")
                    }
            block["label_agreement"] = label_agreement(labels, windows, gt, eps)
            rec["metrics"][gs] = block
        records.append(rec)
        if n % 250 == 0:
            print(f"  {n}/{len(jobs)}  {time.time() - t0:.0f}s elapsed", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, default=float))
    print(
        f"wrote {out}  ({len(records)} records"
        f"{f', {skipped} skipped' if skipped else ''}, {time.time() - t0:.0f}s)"
    )
    return out
