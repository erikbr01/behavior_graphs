#!/usr/bin/env python
"""Fit the behavior graph on TRAINING DEMOS, score progress on the rollouts.

The benchmark fits the graph on a train split of rollouts, which quietly needs
something deployment does not have: the success label of the rollouts it fits
on (`stage_schedule` derives stage order from successful episodes only). The
demonstrations a policy was trained on are available by definition and are
successful by construction, so fitting there removes the dependency -- and
frees the WHOLE rollout set for evaluation, failures included, instead of a
held-out third of it.

Both sides must be the same representation. Rollout dirs carry `cond_proj`,
captured by a forward hook during collection at the sampler's final step
(t = 1/inference_steps); `scripts/rabc/extract_policy_embeddings.py --layer
cond_proj` reproduces exactly that over the demo store. Fitting on `obs_proj`
instead would not fail -- both are hidden_size-wide -- it would just be
meaningless.

Two questions, reported separately because they need different denominators:

  ordering   VOC / Kendall / Pearson / MAE against a linear ramp, over
             SUCCESSFUL rollouts. A failed rollout has no ground-truth progress
             curve to be right about, so scoring it here would be inventing one.
  detection  AUROC of terminal progress against the success flag, over ALL
             rollouts, plus the two terminal distributions -- which is the real
             question of whether progress stalls when the policy fails.

    python scripts/demo_fit_progress.py \\
        --demos  .../demo_embeddings/set_table_baseline_pretrained_dit \\
        --rollouts .../rollouts/set_table_baseline_pretrained_dit \\
        --out .../analysis/demo_fit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_graphs.progress_estimation.estimation import (  # noqa: E402
    progress_regression_metrics,
    success_auroc,
)
from behavior_graphs.progress_estimation.io import (  # noqa: E402
    EpisodeSet,
    EpisodeView,
    save_curves,
)
from behavior_graphs.progress_estimation.methods import (  # noqa: E402
    GraphStage,
    TunedGraphBackbone,
)
from behavior_graphs.stages import IdentityReduction  # noqa: E402

K_CANDIDATES = [8, 16, 32, 64, 128]


def _whole(ds: EpisodeSet) -> EpisodeView:
    return EpisodeView(ds, list(range(len(ds.flags))))


def run(demos: Path, rollouts: Path, args) -> dict:
    demo_set = EpisodeSet.from_dir(demos.name, str(demos), layer=args.layer)
    roll_set = EpisodeSet.from_dir(rollouts.name, str(rollouts), layer=args.layer)
    demo_view, roll_view = _whole(demo_set), _whole(roll_set)
    flags = np.asarray(roll_set.flags, dtype=bool)

    # K is chosen on a validation split of the DEMOS -- the rollouts are the
    # evaluation set here and never inform the fit, not even through K.
    backbone = TunedGraphBackbone(
        [{"n_clusters": k} for k in K_CANDIDATES],
        val_frac=args.val_frac,
        val_seed=args.seed,
        seed=args.seed,
        window_width=args.window_width,
        window_stride=args.window_stride,
        reduction=IdentityReduction(normalize="standard"),
    )
    method = GraphStage(backbone=backbone)
    method.fit(demo_view)
    curves = method.predict(roll_view)

    ordering = progress_regression_metrics(curves, list(flags))
    tail = args.tail
    terminal = np.array(
        [
            float(np.mean(np.clip(np.asarray(c, float), 0, 1)[-tail:])) if len(c) else np.nan
            for c in curves
        ]
    )
    auroc = success_auroc(terminal, flags) if flags.any() and (~flags).any() else float("nan")
    ts, tf = terminal[flags], terminal[~flags]

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        save_curves(
            args.out,
            rollouts.name,
            "graph_direct_stage_demofit",
            curves,
            list(flags),
            list(range(len(curves))),
        )
    return {
        "n_demos": len(demo_view),
        "n_rollouts": len(roll_view),
        "n_succ": int(flags.sum()),
        "n_fail": int((~flags).sum()),
        "k_chosen": backbone.chosen.get("n_clusters") if backbone.chosen else None,
        "ordering_on_successes": {k: float(v) for k, v in ordering.items()},
        "detection_auroc": float(auroc),
        "terminal_success_mean": float(np.nanmean(ts)) if len(ts) else None,
        "terminal_failure_mean": float(np.nanmean(tf)) if len(tf) else None,
        "terminal_gap": (float(np.nanmean(ts) - np.nanmean(tf)) if len(ts) and len(tf) else None),
    }


def _fmt(x: float | None) -> str:
    """Three decimals, or an em dash for a value that genuinely does not exist."""
    if x is None or (isinstance(x, float) and x != x):
        return "--"
    return f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", type=Path, action="append", required=True)
    ap.add_argument("--rollouts-root", type=Path, default=Path("${DATA_PATH}/rollouts"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--layer", default="cond_proj")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--window-width", type=int, default=2)
    ap.add_argument("--window-stride", type=int, default=3)
    ap.add_argument("--tail", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results = {}
    for demo_dir in args.demos:
        name = demo_dir.name
        roll = args.rollouts_root / name
        if not roll.exists():
            print(f"[{name}] no rollout dir at {roll}, skip")
            continue
        print(f"[{name}]", flush=True)
        res = run(demo_dir, roll, args)
        results[name] = res
        o = res["ordering_on_successes"]
        print(
            f"    K={res['k_chosen']}  demos={res['n_demos']}  rollouts={res['n_rollouts']}"
            f" ({res['n_succ']}s/{res['n_fail']}f)"
        )
        print(f"    ordering (successes): VOC {_fmt(o.get('voc'))}  MAE {_fmt(o.get('mae'))}")
        # A one-sided rollout set leaves one of these None -- `nut_assembly_d0`
        # is 0 successes / 50 failures, and formatting None crashed the whole
        # sweep four datasets in. There is nothing to detect when one class is
        # missing, so say so rather than printing a number that cannot exist.
        print(
            f"    detection (all):      AUROC {_fmt(res['detection_auroc'])}   "
            f"terminal succ {_fmt(res['terminal_success_mean'])} vs "
            f"fail {_fmt(res['terminal_failure_mean'])}"
            + ("   [one-sided: detection undefined]" if res["terminal_gap"] is None else ""),
            flush=True,
        )
        # Written every dataset, not once at the end: the previous version lost
        # every completed result when a later dataset raised.
        if args.out:
            (args.out / "demo_fit_results.json").write_text(json.dumps(results, indent=2))
    if args.out:
        print(f"\nwrote {args.out / 'demo_fit_results.json'}")


if __name__ == "__main__":
    main()
