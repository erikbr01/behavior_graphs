"""Phase A of the progress-estimation benchmark: run OUR graph-derived progress
estimator across ALL feasible behavior_graphs rollout datasets, with an adaptive
K-sweep, and emit a combined cross-dataset table.

Feasible = has episodes/metadata.yaml (with per-episode success flags) and at
least one policy_embeddings/*.npz. We exclude datasets with <2 successful
episodes (no progress GT to correlate against). AUROC is NaN where a dataset has
0 failures (nothing to detect), but VOC/Kendall/Pearson/MAE remain valid.

K is NOT universal (LIBERO wants high K, mimicgen wants low K), so we sweep K per
dataset and report the best-by-stage-VOC as well as the full sweep. K candidates
are capped by the number of TRAIN windows so we never ask KMeans for more
clusters than the data can support.

CPU only, no GPU. Uses the native library
(behavior_graphs.progress_estimation: EpisodeSet, GraphBackbone, BenchmarkRunner).

    PYTHONPATH=src ${DATA_ROOT}/behavior_graphs/venvs/diffusion_policy_venv/bin/python \
        scripts/sweep_progress_allds.py --out analysis/out/progress_allds
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import _paths  # noqa: E402

# Thread cap first: it only takes effect before numpy/sklearn initialise.
# See scripts/_threads.py for the measurement that made this necessary.
import _threads  # noqa: F401,E402  (import for its side effect)
import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from behavior_graphs.progress_estimation.io import (  # noqa: E402
    EpisodeSet,
    policy_embedding_layer,
)
from behavior_graphs.progress_estimation.methods import (  # noqa: E402
    FrameIndex,
    GraphAbsorb,
    GraphStage,
    TunedGraphBackbone,
)
from behavior_graphs.progress_estimation.metrics import (  # noqa: E402
    MAE,
    VOC,
    KendallTau,
    Pearson,
    SuccessAUROC,
)
from behavior_graphs.progress_estimation.report import ABSORB, STAGE, sweep_table_md  # noqa: E402
from behavior_graphs.progress_estimation.runner import BenchmarkRunner, KFoldSplit  # noqa: E402
from behavior_graphs.stages import IdentityReduction  # noqa: E402

# The dataset zoo (feasible datasets grouped by family) lives in configs so it is
# the single source of truth; datasets with <2 successful episodes are skipped at
# runtime.
ZOO_CONFIG = Path(__file__).resolve().parents[1] / "configs/progress_estimation/data/zoo.yaml"

K_CANDIDATES = [8, 16, 32, 64, 128]
METRICS = ["voc", "kendall", "pearson", "mae", "auroc"]
# method name -> the sweep_results.json scheme key (kept stable for consumers)
SCHEME = {"graph_stage": STAGE, "graph_absorb": ABSORB, "frame_index": "frame_index"}


def _approx_windows(episodes, ww, ws):
    return int(sum(max(1, (len(e) - ww) // ws + 1) for e in episodes))


def _metrics():
    return [VOC(), KendallTau(), Pearson(), MAE(), SuccessAUROC()]


def sweep_dataset(name, rollout_dir, args):
    layer = args.layer if args.layer != "auto" else policy_embedding_layer(rollout_dir)
    ds = EpisodeSet.from_dir(name, rollout_dir, layer=layer)
    episodes = ds.embeddings()
    flags = ds.flags
    ns, nf = int(np.sum(flags)), len(flags) - int(np.sum(flags))
    if ns < 2:
        return {"skipped": f"only {ns} successful episodes"}

    # cap K so train windows can support it (>= ~10 windows/cluster)
    n_win = _approx_windows(episodes, args.window_width, args.window_stride)
    n_train_win = int(n_win * (1 - 1.0 / args.folds))
    ks = [k for k in K_CANDIDATES if k <= max(8, n_train_win // 10)]
    if 8 not in ks:
        ks = [8]

    split = KFoldSplit(n_folds=args.folds, test_frac=args.test_frac, seed=args.seed)

    # K is chosen inside each outer fold, on a validation split carved out of
    # that fold's TRAIN half, and the winner is refit on the whole of it, so the
    # reported metrics are scored on a test fold that had no say in the choice.
    # There is no other mode: this script used to offer `--select test`, which
    # scored every K on the test folds and reported the maximum -- selection on
    # the number being published. It is gone rather than defaulted off, because
    # the only thing it could be used for is producing a number we would not
    # report.
    bb = TunedGraphBackbone(
        [{"n_clusters": k} for k in ks],
        val_frac=args.val_frac,
        val_seed=args.seed,
        seed=args.seed,
        window_width=args.window_width,
        window_stride=args.window_stride,
        reduction=IdentityReduction(normalize="standard"),
    )
    methods = [GraphStage(backbone=bb), GraphAbsorb(backbone=bb), FrameIndex(backbone=bb)]
    summary = BenchmarkRunner(_metrics(), split).run(ds, methods)
    chosen = [h["chosen"].get("n_clusters") for h in bb.history]
    print(f"    nested: K per fold = {chosen}, stage VOC={summary['graph_stage']['voc'][0]:.3f}")
    return {
        "n_eps": len(episodes),
        "n_succ": ns,
        "n_fail": nf,
        "mean_len": float(np.mean([len(e) for e in episodes])),
        "layer": layer,
        "ks_swept": ks,
        "selection": "nested",
        # No single best_k: K is re-chosen per fold, so what the run has is a
        # choice per fold. Collapsing that to one number would hide the
        # variance the nesting exists to expose.
        "k_per_fold": chosen,
        "val_frac": args.val_frac,
        "per_k": {"selected": {SCHEME[m]: {k: summary[m][k][0] for k in METRICS} for m in summary}},
        "best_k": "selected",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rollouts-root",
        type=Path,
        default=_paths.env_path("${DATA_PATH}/rollouts"),
    )
    ap.add_argument("--out", type=Path, default=Path("analysis/out/progress_allds"))
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.25,
        help="Fraction of each fold's train half held back to choose K on.",
    )
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--window-width", type=int, default=2)
    ap.add_argument("--window-stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated dataset names to restrict to")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    families = OmegaConf.to_container(OmegaConf.load(ZOO_CONFIG), resolve=True)
    only = {x for x in args.only.split(",") if x}
    results = {}
    for family, names in families.items():
        for name in names:
            if only and name not in only:
                continue
            rollout_dir = args.rollouts_root / name
            if not rollout_dir.exists():
                print(f"[{name}] MISSING dir, skip")
                continue
            print(f"[{name}] ({family})")
            try:
                res = sweep_dataset(name, rollout_dir, args)
                res["family"] = family
                results[name] = res
                if "skipped" in res:
                    print(f"    SKIPPED: {res['skipped']}")
            except Exception:
                print(f"    ERROR:\n{traceback.format_exc()}")
                results[name] = {"family": family, "error": traceback.format_exc()}
            (args.out / "sweep_results.json").write_text(json.dumps(results, indent=2))

    table = sweep_table_md(results)
    (args.out / "sweep_table.md").write_text(table)
    print("\n" + table)
    print(f"wrote {args.out}/sweep_results.json and sweep_table.md")


if __name__ == "__main__":
    main()
