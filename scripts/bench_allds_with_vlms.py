#!/usr/bin/env python
"""Score graph + VLM progress estimators on every dataset, saving curves.

Every method is scored on the SAME folds, so its curves come out keyed by the
same episode ids. That matters beyond tidiness: the previous artifacts came from
separate passes that left each method a different `idx` order, which made the
"paired" episode bootstrap in consolidate_progress_results.py compare different
episodes. `KFoldSplit` is seeded and depends only on the success flags, so a run
of `--methods robometer_4b` and a run of `--methods topreward_ll` over the same
dataset get identical folds -- the pairing survives across processes, which is
what makes the method sharding below safe.

Sharded BY METHOD, not by dataset: the three checkpoints (two 4B plus an 8B) do
not fit on one 24GB card together. One model per process, several processes.

Every VLM sees `--vlm-max-frames` frames per episode (64). Not a tuning choice:
ROBOMETER pushes the whole trajectory through the vision tower in one forward,
which OOMs 24GB somewhere past a few hundred frames, and these datasets run to
708 steps. It also deviates from the published LIBERO rows, which used the whole
episode -- those were short enough to fit. Same budget for both VLMs, so they at
least see the same thing as each other.

The graph's K is chosen by `TunedGraphBackbone` on a validation split of each
fold's TRAIN half, never on the fold being scored.

VLM methods need video; 21 of the zoo's datasets have none (the safe_franka /
safe_widowx real-robot sets), so they run graph-only there rather than being
skipped -- a graph curve is still worth having.

    HF_HOME=${HF_HOME} CUDA_VISIBLE_DEVICES=0 \\
      .../robometer/.venv/bin/python scripts/bench_allds_with_vlms.py \\
      --out analysis/out/progress_allmethods --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

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
    MLPRegressorProgress,
    Robometer,
    TopRewardLL,
    TunedGraphBackbone,
)
from behavior_graphs.progress_estimation.metrics import (  # noqa: E402
    MAE,
    VOC,
    KendallTau,
    Pearson,
    SuccessAUROC,
)
from behavior_graphs.progress_estimation.runner import BenchmarkRunner, KFoldSplit  # noqa: E402
from behavior_graphs.stages import IdentityReduction  # noqa: E402

ZOO = Path(__file__).resolve().parents[1] / "configs/progress_estimation/data/zoo.yaml"
K_CANDIDATES = [8, 16, 32, 64, 128]
REPO = str(Path(__file__).resolve().parents[1] / "third_party/robometer")


def _metrics():
    return [VOC(), KendallTau(), Pearson(), MAE(), SuccessAUROC()]


def _has_video(rollout_dir: Path) -> bool:
    media = rollout_dir / "media"
    if not media.is_dir():
        return False
    return bool(next(media.glob("*.mp4"), None) or next(media.glob("*/*.mp4"), None))


def _approx_windows(episodes, width, stride) -> int:
    return sum(max(0, (len(e) - width) // stride + 1) for e in episodes)


def run_dataset(name: str, rollout_dir: Path, args, vlms: dict) -> dict:
    layer = policy_embedding_layer(rollout_dir) if args.layer == "auto" else args.layer
    ds = EpisodeSet.from_dir(name, rollout_dir, layer=layer, suite=name)
    episodes, flags = ds.embeddings(), ds.flags
    n_succ = int(sum(flags))
    if n_succ < 2:
        return {"skipped": f"only {n_succ} successful episodes"}
    if len(flags) - n_succ < 1:
        # AUROC needs both classes; the ranking metrics still work, so run on.
        print("    note: no failures, success-AUROC will be undefined")

    # Cap K by the train-window budget, as the graph-only sweep does.
    n_win = _approx_windows(episodes, args.window_width, args.window_stride)
    n_train = int(n_win * (1 - 1.0 / args.folds))
    ks = [k for k in K_CANDIDATES if k <= max(8, n_train // 10)] or [8]

    backbone = TunedGraphBackbone(
        [{"n_clusters": k} for k in ks],
        val_frac=args.val_frac,
        val_seed=args.seed,
        seed=args.seed,
        window_width=args.window_width,
        window_stride=args.window_stride,
        reduction=IdentityReduction(normalize="standard"),
    )
    methods = []
    if "mlp" in args.wanted:
        # The supervised counterpart to the graph: regresses progress straight
        # from the same policy embedding, so a gap between the two is about the
        # graph structure rather than about the representation. Same folds, so
        # it is fit on the train half and scored on the held-out one.
        methods.append(
            MLPRegressorProgress(
                window_width=args.mlp_window,
                max_iter=args.mlp_max_iter,
                seed=args.seed,
                backend=args.mlp_backend,
            )
        )
    if "graph" in args.wanted:
        methods += [
            GraphStage(backbone=backbone),
            GraphAbsorb(backbone=backbone),
            FrameIndex(backbone=backbone),
        ]
    video = _has_video(rollout_dir)
    if video:
        for key in ("robometer_4b", "topreward_ll"):
            if key in vlms:
                methods.append(vlms[key])
        # The LIBERO-finetuned checkpoint is an in-domain reference and means
        # nothing off it, so it is not run elsewhere.
        if "robometer_fft" in vlms and name.startswith("libero"):
            methods.append(vlms["robometer_fft"])
    if not methods:
        return {"skipped": "no requested method can run here (the VLMs need video)"}

    split = KFoldSplit(n_folds=args.folds, test_frac=args.test_frac, seed=args.seed)
    out_dir = args.out
    summary = BenchmarkRunner(_metrics(), split, out_dir=out_dir).run(ds, methods)
    return {
        "n_eps": len(episodes),
        "n_succ": n_succ,
        "n_fail": len(flags) - n_succ,
        "layer": layer,
        "video": video,
        "ks_swept": ks,
        "k_per_fold": [h["chosen"].get("n_clusters") for h in backbone.history],
        "selection": "nested",
        "val_frac": args.val_frac,
        "folds": args.folds,
        "methods": [m.name for m in methods],
        "metrics": {m: {k: v[0] for k, v in summary[m].items()} for m in summary},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts-root", type=Path, default=Path("${DATA_PATH}/rollouts"))
    ap.add_argument("--out", type=Path, default=Path("analysis/out/progress_allmethods"))
    ap.add_argument("--layer", default="auto")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--window-width", type=int, default=2)
    ap.add_argument("--window-stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-window", type=int, default=1)
    ap.add_argument("--mlp-max-iter", type=int, default=400)
    ap.add_argument(
        "--mlp-backend",
        default="sklearn",
        choices=["sklearn", "torch"],
        help="sklearn is the published setting; torch scales to the big stores.",
    )
    ap.add_argument(
        "--vlm-max-frames",
        type=int,
        default=64,
        help="Frames each VLM sees per episode (linspace-subsampled). Episodes here "
        "run 100-708 steps and ROBOMETER feeds the whole trajectory through the "
        "vision tower in one forward, which OOMs a 24GB card past a few hundred "
        "frames -- and the model was trained on short budgets anyway. 64 is what "
        "topreward_ll already uses, so both VLMs see the same number of frames.",
    )
    ap.add_argument(
        "--vlm-long-side",
        type=int,
        default=448,
        help="Cap on a frame's longer edge for the ROBOMETER methods. Clips here run "
        "from 224x224 (LIBERO) to 1920x960 (vla_foundry), a 37x spread in vision "
        "tokens per frame -- the large ones OOM at any useful frame count, and "
        "comparing methods that saw 37x different amounts of image would be "
        "meaningless anyway. topreward_ll resizes internally to its own image_size.",
    )
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--methods",
        default="graph",
        help="Comma-separated: graph (stage+absorb+frame_index), mlp, robometer_4b, "
        "topreward_ll, robometer_fft. graph and mlp read policy embeddings so they "
        "run on every dataset; the VLMs need video. One VLM per process -- see the "
        "module docstring.",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    unknown = args.wanted - {"graph", "mlp", "robometer_4b", "topreward_ll", "robometer_fft"}
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}")

    families = OmegaConf.to_container(OmegaConf.load(ZOO), resolve=True)
    only = {x for x in args.only.split(",") if x}
    todo = [
        (family, name)
        for family, names in families.items()
        for name in names
        if not only or name in only
    ]
    # Interleaved rather than blocked, so a shard does not get all the big
    # datasets of one family.
    todo = todo[args.shard :: args.num_shards]

    # Only the requested model is constructed, and `fit` is idempotent, so it
    # loads once on the first dataset that needs it and stays resident for the
    # rest of the shard.
    builders = {
        "robometer_4b": lambda: Robometer(
            ckpt="robometer/Robometer-4B",
            name="robometer_4b",
            repo=REPO,
            max_frames=args.vlm_max_frames,
            image_long_side=args.vlm_long_side,
        ),
        "robometer_fft": lambda: Robometer(
            ckpt="jesbu1/robometer-4b-fft-libero",
            name="robometer_fft",
            repo=REPO,
            max_frames=args.vlm_max_frames,
            image_long_side=args.vlm_long_side,
        ),
        "topreward_ll": lambda: TopRewardLL(
            model_id="Qwen/Qwen3-VL-8B-Instruct", repo=REPO, max_frames=args.vlm_max_frames
        ),
    }
    vlms = {name: build() for name, build in builders.items() if name in args.wanted}

    index_path = args.out / f"index_{'-'.join(sorted(args.wanted))}_shard{args.shard}.json"
    results: dict = json.loads(index_path.read_text()) if index_path.is_file() else {}
    for family, name in todo:
        if name in results and "error" not in results[name]:
            print(f"[{name}] done already, skip")
            continue
        rollout_dir = args.rollouts_root / name
        if not rollout_dir.exists():
            print(f"[{name}] MISSING dir, skip")
            continue
        print(f"[{name}] ({family})", flush=True)
        try:
            res = run_dataset(name, rollout_dir, args, vlms)
            res["family"] = family
            results[name] = res
            if "skipped" in res:
                print(f"    SKIPPED: {res['skipped']}")
        except Exception:
            print(f"    ERROR:\n{traceback.format_exc()}", flush=True)
            results[name] = {"family": family, "error": traceback.format_exc()}
        index_path.write_text(json.dumps(results, indent=2))
    print(f"shard {args.shard}: {len(results)} datasets -> {index_path}")


if __name__ == "__main__":
    main()
