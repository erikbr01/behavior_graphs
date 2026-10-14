#!/usr/bin/env python
"""Embed episode video with a frozen vision backbone, into the policy-layer slot.

Writes ``<root>/policy_embeddings/<name>.npz`` with a single ``embeddings``
array, the same format ``extract_rgrl_embeddings.py`` produces. That is
deliberate: the sweep runs under a RAPIDS venv that has cuML and no torch,
while the backbone needs torch and has no cuML, so the two cannot run in one
process. Landing the features on disk in the format the cheap reader already
understands is what lets a frozen backbone be a drop-in extra representation
rather than a separate pipeline.

Run under the vision venv, with HF_HOME pointing at the cache holding the
weights:

    HF_HOME=${HF_HOME} \
    PYTHONPATH=src ${DATA_ROOT}/behavior_graphs/venvs/dino-patch-features/bin/python \
      scripts/ablation_study/extract_vision_features.py --dataset bottles

Needs `study media --dataset <name>` first, which links the raw mp4s into the
canonical layout.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from behavior_graphs.ablation_study import config, media  # noqa: E402
from behavior_graphs.envs.trajectory_dir import RolloutDirectory  # noqa: E402
from behavior_graphs.stages.features.vision_backbone import (  # noqa: E402
    VisionBackboneFeatures,
)

DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--name", default=None, help="output layer name, default from --model")
    ap.add_argument(
        "--cameras",
        default="top,left,right",
        help="views to embed and concatenate. Three by default, to match what "
        "the policy sees: a one-view baseline against three-view policy taps "
        "measures available information rather than representation quality. "
        "Empty string means every linked view.",
    )
    ap.add_argument("--pooling", default="grid", help="grid keeps spatial structure")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--decode-workers", type=int, default=8)
    ap.add_argument("--l2-normalize", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = config.load(args.dataset)
    cameras = media.resolve_cameras(cfg, args.cameras)
    name = args.name or ("dinov3" if "dinov3" in args.model else "dinov2")
    dest = cfg.root / "policy_embeddings" / f"{name}.npz"
    if dest.exists() and not args.overwrite:
        raise SystemExit(f"{dest} exists; pass --overwrite to replace it")

    print(
        f"{cfg.name}: {args.model} pooling={args.pooling} grid={args.grid} "
        f"cameras={','.join(cameras)} image_size={args.image_size}",
        flush=True,
    )
    rollouts = RolloutDirectory(cfg.root)
    n_steps = sum(e.num_samples for e in rollouts.episodes())
    print(f"  {len(rollouts.episodes())} episodes, {n_steps} timesteps", flush=True)

    extractor = VisionBackboneFeatures(
        model_name=args.model,
        pooling=args.pooling,
        grid=args.grid,
        cameras=cameras,
        image_size=args.image_size,
        batch_size=args.batch_size,
        decode_workers=args.decode_workers,
        l2_normalize=args.l2_normalize,
    )
    t0 = time.time()
    fs = extractor.extract(rollouts)
    took = time.time() - t0

    feats = np.asarray(fs.features, dtype=np.float32)
    if feats.shape[0] != n_steps:
        raise SystemExit(
            f"got {feats.shape[0]} rows for {n_steps} timesteps; a short array here "
            "means some episode had no decodable video, which would silently "
            "misalign every window downstream"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, embeddings=feats)
    print(
        f"  wrote {dest}  {feats.shape} {feats.dtype}  "
        f"({dest.stat().st_size / 2**30:.2f} GB, {took / 60:.1f} min)"
    )
    print(
        f"  finite: {bool(np.all(np.isfinite(feats)))}  "
        f"row norm mean {float(np.linalg.norm(feats, axis=1).mean()):.3f}"
    )


if __name__ == "__main__":
    main()
