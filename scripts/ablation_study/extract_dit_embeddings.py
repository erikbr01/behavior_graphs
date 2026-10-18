#!/usr/bin/env python
"""Per-frame ABC-DiT policy embeddings for annotated ABC-130k episodes.

Run in the abc_rabc venv (torch + CUDA) with PYTHONPATH pointing at the abc clone.

Decodes each episode's three h264 camera streams and runs the released ABC-DiT
(``bottles_release_prep_75k.pt``, 2.02B params, hidden 1536, depth 32) forward in
the same pass, so only the embeddings are written -- storing 224x224x3 frames for
three cameras costs ~500 MB per episode and buys nothing.

Three layers, chosen to mirror what was extracted from pi05 on WGO-Bench so the
two studies are comparable:

``abc_vision``
    ``build_vision_tokens`` mean-pooled: DINOv3 ViT-B/16 tokens per camera,
    attention-pooled to 12 queries each, projected to hidden and given a camera
    embedding, then averaged over the 36 tokens. **State-free and
    timestep-free** -- the policy's visual reading of the observation.

``abc_cond``
    ``compute_cond(state, task, t=0)``: the conditioning vector that mixes
    proprioception, the CLIP task embedding and the flow timestep. Deterministic.

``abc_dit``
    Mean-pooled hidden state of the **last DiT block** at the final flow step,
    over the 30 action tokens -- the diffusion transformer's own representation,
    the analogue of PR #46's ``flow_emb``. Depends on the sampled noise, so the
    noise is seeded per episode and the seed recorded.

Caveat worth carrying into any result: this checkpoint was trained on a bottles
mixture (82% real ``throw_the_plastic_bottles_in_the_bin``, 18% sim) with
``vision_lr_scale=1.0``, i.e. the DINOv3 backbone was **not** frozen. So its
representations are specialized to bottle-throwing, and the annotated task here
(``put_the_plastic_bottles_in_the_bin``) is a near-twin delivery, not the
training task itself.

    PYTHONPATH=${DATA_ROOT}/abc_rabc python scripts/ablation_study/extract_dit_embeddings.py \\
        --dataset $DATA_PATH/ablation_bottles --checkpoint .../cache_bottles_75k.pt
"""

from __future__ import annotations

import argparse
import sys as _sys

_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import pathlib
import sys

import _paths  # noqa: E402
import numpy as np
import torch
import yaml

# ruff: noqa: E402
# The h264 decode helpers live beside this script, and abc_minimal comes from the
# abc clone on PYTHONPATH, so both imports have to follow the sys.path edit.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from abc_minimal.config import ClipConfig, DiTConfig
from abc_minimal.dit import CLIPTextEmbedder, DiTPolicy, load_pretrained
from abc_minimal.preprocess import imagenet_normalize, normalize, parse_norm_stats
from extract_frames import (
    CAMERA_TOPICS,
    decode_stream,
    probe_size,
    read_packets,
    resize_pad_uint8,
)


@torch.no_grad()
def episode_embeddings(
    model: DiTPolicy,
    frames: dict[str, np.ndarray],
    state: np.ndarray,
    task_vec: torch.Tensor,
    device: str,
    batch: int,
    flow_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """(T, 1536) arrays for each of the three tap points."""
    T = state.shape[0]
    out = {k: [] for k in ("abc_vision", "abc_cond", "abc_dit")}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    chunk, adim = model.pos_embed.shape[1], model.final_layer.linear.out_features

    for lo in range(0, T, batch):
        hi = min(lo + batch, T)
        n = hi - lo
        images = {}
        for cam in CAMERA_TOPICS:
            arr = torch.from_numpy(frames[cam][lo:hi]).to(device)
            arr = arr.permute(0, 3, 1, 2).float() / 255.0
            images[cam] = imagenet_normalize(arr)

        vision = model.build_vision_tokens(images)  # (n, 36, H)
        st = torch.from_numpy(state[lo:hi]).to(device).float()
        tv = task_vec.to(device).expand(n, -1)

        # abc_cond / abc_vision are timestep-independent reads of the observation.
        cond0 = model.compute_cond(st, tv, torch.zeros(n, device=device))
        out["abc_vision"].append(vision.mean(dim=1).float().cpu().numpy())
        out["abc_cond"].append(cond0.float().cpu().numpy())

        # abc_dit: run the flow-matching Euler loop, keep the last block's hidden.
        x = torch.randn(n, chunk, adim, generator=gen).to(device)
        dt = 1.0 / flow_steps
        hidden = None
        for k in range(flow_steps):
            t = torch.full((n,), 1.0 - k * dt, device=device)
            c = model.compute_cond(st, tv, t)
            z = model.y_embedder(x) + model.pos_embed.data[:, : x.shape[1], :]
            for block in model.blocks:
                z = block(z, c, vision)
            hidden = z
            x = x + dt * model.final_layer(z, c)
        out["abc_dit"].append(hidden.mean(dim=1).float().cpu().numpy())

    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=pathlib.Path, required=True, help="canonical layout root")
    ap.add_argument(
        "--mcap-root",
        type=pathlib.Path,
        default=_paths.env_path("${DATA_PATH}/abc_130k"),
    )
    ap.add_argument("--task", default="put_the_plastic_bottles_in_the_bin")
    ap.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=_paths.env_path("${DATA_ROOT}/abc_rabc/cache_bottles_75k.pt"),
    )
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--flow-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--prompt",
        default=None,
        help="CLIP prompt override. The dataset ships 'put the plastic bottles in "
        "the bin', but the checkpoint was conditioned on task_name_to_prompt() of "
        "its mixture: 'throw plastic bottles in bin' (81.7% of training) and 'sim "
        "put the plastic bottles in the bin' (18.3%). Neither equals the dataset "
        "string, so the in-distribution choice is the training one.",
    )
    ap.add_argument("--suffix", default="", help="appended to output layer names")
    args = ap.parse_args()

    meta = yaml.safe_load((args.dataset / "episodes" / "metadata.yaml").read_text())
    episodes = meta["episodes"][: args.limit] if args.limit else meta["episodes"]
    state_all = np.load(args.dataset / "observations" / "state.npz")["observations"]
    rate = float(meta["source"]["rate_hz"])

    print(f"loading ABC-DiT from {args.checkpoint} ...", flush=True)
    cfg = DiTConfig()
    model = DiTPolicy(cfg).to(args.device).eval()
    ck = load_pretrained(model, str(args.checkpoint))
    stats = parse_norm_stats(ck["norm_stats"])
    model.img_backbone.set_bfloat16(True)
    print(f"  loaded step={ck.get('step')} | state norm mean[:3]={stats['state']['mean'][:3]}")

    # The task vector is the CLIP text embedding of the instruction, exactly as
    # eval_policy.py builds it. ClipConfig is separate from DiTConfig.
    instruction = args.prompt or episodes[0]["instruction"]
    embedder = CLIPTextEmbedder(ClipConfig(), device=args.device)
    task_vec = embedder.encode([instruction]).to(args.device)
    print(f"  task vec for {instruction!r}: {tuple(task_vec.shape)}")

    src = args.mcap_root / "data" / "train" / args.task
    offset = 0
    starts = {}
    for e in meta["episodes"]:
        starts[e["index"]] = offset
        offset += e["length"]

    layers: dict[str, list[np.ndarray]] = {}
    for e in episodes:
        d = src / e["wgo_id"]
        packets, (t0, t1) = read_packets(d / "episode.mcap")
        grid = np.arange(t0, t1 + 1, int(round(1e9 / rate)), dtype=np.int64)
        frames = {}
        for cam, pkts in packets.items():
            w, h = probe_size(pkts)
            dec = decode_stream(pkts, w, h)
            times = np.asarray([t for t, _ in pkts], dtype=np.int64)[: len(dec)]
            idx = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
            take_left = (grid - times[idx - 1]) <= (times[idx] - grid)
            frames[cam] = resize_pad_uint8(dec[np.where(take_left, idx - 1, idx)], 224)

        n = min(e["length"], len(grid), *(v.shape[0] for v in frames.values()))
        sl = slice(starts[e["index"]], starts[e["index"]] + e["length"])
        st = normalize(state_all[sl][:n].astype(np.float32), stats["state"])
        frames = {k: v[:n] for k, v in frames.items()}

        emb = episode_embeddings(
            model,
            frames,
            st,
            task_vec,
            args.device,
            args.batch,
            args.flow_steps,
            seed=e["index"],
        )
        if n < e["length"]:  # pad by repeating the last frame's embedding
            for k, v in emb.items():
                emb[k] = np.concatenate([v, np.repeat(v[-1:], e["length"] - n, axis=0)])
        for k, v in emb.items():
            layers.setdefault(k, []).append(v)
        print(
            f"  ep{e['index']:04d} {e['wgo_id']}: {e['length']} frames "
            f"({n} decoded) -> {tuple(emb['abc_dit'].shape)}",
            flush=True,
        )

    out_dir = args.dataset / "policy_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, chunks in layers.items():
        arr = np.concatenate(chunks, axis=0)
        name = f"{k}{args.suffix}"
        np.savez(out_dir / f"{name}.npz", embeddings=arr)
        print(f"[extract] {name}: {arr.shape} -> {out_dir / (name + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
