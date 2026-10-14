"""Per-frame features from a trained rgrl policy, on the study's own frame axis.

Why this exists: the four-way comparison found that 14 raw joint angles beat all
three taps of the *fine-tuned ABC-DiT*, which is what you would expect from a
checkpoint trained on a different task. This extracts the same kind of features
from a policy trained in-domain on ABC bottles instead, so the representation
limit can be tested rather than assumed.

Two things are done carefully.

**The frame axis matches the existing dataset exactly.** `build_dataset.py` builds
a uniform 10 Hz grid per episode -- `arange(t0, t1 + 1, 1e8)` where t0/t1 are the
min/max timestamps of the *state* topics, not the cameras (which is why an
episode is 1983 frames and not the 1981 a stride-3 camera decimation would give).
That grid is reconstructed here from the same source, so the output drops straight
into `policy_embeddings/` beside `abc_dit.npz` and is scored by the same
annotations, splits and baselines. Only the representation changes.

**Preprocessing matches training.** Frames are letterboxed to the policy's input
size the same way `ResizeImages(letterbox=True)` did when building the training
cache, so the policy sees at inference what it saw at training.

Taps mirror the ABC-DiT ones: the observation encoder, the conditioning vector it
is projected into, and the last DiT block's hidden state.
"""

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

RGRL = pathlib.Path("$HOME/really_good_robot_learning")
sys.path.insert(0, str(RGRL))

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument(
    "--source",
    default=("${DATA_PATH}/abc_130k/data/train/put_the_plastic_bottles_in_the_bin"),
)
ap.add_argument("--out-root", default="${DATA_PATH}/abc_bottles_ds")
ap.add_argument("--rate", type=float, default=10.0, help="must match the dataset root")
ap.add_argument("--height", type=int, default=168)
ap.add_argument("--width", type=int, default=224)
ap.add_argument(
    "--batch",
    type=int,
    default=192,
    help="frames per forward pass; also the span decoded per "
    "camera, so larger amortises the seek further",
)
ap.add_argument("--device", default="cuda:0")
ap.add_argument(
    "--decode-workers",
    type=int,
    default=3,
    help="camera decodes run concurrently, one per camera by default. Measured "
    "on a cold chunk cache: serial 541ms a batch, 3 threads 255ms (2.12x), 6 "
    "threads 455ms -- more threads than cameras just contends with the "
    "concurrent training's dataloader. 0 disables threading entirely.",
)
ap.add_argument("--prefix", default="rgrl", help="output files <prefix>_<tap>.npz")
ap.add_argument("--limit-episodes", type=int, default=0)
ap.add_argument(
    "--cache",
    default=None,
    help=(
        "zarr training cache to read frames from instead of the MCAP sidecars. "
        "Much faster: the cache already holds every frame letterboxed at the "
        "policy's input size as an independently-decodable JPEG, whereas the h264 "
        "sidecars are inter-frame coded, so sampling every third frame still has to "
        "decode the two between (~40 fps measured, vs far more from the cache)."
    ),
)
args = ap.parse_args()

import cv2  # noqa: E402
from mcap.reader import make_reader  # noqa: E402
from mcap_protobuf.decoder import DecoderFactory  # noqa: E402
from rgrl.inference.activations import ActivationCapture  # noqa: E402
from rgrl.inference.checkpoint import InferenceConfig, load_policy  # noqa: E402
from rgrl.transforms import apply_pipeline  # noqa: E402
from robo_dataset.backends.abc_mcap import VECTOR_TOPICS, ABCMcapStore  # noqa: E402

CAMERAS = ["top", "left", "right"]


def state_topic_extent(ep_dir: pathlib.Path) -> tuple[int, int]:
    """min/max timestamp over the state topics -- what defines the 10 Hz grid.

    Deliberately re-derived from the MCAP rather than taken from the camera axis:
    build_dataset.py spans the state streams, and the two differ by a couple of
    frames per episode.
    """
    # Identical to build_dataset.py's STATE_CHANNELS topic set.
    wanted = {topic for topic, _w in VECTOR_TOPICS["state"]}
    lo, hi = None, None
    with (ep_dir / "episode.mcap").open("rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _s, ch, msg, _dec in reader.iter_decoded_messages():
            if ch.topic in wanted:
                lo = msg.log_time if lo is None else min(lo, msg.log_time)
                hi = msg.log_time if hi is None else max(hi, msg.log_time)
    if lo is None:
        raise ValueError(f"{ep_dir}: no state topics, cannot build a grid")
    return int(lo), int(hi)


def heal_sidecars(store, ep: int, ep_dir: pathlib.Path) -> None:
    """Re-remux camera sidecars reaped after `index.json` was written.

    `ABCMcapStore._ensure_index` trusts an existing index.json without checking
    that the mp4s it names are still on disk, so a deleted sidecar surfaces as a
    FileNotFoundError from `av.open` deep inside the decode instead of as a
    rebuild. `reap_sidecars.sh` drops left/right once an episode's frames are in
    the zarr -- right for the 1,009 legos episodes that are in it, wrong for the
    3 that are not and are therefore read from the mcap here.

    Deleting index.json is the whole repair: the next `_ensure_index` rebuilds
    it, and `_remux` returns early for any camera whose mp4 survived, so only
    the missing streams are re-extracted. That costs a minute an episode against
    the alternative of the task dying at episode 605 of 1,012.
    """
    root = (
        pathlib.Path(store._cache_root(ep))
        if hasattr(store, "_cache_root")
        else ep_dir / ".robo_dataset"
    )
    idx_path = root / "index.json"
    if not idx_path.exists():
        return  # nothing to distrust; _ensure_index will build it
    try:
        files = json.loads(idx_path.read_text()).get("camera_files") or {}
    except (OSError, ValueError):
        return
    gone = []
    for cam in CAMERAS:
        f = root / files.get(cam, f"{cam}.mp4")
        if not f.exists() or f.stat().st_size == 0:
            gone.append(cam)
    if not gone:
        return
    print(f"  {ep_dir.name}: sidecars {', '.join(gone)} gone; re-remuxing", flush=True)
    idx_path.unlink()
    # The stale index and its decoders may already be held in memory, since
    # frame_times_ns() reads the index before we get here.
    getattr(store, "_index", {}).pop(ep, None)
    # `_decoder_cache` is an _LRUDecoderCache, not a dict: it exposes
    # get/put/clear and is not iterable, so evicting this episode's entries
    # selectively is not available. Clearing all of it is fine -- the entries
    # are open decoders over a bounded LRU, and reopening one is milliseconds
    # against the remux this function is about to trigger.
    dec = getattr(store, "_decoder_cache", None)
    if hasattr(dec, "clear"):
        dec.clear()


def letterbox(frames: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Same geometry as robo_dataset's ResizeImages(letterbox=True)."""
    n, h, w = frames.shape[:3]
    scale = min(th / h, tw / w)
    sh = max(1, min(th, int(round(h * scale))))
    sw = max(1, min(tw, int(round(w * scale))))
    y0, x0 = (th - sh) // 2, (tw - sw) // 2
    out = np.zeros((n, th, tw, frames.shape[3]), dtype=frames.dtype)
    for i in range(n):
        out[i, y0 : y0 + sh, x0 : x0 + sw] = cv2.resize(
            frames[i], (sw, sh), interpolation=cv2.INTER_AREA
        )
    return out


device = torch.device(args.device)
policy, _store, _ = load_policy(
    pathlib.Path(args.checkpoint),
    InferenceConfig(compile=False, cuda_graph=False, use_ema=True),
    device,
)
policy.eval()

# Last DiT block: named rather than hardcoded, so depth changes don't silently
# tap the wrong layer.
n_blocks = len(policy.net.blocks)
TAPS = {
    "enc": "enc",  # observation encoder output
    "cond": "obs_proj",  # the conditioning vector
    "dit": f"net.blocks.{n_blocks - 1}",  # last DiT block hidden state
}
capture = ActivationCapture(policy, list(TAPS.values()))
print(f"taps: {TAPS}  (DiT depth {n_blocks})")

src = pathlib.Path(args.source)
store = ABCMcapStore(str(src), cameras=CAMERAS, stride=1, require_annotations=True)

# Optional fast path. The cache is keyed by its own episode order (it covers the
# whole task, not just the annotated slice, and skips episodes missing a camera),
# so episodes are matched by directory NAME rather than position. Its frame j is
# native frame j*stride, which makes the grid -> cache mapping exact.
cache_store = cache_pos = cache_stride = None
if args.cache:
    from robo_dataset.store_factory import open_store

    cache_dir = pathlib.Path(args.cache)
    order = json.loads((cache_dir / "abc_episode_order.json").read_text())
    cache_stride = int(order["stride"])
    if (order["height"], order["width"]) != (args.height, args.width):
        raise SystemExit(
            f"cache is {order['height']}x{order['width']} but --height/--width "
            f"say {args.height}x{args.width}; the policy would see the wrong size"
        )
    cache_pos = {name: i for i, name in enumerate(order["episode_dirs"])}
    cache_store = open_store(str(cache_dir))
    print(
        f"frames from cache {cache_dir.name} "
        f"({cache_store.n_episodes} episodes, stride {cache_stride})"
    )

n_ep = args.limit_episodes or store.n_episodes
print(f"{n_ep} episodes; grid {args.rate} Hz; input {args.height}x{args.width}")

collected: dict[str, list[np.ndarray]] = {k: [] for k in TAPS}
lengths = []
n_fallback = 0
step_ns = int(round(1e9 / args.rate))

# Created once for the whole run: a pool per episode would pay thread setup
# 1,012 times and, worse, let a slow episode's threads outlive it.
cam_pool = ThreadPoolExecutor(max_workers=args.decode_workers) if args.decode_workers else None
pre_pool = ThreadPoolExecutor(max_workers=1)

for ep in range(n_ep):
    t_ep = time.time()
    ep_dir = pathlib.Path(store.episode_dir(ep))
    # Resolved here rather than after the frame-axis work below: heal_sidecars
    # has to run before anything reads this episode's index.
    name = ep_dir.name
    from_mcap = cache_store is None or name not in cache_pos
    if from_mcap:
        heal_sidecars(store, ep, ep_dir)
    t0, t1 = state_topic_extent(ep_dir)
    grid = np.arange(t0, t1 + 1, step_ns, dtype=np.int64)
    cam_times = store.frame_times_ns(ep)  # stride=1 camera axis
    # nearest camera frame for each grid instant
    pos = np.searchsorted(cam_times, grid)
    pos = np.clip(pos, 1, len(cam_times) - 1)
    left = cam_times[pos - 1]
    take = np.where(np.abs(grid - left) <= np.abs(cam_times[pos] - grid), pos - 1, pos)

    # Resolve where this episode's frames come from.
    if from_mcap and cache_store is not None:
        # Fall back to the MCAP sidecar rather than skip. The output is one
        # concatenated array split downstream by the study dataset's own
        # per-episode lengths, so dropping an episode does not leave a gap --
        # it shifts every episode after it onto the wrong frames, silently.
        # Legos hits this: build_zarr_cache omits episodes missing a camera, so
        # 3 of its 1,012 annotated episodes are absent, and skipping them would
        # have misattributed the 404 that follow -- across both evaluation sets.
        #
        # Slower for those episodes (the h264 sidecars are inter-frame coded, so
        # sampling every third frame still decodes the two between) but correct,
        # and it is a handful of episodes rather than the task.
        print(f"  ep{ep:03d} {name} absent from cache; reading the MCAP instead", flush=True)
        n_fallback += 1
        src_store, src_ep = store, ep
    elif cache_store is not None:
        c_ep = cache_pos[name]
        c_len = cache_store.episode_length(c_ep)
        # native frame f  ->  cache frame round(f / stride)
        take = np.clip((take + cache_stride // 2) // cache_stride, 0, c_len - 1)
        src_store, src_ep = cache_store, c_ep
    else:
        src_store, src_ep = store, ep

    # State for the whole episode in one read: mmapped .npy (or one zarr slice),
    # so this is nearly free and saves a call per frame.
    state_all = src_store.get_frames(src_ep, "state", 0, src_store.episode_length(src_ep))

    per_tap: dict[str, list[np.ndarray]] = {k: [] for k in TAPS}

    def decode(lo: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Host-side frames for one batch: the part that is not the GPU's.

        One contiguous decode per camera per batch, not one per frame. The 10 Hz
        grid lands on roughly every third native frame, so a span read decodes
        ~3x more frames than are kept -- but sequentially, in a single call,
        instead of paying a random seek (~175 ms on this 250-GOP h264) for every
        one of batch x cameras frames.

        The three cameras go in parallel. Profiled, decode was 68% of this
        script's wall clock against 27% for the forward pass, and the GPU sat at
        0-2% SM throughout; the decode releases the GIL, so threads convert that
        directly (2.12x on a cold chunk cache).
        """
        idx = take[lo : lo + args.batch]
        span_lo, span_hi = int(idx.min()), int(idx.max()) + 1
        rel = idx - span_lo

        def one(cam: str) -> tuple[str, np.ndarray]:
            span = src_store.get_frames(src_ep, cam, span_lo, span_hi)
            frames = span[rel]
            # The cache is already letterboxed to the target size; only the raw
            # sidecar path needs resizing.
            if frames.shape[1:3] != (args.height, args.width):
                frames = letterbox(frames, args.height, args.width)
            return cam, np.ascontiguousarray(frames)

        if cam_pool is None:
            frames = dict(one(cam) for cam in CAMERAS)
        else:
            frames = dict(cam_pool.map(one, CAMERAS))
        return idx, frames

    # Decode batch i+1 while the GPU runs batch i. Two pools rather than one:
    # `decode` submits the cameras to cam_pool, so running it inside the same
    # pool could deadlock once its workers are all held by outer tasks.
    starts = list(range(0, len(grid), args.batch))
    pending = pre_pool.submit(decode, starts[0]) if starts else None
    for pos, lo in enumerate(starts):
        idx, frames = pending.result()
        # Queue the next decode before touching the GPU, so it overlaps.
        pending = pre_pool.submit(decode, starts[pos + 1]) if pos + 1 < len(starts) else None
        n = len(idx)
        batch = {
            # (B, T=1, H, W, C) -- obs_horizon=1
            cam: torch.from_numpy(a).unsqueeze(1).to(device)
            for cam, a in frames.items()
        }
        batch["state"] = torch.from_numpy(state_all[idx]).unsqueeze(1).to(device)

        with torch.no_grad():
            prepared = apply_pipeline(policy.data_transforms_pipeline, batch)
            obs_vec = policy.obs_proj(policy.enc(prepared))
            # One flow step is enough to populate the DiT blocks; the features
            # wanted are the conditioning and hidden state, not the action.
            x = torch.zeros(
                n, policy.cfg.horizon, policy.act_dim, device=device, dtype=obs_vec.dtype
            )
            t = torch.full((n,), 1.0, device=device, dtype=obs_vec.dtype)
            policy.net(x, policy._cond(obs_vec, t))
        got = capture.pop()
        for name, path in TAPS.items():
            a = got[path]
            per_tap[name].append(a.reshape(a.shape[0], -1) if a.ndim == 2 else a.mean(axis=1))

    for name in TAPS:
        collected[name].append(np.concatenate(per_tap[name], axis=0))
    lengths.append(len(grid))
    dt = time.time() - t_ep
    print(
        f"  ep{ep:03d} {len(grid):5d} frames in {dt:5.1f}s "
        f"({len(grid) / max(dt, 1e-9):6.0f} fps)  "
        + "  ".join(f"{k}{collected[k][-1].shape[1]}" for k in TAPS),
        flush=True,
    )

capture.remove()
pre_pool.shutdown(wait=True)
if cam_pool is not None:
    cam_pool.shutdown(wait=True)
out_dir = pathlib.Path(args.out_root) / "policy_embeddings"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"\ntotal frames: {sum(lengths)}  ({len(lengths)} episodes, {n_fallback} read from MCAP)")

# The features are split downstream by the study dataset's per-episode lengths,
# so one episode short of the dataset misattributes every episode after it. This
# is the check that the frame axis still lines up; it costs one metadata read.
ds_meta = out_dir.parent / "episodes" / "metadata.yaml"
if ds_meta.exists():
    import yaml

    m = yaml.safe_load(ds_meta.read_text())
    ds_eps = m["episodes"] if isinstance(m, dict) and "episodes" in m else m
    if len(lengths) != len(ds_eps):
        raise SystemExit(
            f"extracted {len(lengths)} episodes but the dataset at {out_dir.parent} has "
            f"{len(ds_eps)}; writing this would shift the frame axis. Nothing written."
        )
    print(f"  episode count matches the dataset ({len(ds_eps)})")

for name in TAPS:
    arr = np.concatenate(collected[name], axis=0).astype(np.float32)
    path = out_dir / f"{args.prefix}_{name}.npz"
    np.savez_compressed(path, embeddings=arr)
    print(f"  wrote {path}  {arr.shape}")
meta = {
    "lengths": lengths,
    "rate_hz": args.rate,
    "checkpoint": args.checkpoint,
    "taps": TAPS,
    "input_hw": [args.height, args.width],
}
(out_dir / f"{args.prefix}_meta.json").write_text(json.dumps(meta, indent=2))
