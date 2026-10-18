#!/usr/bin/env python
"""Decode ABC-130k camera streams from MCAP into per-episode frame arrays.

ABC stores video as one `foxglove.CompressedVideo` protobuf message per frame,
each holding an h264 Annex-B access unit (the payloads begin with the 00 00 00 01
start code, and the first carries SPS/PPS). Concatenating a camera's payloads in
log-time order therefore yields a valid raw h264 elementary stream, which ffmpeg
decodes directly -- no container rewrite needed.

Frames are decoded at full rate, then subsampled onto the same uniform grid
`build_dataset.py` used for the state, so image features and proprioception line
up index-for-index. Output is one uint8 .npy per (episode, camera) at the DiT's
native 224x224, already resize-with-pad'ed the way `abc_minimal.preprocess` does
it, so the embedding pass is a pure forward.

    python scripts/ablation_study/extract_frames.py --task put_the_plastic_bottles_in_the_bin \\
        --out $DATA_PATH/abc_bottles_frames --rate 10
"""

from __future__ import annotations

import argparse
import sys as _sys

_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import pathlib
import subprocess
import tempfile

import _paths  # noqa: E402
import numpy as np
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

CAMERA_TOPICS = {
    "top": "/top-camera",
    "left": "/left-wrist-camera",
    "right": "/right-wrist-camera",
}
STATE_TOPICS = (
    "/left-arm-state",
    "/left-ee-state",
    "/right-arm-state",
    "/right-ee-state",
)


def read_packets(path: pathlib.Path) -> tuple[dict[str, list[tuple[int, bytes]]], tuple[int, int]]:
    """(camera -> [(log_time, h264 payload)], (state_t0, state_t1)).

    The state span is returned alongside because the resample grid has to be the
    same one `build_dataset.py` derives, or the frames and the state vectors would
    be indexed differently.
    """
    want = set(CAMERA_TOPICS.values())
    out: dict[str, list[tuple[int, bytes]]] = {c: [] for c in CAMERA_TOPICS}
    by_topic = {v: k for k, v in CAMERA_TOPICS.items()}
    starts, ends = [], []
    with path.open("rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic in want:
                out[by_topic[channel.topic]].append((message.log_time, bytes(decoded.data)))
            elif channel.topic in STATE_TOPICS:
                starts.append(message.log_time)
                ends.append(message.log_time)
    for v in out.values():
        v.sort(key=lambda x: x[0])
    if not starts:
        raise SystemExit(f"{path}: no state messages, cannot build a shared time grid")
    return out, (min(starts), max(ends))


def decode_stream(packets: list[tuple[int, bytes]], width: int, height: int) -> np.ndarray:
    """Decode concatenated Annex-B packets to (N, H, W, 3) uint8 via ffmpeg."""
    if not packets:
        return np.zeros((0, height, width, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=True) as raw:
        for _t, payload in packets:
            raw.write(payload)
        raw.flush()
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "h264",
                "-i",
                raw.name,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[:400]}")
    frame_bytes = width * height * 3
    n = len(proc.stdout) // frame_bytes
    if n == 0:
        raise RuntimeError("ffmpeg produced no frames")
    return np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8).reshape(
        n, height, width, 3
    )


def probe_size(packets: list[tuple[int, bytes]]) -> tuple[int, int]:
    """(width, height) from the stream itself, so nothing is hard-coded."""
    with tempfile.NamedTemporaryFile(suffix=".h264", delete=True) as raw:
        for _t, payload in packets[:60]:
            raw.write(payload)
        raw.flush()
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                raw.name,
            ],
            capture_output=True,
        )
    txt = proc.stdout.decode().strip().split("\n")[0]
    w, h = (int(x) for x in txt.split(",")[:2])
    return w, h


def resize_pad_uint8(frames: np.ndarray, size: int = 224) -> np.ndarray:
    """Aspect-preserving resize + centre pad to (size, size), matching the DiT's
    `resize_with_pad`, done with ffmpeg-free numpy/cv2 for speed."""
    import cv2

    n, h, w, _ = frames.shape
    ratio = max(w / size, h / size)
    nh, nw = max(1, round(h / ratio)), max(1, round(w / ratio))
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.zeros((n, size, size, 3), dtype=np.uint8)
    for i in range(n):
        r = cv2.resize(frames[i], (nw, nh), interpolation=cv2.INTER_AREA)
        out[i, top : top + nh, left : left + nw] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=_paths.env_path("${DATA_PATH}/abc_130k"),
    )
    ap.add_argument("--task", default="put_the_plastic_bottles_in_the_bin")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = args.root / "data" / "train" / args.task
    eps = sorted(
        p
        for p in src.glob("*")
        if (p / "annotation.mcap").exists() and (p / "episode.mcap").exists()
    )
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        raise SystemExit(f"no episodes with both mcaps under {src}")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[frames] {len(eps)} episodes, rate {args.rate}Hz, {args.size}px", flush=True)

    for d in eps:
        dest = args.out / f"{d.name}.npz"
        if dest.exists():
            print(f"  have {d.name}")
            continue
        packets, (t0, t1) = read_packets(d / "episode.mcap")
        step_ns = int(round(1e9 / args.rate))
        grid = np.arange(t0, t1 + 1, step_ns, dtype=np.int64)

        per_cam = {}
        for cam, pkts in packets.items():
            if not pkts:
                print(f"  ! {d.name}: camera {cam} has no packets")
                continue
            w, h = probe_size(pkts)
            frames = decode_stream(pkts, w, h)
            times = np.asarray([t for t, _ in pkts], dtype=np.int64)[: len(frames)]
            # nearest decoded frame for each grid point, same rule as the state
            idx = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
            take_left = (grid - times[idx - 1]) <= (times[idx] - grid)
            chosen = np.where(take_left, idx - 1, idx)
            per_cam[cam] = resize_pad_uint8(frames[chosen], args.size)

        if len(per_cam) != len(CAMERA_TOPICS):
            print(f"  ! {d.name}: only {sorted(per_cam)} decoded, skipped")
            continue
        np.savez_compressed(dest, **per_cam)
        print(
            f"  {d.name}: {len(grid)} grid frames x {len(per_cam)} cams "
            f"-> {dest.stat().st_size / 1e6:.0f} MB",
            flush=True,
        )


if __name__ == "__main__":
    main()
