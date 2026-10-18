#!/usr/bin/env python
"""Convert downloaded ABC-130k episodes into the canonical trajectory layout.

Deliberately emits *exactly* the layout `behavior_graphs.wgo_bench.data.WGODroid`
already reads, so the whole evaluation stack built for WGO-Bench -- boundary and
segmental metrics with their count-matched uniform references, scene-grouped
splits, 1-SE/seed-averaged selection, scene bootstraps, the selector family --
runs on ABC with no changes beyond pointing it at a different directory. The
package name says `wgo_bench`; the code inside it is dataset-agnostic apart from
`data.py`.

What ABC gives us, read off the files rather than the (gated) format doc:

* `annotation.mcap`: one channel `/subtask-annotation`, protobuf `Annotation`
  with `.data` (subtask text) and `.timestamp`. These are **start markers**, not
  intervals, so segment *i* runs from marker *i* to marker *i+1* and the last one
  runs to the end of the recording.
* `episode.mcap`: `/instruction` (one message), three h264 camera streams at
  ~60 Hz, and `{left,right}-{arm,ee}-{state,action}` at ~200-276 Hz. `RobotState.position`
  is 6 joint angles per arm; `GripperState.position` is a single aperture.

Streams sit on independent clocks -- the four state channels here have message
counts differing by tens -- so the README's instruction is followed literally:
resample onto one uniform grid by nearest-neighbour on absolute timestamps.

    python scripts/ablation_study/build_dataset.py --out $DATA_PATH/abc_legos --rate 10
"""

from __future__ import annotations

import argparse
import sys as _sys

_sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import json
import pathlib

import _paths  # noqa: E402
import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

ANNOTATION_TOPIC = "/subtask-annotation"
INSTRUCTION_TOPIC = "/instruction"
#: Channel -> width. The concatenation order defines the state vector, and is
#: fixed here so an embedding extracted today matches one extracted later.
STATE_CHANNELS = (
    ("/left-arm-state", 6),
    ("/left-ee-state", 1),
    ("/right-arm-state", 6),
    ("/right-ee-state", 1),
)
ACTION_CHANNELS = (
    ("/left-arm-action", 6),
    ("/left-ee-action", 1),
    ("/right-arm-action", 6),
    ("/right-ee-action", 1),
)


def read_streams(path: pathlib.Path, topics: set[str]) -> dict[str, list[tuple[int, list]]]:
    """topic -> [(log_time_ns, position list)] for the requested topics."""
    out: dict[str, list[tuple[int, list]]] = {t: [] for t in topics}
    with path.open("rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, decoded in reader.iter_decoded_messages():
            if channel.topic not in topics:
                continue
            if channel.topic in (ANNOTATION_TOPIC, INSTRUCTION_TOPIC):
                out[channel.topic].append((message.log_time, str(decoded.data)))
            else:
                out[channel.topic].append((message.log_time, list(decoded.position)))
    for v in out.values():
        v.sort(key=lambda x: x[0])
    return out


def resample_nearest(stream: list[tuple[int, list]], grid_ns: np.ndarray, width: int) -> np.ndarray:
    """Nearest-neighbour resample a (time, vector) stream onto ``grid_ns``.

    ABC's streams are on independent clocks, so this is what the dataset card
    prescribes. Nearest rather than interpolated because a gripper aperture that
    is being commanded open/closed should not be smeared across the transition --
    and the transition is exactly what we are trying to detect.
    """
    if not stream:
        return np.zeros((len(grid_ns), width), dtype=np.float32)
    times = np.asarray([t for t, _ in stream], dtype=np.int64)
    vals = np.asarray([(v + [0.0] * width)[:width] for _, v in stream], dtype=np.float32)
    idx = np.searchsorted(times, grid_ns)
    idx = np.clip(idx, 1, len(times) - 1)
    left, right = times[idx - 1], times[idx]
    take_left = (grid_ns - left) <= (right - grid_ns)
    chosen = np.where(take_left, idx - 1, idx)
    return vals[chosen]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=pathlib.Path,
        default=_paths.env_path("${DATA_PATH}/abc_130k"),
    )
    ap.add_argument("--task", default="sort_the_legos_into_containers_by_color")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--rate", type=float, default=10.0, help="resample rate in Hz")
    ap.add_argument("--actions", action="store_true", help="also write actions.npz")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = args.root / "data" / "train" / args.task
    ep_dirs = sorted(p for p in src.glob("*") if (p / "annotation.mcap").exists())
    if args.limit:
        ep_dirs = ep_dirs[: args.limit]
    if not ep_dirs:
        raise SystemExit(f"no episodes with annotation.mcap under {src}")

    have_full = [p for p in ep_dirs if (p / "episode.mcap").exists()]
    print(
        f"[build] {len(ep_dirs)} annotated episodes, {len(have_full)} with episode.mcap "
        f"(state lives there, so only those can be converted)",
        flush=True,
    )
    ep_dirs = have_full
    if not ep_dirs:
        raise SystemExit("no episode.mcap present -- re-run download_samples.py with --with-video")

    for d in (args.out / "episodes", args.out / "observations", args.out / "annotations"):
        d.mkdir(parents=True, exist_ok=True)

    state_topics = {t for t, _ in STATE_CHANNELS}
    action_topics = {t for t, _ in ACTION_CHANNELS} if args.actions else set()

    episodes, annotations, states, actions = [], [], [], []
    for index, d in enumerate(ep_dirs):
        ann = read_streams(d / "annotation.mcap", {ANNOTATION_TOPIC})[ANNOTATION_TOPIC]
        streams = read_streams(
            d / "episode.mcap", state_topics | action_topics | {INSTRUCTION_TOPIC}
        )
        instruction = streams[INSTRUCTION_TOPIC][0][1] if streams[INSTRUCTION_TOPIC] else args.task

        # One grid spanning the state streams; annotations are placed on it by time.
        starts = [s[0][0] for t, s in streams.items() if t in state_topics and s]
        ends = [s[-1][0] for t, s in streams.items() if t in state_topics and s]
        if not starts:
            print(f"  ep{index:04d} {d.name}: no state channels, skipped", flush=True)
            continue
        t0, t1 = min(starts), max(ends)
        step_ns = int(round(1e9 / args.rate))
        grid = np.arange(t0, t1 + 1, step_ns, dtype=np.int64)

        state = np.concatenate(
            [resample_nearest(streams[t], grid, w) for t, w in STATE_CHANNELS], axis=1
        )
        if args.actions:
            actions.append(
                np.concatenate(
                    [resample_nearest(streams[t], grid, w) for t, w in ACTION_CHANNELS],
                    axis=1,
                )
            )

        duration = (t1 - t0) / 1e9
        marks = [((t - t0) / 1e9, text) for t, text in ann]
        segs = []
        for i, (start, text) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else duration
            if end > start:
                segs.append({"start_sec": float(start), "end_sec": float(end), "subtask": text})
        if len(segs) < 2:
            print(f"  ep{index:04d} {d.name}: {len(segs)} segments, skipped", flush=True)
            continue

        episodes.append(
            {
                "index": index,
                "length": int(len(grid)),
                "success": True,
                "wgo_id": d.name,
                "instruction": instruction,
                "fps": float(args.rate),
                "duration_sec": float(duration),
                # Each ABC episode is its own recording: no duplicated camera
                # views, so scene == episode and the scene grouping is a no-op.
                "source_episode_id": d.name,
                "source_camera_view": "top",
                "n_segments": len(segs),
            }
        )
        annotations.append(
            {
                "index": index,
                "wgo_id": d.name,
                "instruction": instruction,
                "fps": float(args.rate),
                "length": int(len(grid)),
                "duration_sec": float(duration),
                "segments": segs,
            }
        )
        states.append(state.astype(np.float32))
        print(
            f"  ep{index:04d} {d.name}: {len(grid)} frames @ {args.rate:g}Hz "
            f"({duration:.1f}s), {len(segs)} segments, state {state.shape[1]}d",
            flush=True,
        )

    if not episodes:
        raise SystemExit("nothing converted")

    total = sum(e["length"] for e in episodes)
    state_all = np.concatenate(states, axis=0)
    assert state_all.shape[0] == total, (state_all.shape, total)

    (args.out / "episodes" / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "source": {
                    "kind": "benchmark",
                    "dataset": "XDOF/ABC-130k",
                    "task": args.task,
                    "rate_hz": args.rate,
                    "state_layout": [f"{t}:{w}" for t, w in STATE_CHANNELS],
                },
                "episodes": episodes,
            },
            sort_keys=False,
        )
    )
    np.savez(args.out / "observations" / "state.npz", observations=state_all)
    if args.actions and actions:
        np.savez(args.out / "actions.npz", actions=np.concatenate(actions, axis=0))
    (args.out / "annotations" / "segments.json").write_text(json.dumps(annotations, indent=2))

    nseg = sum(e["n_segments"] for e in episodes)
    print(
        f"[build] {len(episodes)} episodes, {total} frames, {nseg} segments "
        f"({nseg - len(episodes)} interior boundaries) -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
