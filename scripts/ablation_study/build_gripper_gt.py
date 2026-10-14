"""Cheap subtask ground truth for a DROID rollout dir, cut at the gripper events.

Hand annotation is the bottleneck on every new task, and on a prehensile task
most subtask boundaries are a grasp or a release -- which the robot already
recorded. This writes that skeleton out as ground truth for a whole dataset in
one pass, in both formats the codebase reads:

  annotations/subtasks/ep*.json   the annotation tool's per-episode files, so
                                  the app shows the spans and a human can
                                  correct them by dragging edges
  annotations/segments.json       the ABC study's format, which
                                  `_boundary_metrics.build_ground_truth` reads
                                  and every segmentation metric scores against

**What this is not.** It is not a segmentation method being evaluated, and it is
not a substitute for watching the video. Two subtasks that hand over without the
gripper moving become one segment; a regrasp inside one subtask becomes two. Used
as ground truth it therefore measures "can this representation recover the
gripper's own event structure", which is a *proxy* for subtask structure and
overstates any representation that can see the gripper. See
the circularity note below.

**The circularity to keep in mind.** Any representation carrying the gripper
channel -- raw proprioception most obviously, but also a policy embedding that
conditions on proprioception -- has partial access to the signal these labels
were derived from. A score here is an upper bound for such an arm, and the
comparison that stays meaningful is *between* arms that share the same access.
`scripts/ablation_study/build_droid_state.py` writes a proprioception array with the
gripper channel removed for exactly this reason.

Provenance is recorded rather than implied: every file is signed by
``--annotator`` (default ``gripper-skeleton``, not a person), and carries a note
saying what produced it and with which settings. `reviewed` is set true because
the annotation tool exports reviewed episodes only -- so the flag here means
"complete enough to score", and the annotator field is what says no human
watched it.

Usage::

    python scripts/ablation_study/build_gripper_gt.py \\
        ${DATA_PATH}/rollouts/cooking_baseline_dit_200

    # a different settle time, and see what it would do without writing
    python scripts/ablation_study/build_gripper_gt.py <dir> --settle 1.0 --dry-run
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_graphs.annotations import gripper, subtasks  # noqa: E402

#: What a span is called. Deliberately uninformative: the gripper says *where* a
#: subtask boundary is, never what the subtask was, and a made-up label would be
#: indistinguishable from a real one once it is in the file. The ABC format has
#: no unlabelled state -- an empty string would be read as a segment named "" --
#: so every span gets the same name, which `build_ground_truth` treats as task
#: structure (it is not in NON_TASK_LABELS) while naming nothing.
SEGMENT_LABEL = "segment"

#: Minimum spans for an episode to be usable as ground truth: a single span over
#: the whole episode carries no interior boundary, so it contributes nothing to a
#: boundary metric and `build_dataset.py` skips it for the same reason.
MIN_SPANS = 2


def episodes_of(rollout_dir: pathlib.Path) -> list[dict]:
    meta = yaml.safe_load((rollout_dir / "episodes" / "metadata.yaml").read_text())
    return list(meta["episodes"])


def rate_of(rollout_dir: pathlib.Path, override: float | None) -> float:
    """The control rate, from the collector's own record unless overridden.

    This is the axis the annotation is on -- the policy stepped at it and the
    telemetry has one row per step -- and it is not the clip's frame rate.
    """
    if override:
        return float(override)
    for ep in episodes_of(rollout_dir):
        args = (ep.get("args") or {}).get("rollout_env") or {}
        for key in ("control_hz", "policy_hz"):
            value = args.get(key) or (ep.get("args") or {}).get(key)
            if isinstance(value, int | float) and value and value > 0:
                return float(value)
    raise SystemExit(
        f"no control rate recorded in {rollout_dir}/episodes/metadata.yaml; pass --rate"
    )


def build(
    rollout_dir: pathlib.Path,
    *,
    rate: float,
    closed_above: float,
    settle_sec: float,
) -> tuple[list[dict], list[tuple[int, str]]]:
    """``(segments.json records, skipped)`` for every episode of the dataset."""
    records: list[dict] = []
    skipped: list[tuple[int, str]] = []
    for i, ep in enumerate(episodes_of(rollout_dir)):
        frames = int(ep["length"])
        trace = gripper.trace(rollout_dir, i, num_frames=frames)
        if trace is None:
            skipped.append((i, "no gripper telemetry matching this episode"))
            continue
        spans = gripper.skeleton(
            trace,
            frames,
            fps=rate,
            closed_above=closed_above,
            min_dwell_sec=settle_sec,
        )
        if len(spans) < MIN_SPANS:
            skipped.append((i, f"{len(spans)} span(s), no interior boundary"))
            continue
        records.append(
            {
                "index": i,
                "length": frames,
                "fps": rate,
                "duration_sec": frames / rate,
                "success": ep.get("success"),
                "instruction": _instruction(ep),
                "segments": [
                    {
                        "start_sec": a / rate,
                        "end_sec": b / rate,
                        "subtask": SEGMENT_LABEL,
                    }
                    for a, b in spans
                ],
                "_spans_frames": spans,
            }
        )
    return records, skipped


def _instruction(ep: dict) -> str:
    args = ep.get("args") or {}
    for value in (args.get("instruction"), (args.get("inference") or {}).get("instruction")):
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def write_app_format(
    rollout_dir: pathlib.Path,
    records: list[dict],
    *,
    dataset: str,
    annotator: str,
    note: str,
) -> int:
    """Per-episode files for the annotation page, so a human can correct them."""
    written = 0
    for rec in records:
        ann = subtasks.EpisodeAnnotation(
            dataset=dataset,
            episode=int(rec["index"]),
            num_frames=int(rec["length"]),
            fps=float(rec["fps"]),
            success=rec.get("success"),
            task=rec.get("instruction") or "",
            segments=[
                subtasks.Segment(id=subtasks.new_segment_id(), start=a, end=b)
                for a, b in rec["_spans_frames"]
            ],
            notes=note,
            # The tool exports reviewed episodes only, so this has to be set for
            # the annotation to count at all. It means "complete enough to
            # score" here; `annotator` is what records that no human watched it.
            reviewed=True,
            annotator=annotator,
        )
        subtasks.save(rollout_dir, ann)
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rollout_dir", type=pathlib.Path)
    ap.add_argument(
        "--rate",
        type=float,
        default=None,
        help="control rate in Hz (default: whatever the collector recorded)",
    )
    ap.add_argument(
        "--closed-above",
        type=float,
        default=gripper.CLOSED_ABOVE,
        help=f"closedness that counts as closed (default {gripper.CLOSED_ABOVE})",
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=gripper.MIN_DWELL_SEC,
        help=f"seconds a new gripper state must hold (default {gripper.MIN_DWELL_SEC})",
    )
    ap.add_argument(
        "--annotator",
        default="gripper-skeleton",
        help="what signs the files; keep it non-human so the provenance survives",
    )
    ap.add_argument(
        "--no-app-format",
        action="store_true",
        help="write only annotations/segments.json, not the per-episode tool files",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rollout_dir = args.rollout_dir
    if not (rollout_dir / "episodes" / "metadata.yaml").is_file():
        raise SystemExit(f"{rollout_dir} is not a canonical rollout dir")

    rate = rate_of(rollout_dir, args.rate)
    records, skipped = build(
        rollout_dir,
        rate=rate,
        closed_above=args.closed_above,
        settle_sec=args.settle,
    )

    n_seg = sum(len(r["segments"]) for r in records)
    lengths = np.array([s["end_sec"] - s["start_sec"] for r in records for s in r["segments"]])
    per_ep = np.array([len(r["segments"]) for r in records])
    print(f"{rollout_dir.name} at {rate:g} Hz")
    print(f"  closed above {args.closed_above:.2f}, settling {args.settle:.1f}s")
    print(f"  {len(records)} episodes usable, {len(skipped)} skipped")
    print(f"  {n_seg} segments, {n_seg - len(records)} interior boundaries")
    if per_ep.size:
        print(
            f"  segments per episode: min {per_ep.min()} "
            f"median {int(np.median(per_ep))} max {per_ep.max()}"
        )
        print(
            f"  segment seconds: p10 {np.percentile(lengths, 10):.1f} "
            f"median {np.median(lengths):.1f} p90 {np.percentile(lengths, 90):.1f}"
        )
    for i, why in skipped[:10]:
        print(f"  skipped ep{i:04d}: {why}")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    note = (
        f"Machine-made skeleton: boundaries at every gripper state change held "
        f"{args.settle:g}s, closedness threshold {args.closed_above:g}, "
        f"{rate:g} Hz. Not reviewed by a human. "
        f"scripts/ablation_study/build_gripper_gt.py"
    )
    stamped = datetime.datetime.now().isoformat(timespec="seconds")

    out = rollout_dir / "annotations" / "segments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out} ({len(payload)} episodes)")

    # Provenance beside the file rather than inside it: segments.json has a
    # fixed schema that build_ground_truth reads positionally, and a reader that
    # chokes on an unexpected key is not worth risking for a note.
    sidecar = out.with_name("segments_provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "generator": "scripts/ablation_study/build_gripper_gt.py",
                "written_at": stamped,
                "annotator": args.annotator,
                "rate_hz": rate,
                "closed_above": args.closed_above,
                "settle_sec": args.settle,
                "n_episodes": len(payload),
                "n_segments": n_seg,
                "n_interior_boundaries": n_seg - len(payload),
                "skipped": [{"index": i, "why": w} for i, w in skipped],
                "caveat": (
                    "Gripper-derived, not human. Overstates any representation "
                    "carrying the gripper channel."
                ),
            },
            indent=2,
        )
    )
    print(f"wrote {sidecar}")

    if not args.no_app_format:
        n = write_app_format(
            rollout_dir,
            records,
            dataset=rollout_dir.name,
            annotator=args.annotator,
            note=note,
        )
        print(f"wrote {n} per-episode files under {subtasks.store_dir(rollout_dir)}")


if __name__ == "__main__":
    main()
