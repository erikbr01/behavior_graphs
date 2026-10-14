"""Standalone VLM node labeler for an existing run directory.

Usage:
    python scripts/label_nodes.py <run_dir> [options]

Example:
    python scripts/label_nodes.py deploy/data/runs/lift_mh_jan26__agg_first__infembed_w3_s1_seed0_kmeans_k10 \
        --rollout-dir deploy/data/rollouts/lift_mh_jan26 \
        --task "pick up the red block and lift it" \
        --gcp-project gcp-driven-data --gcp-location global

Multi-camera datasets label from one view unless told otherwise:

    python scripts/label_nodes.py <run_dir> --list-cameras
    python scripts/label_nodes.py <run_dir> \
        --camera exterior_image_1_left --camera wrist_image_left
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _camera_names(rollout_dir: Path) -> list[str]:
    from behavior_graphs.envs.trajectory_dir import RolloutDirectory

    return RolloutDirectory(rollout_dir).camera_names()


def _resolve_cameras(rollout_dir: Path, requested: list[str] | None) -> list[str] | None:
    """Validate ``--camera`` against the rollout dir, or ``None`` for the default.

    Unknown names exit rather than being dropped: an unmatched view contributes
    no rows at all, so a typo would otherwise buy a full-price labeling pass over
    sheets missing the view it was run for — and, if it was the only ``--camera``,
    over empty sheets.

    ``None`` rather than the declared primary view spelled out as a one-element
    list: :class:`~behavior_graphs.stages.labeling.vlm_labeler.VLMLabeler` resolves
    that from ``configs/data/`` for every entry point, and an explicit list also
    *restricts* — an episode that camera missed would contribute no row at all,
    where the default keeps the reader's fallback to a view that has it.
    """
    if not requested:
        return None

    available = _camera_names(rollout_dir)
    unknown = [c for c in requested if c not in available]
    if unknown:
        print(
            f"ERROR: no such camera view(s) in {rollout_dir}: {', '.join(unknown)}. "
            f"Available: {', '.join(available) or '(none)'}",
            file=sys.stderr,
        )
        sys.exit(1)
    return requested


def main():
    parser = argparse.ArgumentParser(description="Run VLM node labeler on an existing run dir.")
    parser.add_argument("run_dir", type=Path, help="Path to the run artifact directory.")
    parser.add_argument(
        "--rollout-dir",
        type=Path,
        default=None,
        help="Path to rollout videos. Defaults to rollout_dir in metrics.json.",
    )
    parser.add_argument(
        "--data-path",
        default=os.environ.get("DATA_PATH"),
        help=(
            "Repo-wide data root that the rollout_dir recorded in metrics.json "
            "is relative to (default: $DATA_PATH)."
        ),
    )
    parser.add_argument(
        "--task",
        default="",
        help=(
            "Free-text task description for the VLM (strongly recommended). "
            "If omitted, the VLM will attempt to infer the task from successful episode "
            "contact sheets, which may be unreliable for complex multi-step tasks."
        ),
    )
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--frames-per-episode", type=int, default=5)
    parser.add_argument(
        "--tile-px",
        type=int,
        default=224,
        help="Shorter side of each contact-sheet tile (raise for >224px footage)",
    )
    parser.add_argument("--max-episodes", type=int, default=4)
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        metavar="NAME",
        help=(
            "Camera view to build the contact sheets from; repeat for several, in the "
            "order they should appear. Each episode then contributes one labelled row "
            "per view, sampled at the same timesteps, and the VLM is told they are "
            "simultaneous. Default: one view per episode. Pass --list-cameras to see "
            "what the rollout dir has."
        ),
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Print the rollout dir's camera views and exit.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="After labeling, run graph analysis (task summary, success/failure modes, "
        "similar cluster pairs). Writes graph_analysis.json to run_dir.",
    )
    # Vertex AI / GCP args (alternative to setting env vars externally)
    parser.add_argument(
        "--gcp-project",
        default="gcp-driven-data",
        help="GCP project ID for Vertex AI (default: gcp-driven-data).",
    )
    parser.add_argument(
        "--gcp-location",
        default="global",
        help="GCP region/location for Vertex AI (default: global).",
    )
    args = parser.parse_args()

    # Apply Vertex AI config before any google-genai imports
    os.environ["GOOGLE_CLOUD_PROJECT"] = args.gcp_project
    os.environ["GOOGLE_CLOUD_LOCATION"] = args.gcp_location
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    from behavior_graphs.pipeline.artifacts import RunArtifacts, resolve_rollout_dir
    from behavior_graphs.stages.labeling.vlm_labeler import VLMLabeler

    artifacts = RunArtifacts(run_dir)
    result = artifacts.load_result()

    # Resolve rollout_dir: CLI override (handles Docker-path translation),
    # else the path recorded in metrics.json joined onto --data-path.
    if args.rollout_dir is not None:
        rollout_dir = args.rollout_dir.resolve()
        if not rollout_dir.is_dir():
            print(f"ERROR: rollout_dir not found: {rollout_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        raw = result.metrics.get("rollout_dir")
        rollout_dir = resolve_rollout_dir(raw, args.data_path)
        if rollout_dir is None or not rollout_dir.is_dir():
            print(
                f"ERROR: rollout_dir {raw!r} from metrics.json resolves to "
                f"{rollout_dir} which doesn't exist. Recorded paths are relative "
                "to the data root — pass --data-path (or set $DATA_PATH), or pass "
                "--rollout-dir with the host path.",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.list_cameras:
        print(f"Camera views in {rollout_dir}:")
        for name in _camera_names(rollout_dir) or ["(none — no media/ clips)"]:
            print(f"  {name}")
        sys.exit(0)

    cameras = _resolve_cameras(rollout_dir, args.cameras)

    print(f"Run dir:     {run_dir}")
    print(f"Rollout dir: {rollout_dir}")
    print(f"Model:       {args.model}")
    print(f"Task hint:   {args.task or '(will infer from successful episodes)'}")
    print(f"Cameras:     {', '.join(cameras) if cameras else '(the default view)'}")
    print()

    labeler = VLMLabeler(
        model=args.model,
        frames_per_episode=args.frames_per_episode,
        tile_px=args.tile_px,
        max_episodes_per_node=args.max_episodes,
        task_description=args.task,
        data_path=args.data_path,
        cameras=cameras,
    )

    print("Labeling nodes...")
    summary = labeler.label(result, run_dir=run_dir, rollout_dir=rollout_dir)
    if "error" in summary:
        print(f"ERROR: {summary['error']}", file=sys.stderr)
        sys.exit(1)

    from behavior_graphs.core.node_labels import load_node_labels

    if summary.get("inferred_task"):
        print(f"Inferred task: {summary['inferred_task']}\n")
    print(f"\nLabeled {summary['nodes_labeled']} nodes with {summary['model']}:")
    for nid, entry in sorted(load_node_labels(run_dir).items()):
        conf = entry.get("confidence", "?")
        conf_str = f"{conf:.0%}" if isinstance(conf, float) else str(conf)
        print(f"  Node {nid:>3}: [{conf_str}] {entry['label']}")
        print(f"             {entry.get('description', '')}")

    print(f"\nLabels written to: {run_dir}/node_labels.json")

    if args.analyze:
        print("\nAnalyzing graph...")
        analysis = labeler.analyze(result, run_dir=run_dir, rollout_dir=rollout_dir)
        if "error" in analysis:
            print(f"ERROR: {analysis['error']}", file=sys.stderr)
            sys.exit(1)

        print(f"\nTask summary:\n  {analysis.get('task_summary', '')}")

        print("\nSuccess modes:")
        for m in analysis.get("success_modes", []):
            print(f"  {m.get('path_str', '')} ({m.get('probability', 0):.0%})")
            print(f"    {m.get('description', '')}")

        print("\nFailure modes:")
        for fi, m in enumerate(analysis.get("failure_modes", [])):
            prob = m.get("probability")
            prob_str = f" ({prob:.0%})" if prob is not None else ""
            dominant = " [dominant]" if fi == 0 else ""
            print(
                f"  Nodes {m.get('key_node_ids', [])}{prob_str}{dominant}: {m.get('description', '')}"
            )

        pairs = analysis.get("similar_pairs", [])
        if pairs:
            print("\nSimilar cluster pairs:")
            for p in pairs:
                print(f"  B{p['node_a']} ≈ B{p['node_b']}: {p.get('distinction', '')}")

        print(f"\nOverall:\n  {analysis.get('overall_assessment', '')}")
        print(f"\nAnalysis written to: {run_dir}/graph_analysis.json")


if __name__ == "__main__":
    main()
