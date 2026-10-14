#!/usr/bin/env python
"""One entrypoint for the ABC subtask-boundary ablation study.

    study.py splits   --dataset legos          # draw and write the partition
    study.py media    --dataset bottles        # link video for vision features
    study.py policyids --dataset legos         # rgrl episode_ids for train only
    study.py policyconfig --dataset legos      # write rgrl's data config
    study.py sweep    --dataset legos --layers dinov3
    study.py rescore  --dataset bottles        # re-score cached labels
    study.py report   --dataset bottles --which dials
    study.py report   --dataset bottles --all --out experiments/bottles

A dataset is one file in ``configs/ablation_study/``. Nothing here takes a path, so
same command works for every dataset and there is no per-dataset script to keep
in sync. Reports live in ``behavior_graphs.ablation_study.report`` and are importable
and testable on their own.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from behavior_graphs.ablation_study import (  # noqa: E402
    cells,
    config,
    core,
    media,
    policy_config,
    rescore,
    splits,
    sweep,
)
from behavior_graphs.ablation_study.report import REPORTS, aggregate  # noqa: E402


def cmd_splits(cfg, args) -> str:
    if cfg.split.file:
        named = splits.resolve(cfg)
        return (
            f"{cfg.name}: split read from {cfg.splits_path} "
            f"(legacy keys {cfg.split.keys}); nothing to draw.\n" + splits.report(cfg, named)
        )
    payload = splits.generate(cfg)
    if args.dry_run:
        named = {k: payload[k] for k in (splits.TRAIN, splits.VAL, splits.TEST)}
        return f"{cfg.name}: dry run, would write {cfg.splits_path}\n" + splits.report(cfg, named)
    dest = splits.write(cfg, payload)
    named = splits.resolve(cfg)
    return (
        f"{cfg.name}: wrote {dest}\n"
        f"  {payload['n_annotated']} annotated episodes, seed {payload['seed']}, "
        f"sha256 {payload['sha256'][:16]}\n" + splits.report(cfg, named)
    )


def cmd_policyids(cfg, args) -> str:
    """Emit the policy's episode_ids from the same draw as the study split."""
    orders = sorted(
        (cfg.policy_zarr or pathlib.Path("/nonexistent")).glob("*/abc_episode_order.json")
    )
    if not orders:
        raise SystemExit(
            f"{cfg.name}: no abc_episode_order.json under {cfg.policy_zarr}; "
            "build the zarr cache first"
        )
    payload = splits.rgrl_train_ids(cfg, orders[-1])
    dest = cfg.root / "rgrl_train_ids.json"
    if not args.dry_run:
        dest.write_text(json.dumps(payload, indent=2))
    ids = payload["episode_ids"]
    return (
        f"{cfg.name}: {payload['count']} train episodes of "
        f"{payload['n_train_episodes']} in the split "
        f"({payload['n_missing_from_cache']} not in the cache)\n"
        f"  held out and excluded: {payload['n_held_out_excluded']}\n"
        f"  episode_ids sha256 {payload['expected_sha256'][:16]}, "
        f"split sha256 {payload['split_sha256'][:16]}\n"
        f"  first ids {ids[:8]}{'...' if len(ids) > 8 else ''}\n"
        + (f"  wrote {dest}" if not args.dry_run else "  dry run, nothing written")
    )


def cmd_report(cfg, args) -> str:
    which = list(REPORTS) if args.all else [args.which]
    unknown = [w for w in which if w not in REPORTS]
    if unknown:
        raise SystemExit(f"unknown report(s) {unknown}; available: {', '.join(REPORTS)}")
    # Pre-flight, so a missing sweep aborts once with a useful message instead
    # of once per report. Either artefact will do: most reports read the wide
    # `rescored.json`, while the ones built on the balanced core design read
    # `rescored_core.json`, which is 180 records and refreshes in minutes.
    artefacts = [cfg.rescored, cfg.sweep_dir / "rescored_core.json"]
    if not any(p.exists() for p in artefacts):
        raise SystemExit(
            f"{cfg.name}: none of {', '.join(p.name for p in artefacts)} under "
            f"{cfg.sweep_dir}; run `study.py rescore --dataset {cfg.name}` first"
        )
    chunks = []
    failed = []
    for name in which:
        kwargs = {"best_achievable": args.best_achievable} if name == "ablation" else {}
        try:
            body = REPORTS[name](cfg, **kwargs)
        # SystemExit too, not just Exception: the reports raise it for their own
        # preconditions -- an arm with no cells under the chosen criterion, for
        # instance -- and it does not derive from Exception, so catching only
        # that let one arm's precondition kill the whole batch.
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            # Keep going, but only under --all, and record the traceback rather
            # than a one-line summary. A single report that cannot run -- a
            # missing artefact, an arm with no cells -- used to abort the whole
            # batch, which under the unattended report chain meant losing every
            # other report as well. Asking for one report by name still raises,
            # since there is nothing left to salvage.
            if not args.all:
                raise
            body = f"FAILED: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
            failed.append(name)
        chunks.append(f"{'=' * 88}\n{name}\n{'=' * 88}\n{body}")
        if args.out:
            out = pathlib.Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{cfg.name}_{name}.txt").write_text(body + "\n")
    if failed:
        chunks.append(f"{len(failed)} of {len(which)} reports failed: {', '.join(failed)}")
    return "\n\n".join(chunks)


def cmd_summary(cfg, args) -> str:
    """One-line health check: does the dataset load and what is on disk."""
    lines = [f"{cfg.name}  task={cfg.task}", f"  root       {cfg.root}"]
    lines.append(
        f"  annotated  {cfg.annotations} {'ok' if cfg.annotations.exists() else 'MISSING'}"
    )
    lines.append(
        f"  splits     {cfg.splits_path} {'ok' if cfg.splits_path.exists() else 'MISSING'}"
    )
    lines.append(f"  rescored   {cfg.rescored} {'ok' if cfg.rescored.exists() else 'MISSING'}")
    if cfg.splits_path.exists() and cfg.annotations.exists():
        named = splits.resolve(cfg)
        lines.append("  " + splits.report(cfg, named).replace("\n", "\n  "))
    if cfg.rescored.exists():
        rows = cells.load(cfg)
        top = max(rows, key=lambda c: c.score)
        lines.append(
            f"  sweep      {len(rows)} cells, best {top.score:.4f} "
            f"({top.layer} K={top.k} w={top.width} s={top.stride})"
        )
    return "\n".join(lines)


def cmd_policyconfig(cfg, args) -> str:
    """Write the rgrl data config with episode_ids derived from the split."""
    dest, payload = policy_config.write(cfg, dry_run=args.dry_run)
    return (
        f"{cfg.name}: {payload['count']} train episodes, "
        f"{payload['n_held_out_excluded']} held out and excluded, "
        f"{payload['n_missing_from_cache']} train episodes absent from the store\n"
        f"  split sha {payload['split_sha256'][:16]}  "
        f"ids sha {payload['expected_sha256'][:16]}\n"
        + (f"  wrote {dest}" if not args.dry_run else f"  dry run, would write {dest}")
    )


def cmd_sweep(cfg, args) -> str:
    """Run the ablation grid. Shells out to the cuML interpreter explicitly."""
    layers = tuple(args.layers.split(",")) if args.layers else None
    if args.dry_run:
        return sweep.plan(cfg, layers, grid=args.grid)
    return sweep.run(cfg, layers, workers=args.workers, grid=args.grid)


def cmd_rescore(cfg, args) -> str:
    """Re-score every completed job from its cached labels.

    Needed whenever the metric changes, and whenever an arm's features are
    replaced: a job's hash covers its overrides, not the bytes of the feature
    file, so ``rescored.json`` keeps reporting the old numbers until this runs.

    ``--core`` restricts it to the balanced design and defaults the output to
    `rescored_core.json`, which is what the core-based reports read. Worth
    having as its own flag rather than a note in the docs: the whole bottles
    grid is thousands of jobs at ~0.9s each, the core is 180, and after a
    feature replacement the core is usually all that needs refreshing.
    """
    out = pathlib.Path(args.out) if args.out else None
    only = None
    if args.core:
        only = core.is_core_dials
        out = out or cfg.sweep_dir / "rescored_core.json"
    if args.dry_run:
        n = len(rescore.job_dirs(cfg))
        scope = "core-design jobs among" if args.core else "of"
        return f"{cfg.name}: dry run, would rescore {scope} {n} jobs into {out or cfg.rescored}"
    return f"rescored {rescore.run(cfg, out, only=only)}"


def cmd_media(cfg, args) -> str:
    """Symlink raw episode video into the canonical layout for vision features."""
    cams = tuple(args.cameras.split(",")) if args.cameras else ("top",)
    stats = media.link(cfg, cams, dry_run=args.dry_run)
    body = media.report(cfg, stats, cams)
    return body + ("\n  dry run, nothing written" if args.dry_run else "")


def cmd_cache_rows(cfg, args) -> str:
    """Compute this dataset's held-out ablation rows and cache them beside the sweep.

    Its own command because it is the slow half of the `latex` report -- an
    argmax per level on the selection fold, then a pipeline scoring run per
    level on the reserved episodes -- and the pooled table
    (`report --which aggregate`, and the Figures page) needs it for every
    dataset. Reading eight of these live is instant; computing them is minutes
    each, so the computing happens here, once, on purpose.
    """
    if args.dry_run:
        return f"{cfg.name}: dry run, would write {aggregate.cache_path(cfg)}"
    path = aggregate.write_cache(cfg)
    payload = aggregate.read_cache(cfg) or {}
    n = len(payload.get("entries") or [])
    return f"{cfg.name}: cached {n} ablation rows -> {path}"


COMMANDS = {
    "cache_rows": cmd_cache_rows,
    "splits": cmd_splits,
    "policyids": cmd_policyids,
    "media": cmd_media,
    "policyconfig": cmd_policyconfig,
    "sweep": cmd_sweep,
    "rescore": cmd_rescore,
    "report": cmd_report,
    "summary": cmd_summary,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--dataset", required=True, help=f"one of {config.available()}")
    ap.add_argument("--which", default="dials", help=f"report: {', '.join(REPORTS)}")
    ap.add_argument("--all", action="store_true", help="every report")
    ap.add_argument(
        "--out",
        default=None,
        help="report: a directory to also write each report into; rescore: the json file to write",
    )
    ap.add_argument(
        "--best-achievable",
        action="store_true",
        help="ablation: add the conflated best-per-level view",
    )
    ap.add_argument("--cameras", default=None, help="media: comma-separated views, default top")
    ap.add_argument("--layers", default=None, help="sweep: comma-separated representations to run")
    ap.add_argument(
        "--grid",
        default=None,
        help=(
            "sweep: run this grid for every representation, overriding the "
            "per-representation mapping. Use `ablation_core` to fill the complete "
            "balanced design, which is a property of the design rather than of "
            "one arm, so `state` has to run it on the same dials as the rest"
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "sweep: representations to run concurrently. Above 1 the passes are "
            "split per layer so no two runners contend for a job. Bounded by RAM, "
            "not cores: each worker holds its own features and windows"
        ),
    )
    ap.add_argument(
        "--core",
        action="store_true",
        help="rescore: only jobs inside the balanced core design, into "
        "rescored_core.json unless --out says otherwise",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = config.load(args.dataset)
    print(COMMANDS[args.command](cfg, args))


if __name__ == "__main__":
    main()
