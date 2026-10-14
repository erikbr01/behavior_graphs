"""Link the raw episode video into the canonical layout, so vision features work.

``build_dataset.py`` writes state, actions and annotations but no video, because
nothing in the state-or-policy-feature path needs it. A frozen vision backbone
does, and :class:`RolloutDirectory` looks for
``media/<camera>/ep{idx:04d}_{success|failure}.mp4``.

The mp4s already exist under the raw download, remuxed out of each
``episode.mcap`` into ``.robo_dataset/``, so this links rather than copies:
bottles' three views are 127 GB and a second copy is not worth having.

The remux is ``ffmpeg -c copy`` and every sidecar was built at ``stride: 1``, so
the mp4 is **not** a decimated derivative -- its coded pictures are the mcap's,
byte for byte, at the full camera rate. That is why the local ``episode.mcap``
files could be deleted (2026-09-10, 557 GB) while this path kept working. What
the mcaps still held and the sidecars do not: the ``left``/``right`` streams for
all but 478 episodes, and state/action at their native ~276 Hz rather than
resampled onto the frame axis. Restore from
``gs://behavior-graphs-data/ablation_study/abc_130k_mcap`` (ARCHIVE) or the
ABC-130k release; ``abc_130k/deleted_mcap_manifest.tsv`` lists exactly what went.
"""

from __future__ import annotations

import pathlib

import yaml

from behavior_graphs.ablation_study.config import DatasetConfig

#: The views ABC records. `top` is the third-person view and the default: a
#: wrist camera mostly sees gripper-local texture, which is the opposite of the
#: scene layout a patch-grid embedding is meant to capture.
CAMERAS = ("top", "left", "right")


def episode_dirs(cfg: DatasetConfig) -> dict[int, str]:
    """Study episode index -> raw episode directory name."""
    meta = yaml.safe_load((cfg.root / "episodes" / "metadata.yaml").read_text())
    eps = meta["episodes"] if isinstance(meta, dict) and "episodes" in meta else meta
    return {i: e.get("wgo_id") for i, e in enumerate(eps)}


def link(
    cfg: DatasetConfig, cameras: tuple[str, ...] = ("top",), dry_run: bool = False
) -> dict[str, int]:
    """Symlink each episode's video into ``<root>/media/<camera>/``.

    Returns per-camera counts of what was linked and what had no source, so a
    partial download is visible rather than silently producing a short feature
    array later.
    """
    if cfg.raw is None:
        raise SystemExit(f"{cfg.name}: no `raw` path in the config, nothing to link")
    dirs = episode_dirs(cfg)
    stats: dict[str, int] = {}
    for camera in cameras:
        out = cfg.root / "media" / camera
        if not dry_run:
            out.mkdir(parents=True, exist_ok=True)
        linked = missing = 0
        for idx, name in sorted(dirs.items()):
            src = cfg.raw / name / ".robo_dataset" / f"{camera}.mp4"
            if not src.exists():
                missing += 1
                continue
            # Every ABC episode is a demonstration, so the tag is always success.
            dest = out / f"ep{idx:04d}_success.mp4"
            if not dry_run:
                if dest.is_symlink() or dest.exists():
                    dest.unlink()
                dest.symlink_to(src)
            linked += 1
        stats[camera] = linked
        stats[f"{camera}_missing"] = missing
    return stats


def report(cfg: DatasetConfig, stats: dict[str, int], cameras) -> str:
    lines = [f"{cfg.name}: media links under {cfg.root / 'media'}"]
    for camera in cameras:
        lines.append(
            f"  {camera:>6}  linked {stats.get(camera, 0):>5}  "
            f"no source {stats.get(f'{camera}_missing', 0):>5}"
        )
    miss = sum(stats.get(f"{c}_missing", 0) for c in cameras)
    if miss:
        lines.append(
            "  Episodes without a source mp4 will have no vision features. That is "
            "either an\n  incomplete download or an episode whose extraction never ran."
        )
    return "\n".join(lines)


def resolve_camera(cfg: DatasetConfig, camera: str | None) -> str:
    """Pick a camera that actually has links, so failures are early and clear."""
    media = cfg.root / "media"
    have = (
        [c.name for c in sorted(media.glob("*")) if any(c.glob("*.mp4"))] if media.exists() else []
    )
    if camera:
        if camera not in have:
            raise SystemExit(
                f"{cfg.name}: no linked video for camera {camera!r} "
                f"(have {have or 'none'}); run `study media --dataset {cfg.name}`"
            )
        return camera
    if not have:
        raise SystemExit(
            f"{cfg.name}: no linked video at all; run `study media --dataset {cfg.name}`"
        )
    for preferred in CAMERAS:
        if preferred in have:
            return preferred
    return have[0]


def resolve_cameras(cfg: DatasetConfig, spec: str | None) -> list[str]:
    """Resolve a comma-separated camera list, in canonical order.

    The vision-backbone baseline embeds each view and concatenates, so it should
    see the same three cameras the policy does -- giving it one view when the
    policy taps get three makes the comparison a test of information available,
    not of representation. `None` therefore means *all linked cameras*, not the
    first one.

    Order is canonicalised to `CAMERAS` rather than taken from the argument, so
    `top,left` and `left,top` cannot produce two different feature layouts under
    the same file name.
    """
    media = cfg.root / "media"
    have = (
        [c.name for c in sorted(media.glob("*")) if any(c.glob("*.mp4"))] if media.exists() else []
    )
    if not have:
        raise SystemExit(
            f"{cfg.name}: no linked video at all; run "
            f"`study media --dataset {cfg.name} --cameras top,left,right`"
        )
    if spec:
        want = [c.strip() for c in spec.split(",") if c.strip()]
        missing = [c for c in want if c not in have]
        if missing:
            raise SystemExit(
                f"{cfg.name}: no linked video for {', '.join(missing)} "
                f"(have {', '.join(have)}); run "
                f"`study media --dataset {cfg.name} --cameras {','.join(want)}`"
            )
    else:
        want = have
    ordered = [c for c in CAMERAS if c in want] + [c for c in want if c not in CAMERAS]
    return ordered


def _unused() -> pathlib.Path:  # keeps pathlib imported for type annotations
    return pathlib.Path(".")
