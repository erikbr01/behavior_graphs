"""Proprioception for a DROID rollout dir, as ``observations/state.npz``.

The canonical layout keeps raw per-timestep observations in
``observations/<key>.npz`` (see
:class:`~behavior_graphs.envs.trajectory_dir.TrajectoryDirectory`), which is what
:class:`~behavior_graphs.stages.features.state.StateFeatures` reads. A collection
assembled by ``assemble_droid_sync_recordings.py`` does not get one -- that
script carries the embeddings and the videos across and leaves the telemetry in
the raw ``recordings/<capture>/trajectory.h5`` captures -- so the ``state`` arm
of a sweep has nothing to load. This fills that in.

One row per policy step, so no resampling: the DROID capture already records one
row per step and the episode lengths in ``episodes/metadata.yaml`` match it
exactly, which this checks rather than assumes.

**Proprioception as recorded, gripper included** -- 14 dimensions, the same
definition ``build_dataset.py`` writes for the ABC tasks (6 arm joints plus one
gripper channel per arm there, 7 joints plus 6 cartesian plus the gripper here).
Keeping the definition means the ``state`` row can be read across the two
studies, which a trimmed variant would forfeit.

Worth knowing rather than worked around: on a dataset whose ground truth came
from ``scripts/ablation_study/build_gripper_gt.py``, the gripper column is the signal
those labels were derived from, so the ``state`` arm has direct access to it and
its score there is an upper bound rather than a measurement of the
representation. ``--no-gripper`` writes the 13-d variant to
``observations/state_nogrip.npz`` for anyone who wants to quantify that gap; the
default does not hide the channel.

**Rewriting this file does not invalidate a sweep on its own.**
:func:`behavior_graphs.pipeline.cache.data_fingerprint` covers
``episodes/metadata.yaml``, and ``embedding_fingerprint`` covers
``policy_embeddings/<layer>.npz`` -- neither covers ``observations/``. So a
re-run reuses the previous array's cached features silently, which is the
failure ``embedding_fingerprint``'s own docstring was written about. This script
therefore bumps the mtime of ``episodes/metadata.yaml`` after writing, which is
what the fingerprint reads; pass ``--no-touch`` to skip that if you know the
sweep has not run yet.

Usage::

    python scripts/ablation_study/build_droid_state.py \\
        ${DATA_PATH}/rollouts/cooking_baseline_dit_200

    # the gripper-free variant, as observations/state_nogrip.npz
    python scripts/ablation_study/build_droid_state.py <dir> --no-gripper
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from behavior_graphs.annotations import gripper as G  # noqa: E402

#: h5 dataset -> width, in the order they are concatenated. Measured state, not
#: the commanded action: the command leads the measurement by however long the
#: arm takes to get there, and a representation of what was *asked for* is a
#: representation of the policy's output rather than of the robot.
CHANNELS: tuple[tuple[str, int], ...] = (
    ("observation/robot_state/joint_positions", 7),
    ("observation/robot_state/cartesian_position", 6),
)

#: Last, and included by default. Last so that ``--no-gripper`` is exactly this
#: array without its final column, which is what makes the two variants
#: comparable rather than merely similar.
GRIPPER_CHANNEL = ("observation/robot_state/gripper_position", 1)


def episode_lengths(rollout_dir: pathlib.Path) -> list[int]:
    meta = yaml.safe_load((rollout_dir / "episodes" / "metadata.yaml").read_text())
    return [int(ep["length"]) for ep in meta["episodes"]]


def capture_dirs(rollout_dir: pathlib.Path) -> list[pathlib.Path]:
    recordings = rollout_dir / G.RECORDINGS_DIR
    if not recordings.is_dir():
        raise SystemExit(f"no {recordings}; nothing to read telemetry from")
    return sorted(p for p in recordings.iterdir() if p.is_dir())


def read_episode(capture: pathlib.Path, channels, frames: int) -> np.ndarray:
    import h5py

    columns: list[np.ndarray] = []
    with h5py.File(capture / "trajectory.h5", "r") as handle:
        for key, width in channels:
            if key not in handle:
                raise SystemExit(f"{capture / 'trajectory.h5'} has no {key}")
            values = np.asarray(handle[key][:], dtype=np.float32)
            if values.ndim == 1:
                values = values[:, None]
            if values.shape[1] != width:
                raise SystemExit(
                    f"{key} in {capture.name} is {values.shape[1]} wide, expected {width}"
                )
            if values.shape[0] != frames:
                raise SystemExit(
                    f"{key} in {capture.name} has {values.shape[0]} rows but the episode "
                    f"is {frames} frames -- the captures and the metadata disagree, so "
                    "the sorted-position mapping is wrong and nothing here would align"
                )
            columns.append(values)
    return np.concatenate(columns, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("rollout_dir", type=pathlib.Path)
    ap.add_argument(
        "--no-gripper",
        action="store_true",
        help="drop the gripper column and write state_nogrip.npz instead, for "
        "quantifying how much of a score on gripper-derived labels is circular",
    )
    ap.add_argument(
        "--no-touch",
        action="store_true",
        help="skip bumping episodes/metadata.yaml's mtime; only safe when no "
        "sweep has cached features from an earlier version of this array",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rollout_dir = args.rollout_dir
    if not (rollout_dir / "episodes" / "metadata.yaml").is_file():
        raise SystemExit(f"{rollout_dir} is not a canonical rollout dir")

    channels = CHANNELS if args.no_gripper else CHANNELS + (GRIPPER_CHANNEL,)
    width = sum(w for _, w in channels)
    name = "state_nogrip" if args.no_gripper else "state"

    lengths = episode_lengths(rollout_dir)
    captures = capture_dirs(rollout_dir)
    if len(captures) != len(lengths):
        raise SystemExit(
            f"{len(captures)} captures but {len(lengths)} episodes in metadata; the "
            "episode index is assigned by sorted capture position, so a mismatch "
            "means every array after the gap would be misaligned"
        )

    blocks = [read_episode(c, channels, n) for c, n in zip(captures, lengths, strict=True)]
    state = np.concatenate(blocks, axis=0)
    assert state.shape == (sum(lengths), width), state.shape

    print(f"{rollout_dir.name}: {len(blocks)} episodes, {state.shape[0]} frames, {width}-d")
    print("  channels: " + ", ".join(f"{k.rsplit('/', 1)[-1]}({w})" for k, w in channels))
    print(f"  range per column: min {state.min(axis=0).round(3)}")
    print(f"                    max {state.max(axis=0).round(3)}")
    if args.no_gripper:
        print("  gripper column dropped, so this is not the ABC `state` definition")

    if args.dry_run:
        print("\nDRY RUN -- nothing written.")
        return

    out = rollout_dir / "observations" / f"{name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Key "observations", matching what TrajectoryDirectory.observations reads
    # and what build_dataset.py writes for the ABC datasets.
    np.savez_compressed(out, observations=state)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    if not args.no_touch:
        # The stage cache keys features on `data_fingerprint`, which reads the
        # size and mtime of episodes/metadata.yaml and nothing under
        # observations/. Without this bump a re-run silently reuses the features
        # built from the *previous* array -- see the module docstring.
        meta = rollout_dir / "episodes" / "metadata.yaml"
        meta.touch()
        print(
            f"bumped {meta.name} mtime so the stage cache treats this as new data "
            "(every arm's features recompute, not just state)"
        )


if __name__ == "__main__":
    main()
