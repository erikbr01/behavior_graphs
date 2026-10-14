"""Skeleton subtask spans proposed from the gripper's own telemetry.

Annotating a long manipulation episode by hand is mostly *placing* boundaries,
and for a parallel-jaw gripper most of them are already recorded: a subtask
usually ends where the hand closes on something or lets it go. So a first pass
can be machine-made — cut the episode at every grasp and every release, leave
the pills contiguous and unlabelled, and let the annotator drag the edges that
are wrong rather than find every edge from scratch.

This is deliberately *not* a segmentation method. It is the same open/closed
flip that :func:`behavior_graphs.wgo_bench.baselines.gripper_events` uses as the
baseline a real method has to beat, reused here as a labour-saving device. It
knows nothing about what a subtask is: an episode where two subtasks hand over
without the gripper moving gets one pill across both, and a regrasp inside one
subtask gets two. Both are the annotator's job to fix, which is why the spans
carry no labels — a skeleton is boundary placement, not an opinion.

The signal is *closedness* in ``[0, 1]``: 0 is fully open. That is what DROID
records in ``observation/robot_state/gripper_position``, and the only source
read here — the assembled rollout layout keeps the raw per-episode captures in
``recordings/``, so nothing has to be re-derived. A dataset without them has no
skeleton available and says so (:func:`trace` returns None), which is the
absence the page checks for.

Frames, not seconds, throughout: the annotation is indexed by policy step (see
:mod:`behavior_graphs.annotations.subtasks`), and the h5 telemetry is one row per
step, so the two line up 1:1. :func:`trace` verifies that against the episode's
recorded length rather than assuming it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

_log = logging.getLogger(__name__)

RECORDINGS_DIR = "recordings"

#: The h5 dataset holding gripper closedness in ``[0, 1]``, best first. DROID
#: records both the measured state and the commanded action; the measurement is
#: what the hand actually did, and the command leads it by the time it takes to
#: close — which would put every boundary a few frames early.
_H5_GRIPPER_KEYS = (
    "observation/robot_state/gripper_position",
    "action/gripper_position",
)

#: Closedness above this counts as closed. Absolute rather than a per-episode
#: rescaling: the channel is already normalised, and an episode where the
#: gripper never grasps anything has a range of pure noise to stretch.
CLOSED_ABOVE = 0.5

#: How long a new gripper state must persist to count as an event, in seconds.
#: The gripper chatters as it settles on an object, and each rattle would
#: otherwise become two boundaries a few frames apart. Measured on
#: ``cooking_baseline_dit_200``: 0.7% of flips fall within 0.5 s of the previous
#: one, while the median gap between real events is 4 s — so this discards the
#: rattle and nothing else.
MIN_DWELL_SEC = 0.5


def trace(rollout_dir: Path | str, episode: int, *, num_frames: int = 0) -> np.ndarray | None:
    """Per-frame gripper closedness for one episode, or None when unavailable.

    None rather than an exception for every ordinary absence — a dataset with no
    ``recordings/`` (every simulator benchmark), a build without ``h5py``, a
    capture that does not record the gripper — because the caller's response to
    all of them is the same: offer no skeleton, and say why.

    *num_frames*, when given, is required to match the length of the telemetry.
    Episode index maps to a capture directory by sorted position, which is how
    ``assemble_droid_sync_recordings.py`` assigned the indices in the first
    place; if that assembly skipped an incomplete episode the positions shift,
    and the length check is what catches it. Returning a misaligned trace would
    put every boundary in the wrong place on every episode after the gap, which
    is worse than offering nothing.
    """
    directory = _capture_dir(Path(rollout_dir), episode)
    if directory is None:
        return None
    try:
        import h5py  # lazy: only this feature needs it, like the env adapters
    except ImportError:
        _log.info("no h5py in this environment, so no gripper telemetry to read")
        return None

    path = directory / "trajectory.h5"
    try:
        with h5py.File(path, "r") as handle:
            for key in _H5_GRIPPER_KEYS:
                if key in handle:
                    values = np.asarray(handle[key][:], dtype=np.float64).reshape(-1)
                    break
            else:
                _log.info("%s records no gripper channel (tried %s)", path, _H5_GRIPPER_KEYS)
                return None
    except (OSError, KeyError) as exc:
        _log.warning("cannot read gripper telemetry from %s (%s)", path, exc)
        return None

    if num_frames and values.size != num_frames:
        _log.warning(
            "gripper telemetry in %s is %d frames but episode %d is %d — refusing to "
            "propose a skeleton from a trace that is not this episode's",
            path,
            values.size,
            episode,
            num_frames,
        )
        return None
    return values


def _capture_dir(rollout_dir: Path, episode: int) -> Path | None:
    """The raw capture directory for *episode*, by sorted position."""
    recordings = rollout_dir / RECORDINGS_DIR
    try:
        captures = sorted(p for p in recordings.iterdir() if p.is_dir())
    except OSError:
        return None
    if not 0 <= episode < len(captures):
        return None
    return captures[episode]


def events(
    closedness: np.ndarray,
    *,
    closed_above: float = CLOSED_ABOVE,
    min_dwell: int = 1,
) -> list[int]:
    """Frames where the gripper's open/closed state changes and stays changed.

    A dwell filter rather than a refractory period: a flip is accepted only once
    the new state has held for *min_dwell* frames, so a rattle is rejected
    outright and leaves the state where it was. A refractory period would accept
    the first edge of a rattle and suppress the second, leaving a boundary in the
    middle of a settling grasp and none at the grasp itself.

    The returned frame is the first one in the new state, matching the half-open
    span convention: it is where the next subtask begins.
    """
    closed = np.asarray(closedness, dtype=np.float64).reshape(-1) >= closed_above
    n = closed.size
    if n < 2:
        return []
    dwell = max(1, int(min_dwell))
    out: list[int] = []
    state = bool(closed[0])
    i = 1
    while i < n:
        if bool(closed[i]) == state:
            i += 1
            continue
        run_end = i + 1
        while run_end < n and bool(closed[run_end]) == bool(closed[i]):
            run_end += 1
        # A run that reaches the end of the episode counts however short it is:
        # a release on the final frames is a real event that simply ran out of
        # episode to persist into.
        if run_end - i >= dwell or run_end == n:
            out.append(i)
            state = bool(closed[i])
        i = run_end
    return out


def spans(num_frames: int, boundaries: list[int], *, min_span: int = 1) -> list[tuple[int, int]]:
    """*boundaries* as contiguous half-open spans tiling ``[0, num_frames)``.

    Contiguous by construction, because that is the shape a first pass wants:
    gaps are a deliberate statement that a stretch of the episode is no
    subtask, and the gripper has no view on that. The annotator opens gaps where
    they belong with ``I``/``O``.

    A boundary that would leave a span shorter than *min_span* is dropped rather
    than rounded, scanning forward, so both a flip a few frames into the episode
    and a pair of flips a few frames apart collapse into their neighbour instead
    of leaving a pill too small to grab.
    """
    if num_frames < max(1, int(min_span)):
        return []
    floor = max(1, int(min_span))
    edges = [0]
    for frame in sorted({int(b) for b in boundaries}):
        if floor <= frame <= num_frames - floor and frame - edges[-1] >= floor:
            edges.append(frame)
    edges.append(int(num_frames))
    # No length filter on the result: the two conditions above already keep
    # every edge at least *floor* from its predecessor and from the episode's
    # end, and dropping a span here would leave the tiling with a hole in it.
    return list(zip(edges[:-1], edges[1:], strict=True))


def skeleton(
    closedness: np.ndarray,
    num_frames: int,
    *,
    fps: float = 10.0,
    closed_above: float = CLOSED_ABOVE,
    min_dwell_sec: float = MIN_DWELL_SEC,
) -> list[tuple[int, int]]:
    """The whole proposal: contiguous spans cut at every debounced gripper event.

    *min_dwell_sec* is in seconds and converted here, so the same setting means
    the same thing on a 5 Hz VLA and a 60 Hz teleop replay.
    """
    dwell = max(1, int(round(max(0.0, min_dwell_sec) * max(fps, 1e-6))))
    found = events(closedness, closed_above=closed_above, min_dwell=dwell)
    return spans(num_frames, found, min_span=dwell)
