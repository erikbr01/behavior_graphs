"""Ground-truth subtask segments for a dataset, for display under its videos.

The benchmark datasets carry a human segmentation at
``<rollout_dir>/annotations/segments.json`` -- the same file
:mod:`behavior_graphs.stages.evaluation._boundary_metrics` scores against. This
reads it for *display*, which is a different job from scoring and so keeps
different things:

* seconds, not frames. The file is authored in seconds and a clip is played in
  seconds, so converting to policy steps and back would only introduce the
  frames-vs-wall-clock rounding the players already work around.
* every segment, including the non-task ones. Scoring folds ``mistake``,
  ``reset both arms``, ``adjust`` and ``trim`` into ignore regions; a viewer
  wants to see that a span was annotated and excluded rather than see a hole.
  ``is_task=False`` marks them so the strip can grey them out.

Distinct from :mod:`behavior_graphs.annotations.subtasks`, which is the *editable*
per-episode store the annotation page writes (``annotations/subtasks/ep0000.json``).
A dataset can have either, both or neither; this module only reads the
benchmark's single-file ground truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from behavior_graphs.stages.evaluation._boundary_metrics import (
    NON_TASK_LABELS,
    load_annotations,
)

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GtSegment:
    """One annotated span of an episode, in seconds."""

    start_sec: float
    end_sec: float
    label: str
    is_task: bool


@dataclass(frozen=True)
class GtEpisode:
    """One episode's ground truth, plus what the strip needs to lay it out."""

    segments: tuple[GtSegment, ...]
    duration_sec: float
    instruction: str

    @property
    def n_task_segments(self) -> int:
        return sum(1 for s in self.segments if s.is_task)


def _fingerprint(path: Path) -> tuple[float, int]:
    """(mtime, size) so an edited annotation file invalidates the cache."""
    try:
        st_ = path.stat()
    except OSError:
        return (0.0, 0)
    return (st_.st_mtime, st_.st_size)


@st.cache_data(show_spinner=False, max_entries=16)
def _load(root_str: str, _fp: tuple[float, int]) -> dict[int, GtEpisode]:
    """``episode index -> GtEpisode``, or ``{}`` when the dataset has no file.

    Cached on (path, mtime, size) rather than by TTL: annotations are authored
    once per dataset and the file is a few hundred KB, so re-reading it on a
    timer would cost more than it could ever catch.
    """
    out: dict[int, GtEpisode] = {}
    try:
        raw = load_annotations(root_str)
    except FileNotFoundError:
        return out
    except (OSError, ValueError) as e:
        # A malformed file must not take the page down: the videos and the
        # graph are still worth showing without the strip.
        _log.warning("ground-truth segments at %s unreadable (%s)", root_str, e)
        return out

    for ep in raw:
        try:
            idx = int(ep["index"])
            segs = ep.get("segments") or []
            spans = tuple(
                GtSegment(
                    start_sec=float(s["start_sec"]),
                    end_sec=float(s["end_sec"]),
                    label=str(s["subtask"]).strip(),
                    is_task=str(s["subtask"]).strip().lower() not in NON_TASK_LABELS,
                )
                for s in segs
            )
        except (KeyError, TypeError, ValueError) as e:
            _log.warning("skipping a malformed ground-truth episode in %s (%s)", root_str, e)
            continue
        if not spans:
            continue
        # `duration_sec` is the clip's own length and is what the player scales
        # against; the last segment's end is the fallback for a file that omits
        # it, and the two agree wherever both are present.
        duration = float(ep.get("duration_sec") or 0.0) or spans[-1].end_sec
        out[idx] = GtEpisode(
            segments=spans,
            duration_sec=duration,
            instruction=str(ep.get("instruction") or ""),
        )
    return out


def ground_truth(rollout_dir: Path | str | None) -> dict[int, GtEpisode]:
    """Every annotated episode of *rollout_dir*, keyed by episode index.

    ``{}`` for a dataset with no ground truth, which is the common case and not
    an error — callers show the strip only when there is something to show.
    """
    if rollout_dir is None:
        return {}
    root = Path(rollout_dir)
    return _load(str(root), _fingerprint(root / "annotations" / "segments.json"))


def episode_ground_truth(rollout_dir: Path | str | None, episode_idx: int) -> GtEpisode | None:
    """One episode's ground truth, or None if the dataset or episode lacks it."""
    return ground_truth(rollout_dir).get(int(episode_idx))
