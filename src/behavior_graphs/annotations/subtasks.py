"""Subtask annotations: which frames of an episode are which behavior.

The ground truth a behavior graph's segmentation is scored against — the same
thing WGO-bench provides for a handful of tasks, collected here over any
dataset in the canonical rollout layout.

The unit is a **segment**: a half-open frame span ``[start, end)`` with an
optional label, edited like a clip on a video-editor timeline (in/out markers,
draggable edges, split, merge). Spans rather than bare boundaries because a
subtask has a beginning *and* an end, and the difference is real: the frames
between two subtasks — reset, idle, a regrasp nobody wants to call either — are
a gap, not a misplaced boundary. A fully contiguous annotation is the common
case and reads back as a boundary set (:meth:`EpisodeAnnotation.boundaries`),
so scoring against a segmentation does not have to care which way it was drawn.

Storage
-------
One JSON file per episode, inside the dataset it describes::

    <rollout_dir>/annotations/subtasks/ep0000.json

Inside the rollout dir so an annotation cannot drift away from the episode it
labels: copy, move or archive a dataset and its annotations come along, and
"which dataset is this about" is the path rather than a field someone has to
keep in sync. One file per episode rather than one per dataset so that two
annotators on the same dataset cannot clobber each other's episodes, and so a
single write is atomic without holding a lock over 1000 episodes' worth of
state. :func:`load_all` reads them back as a dataset-level mapping, and
``scripts/export_subtask_annotations.py`` flattens that into the single file an
evaluation wants.

``$BEHAVIOR_GRAPHS_ANNOTATIONS_ROOT`` redirects the whole tree to
``<root>/<dataset name>/`` for a dataset on a read-only mount. Off by default:
the co-located form is the one that survives being moved.

Semantics
---------
Frames, not seconds: the features, the windows and the graph are all indexed by
policy step, and for a low-Hz VLA the clip's wall-clock duration does not
divide into steps evenly. Spans are half-open and ``0 <= start < end <=
num_frames``, kept sorted and **non-overlapping** — a drag that would cross a
neighbour is clamped to it, which is what makes dragging the shared edge of two
adjacent segments behave like moving the boundary between them.

``reviewed`` is what separates "watched it, it is one single subtask" from "not
looked at yet". An evaluation must count only reviewed episodes, which is why
an unreviewed episode with nothing on it is not written to disk at all (see
:func:`save`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from behavior_graphs.annotations import store
from behavior_graphs.core.io import atomic_write_json

_log = logging.getLogger(__name__)

#: Bumped only for a change a reader cannot handle transparently. Readers
#: accept anything <= this and fill missing optional fields with defaults.
SCHEMA_VERSION = 1

#: Re-exported from :mod:`behavior_graphs.annotations.store`, which owns the
#: on-disk layout every annotation kind shares.
ANNOTATIONS_DIR = store.ANNOTATIONS_DIR
ENV_ANNOTATIONS_ROOT = store.ENV_ANNOTATIONS_ROOT

SUBTASKS_DIR = "subtasks"

ENV_ANNOTATOR = "BEHAVIOR_GRAPHS_ANNOTATOR"

#: Shortest span the editor will create or leave behind. One frame, so a
#: single-step behavior is expressible; the clamps below rely on it to keep
#: ``start < end`` through any drag.
MIN_SPAN = 1

_EP_FILE_RE = re.compile(r"^ep(\d+)\.json$")


def annotator_name() -> str:
    """Who is annotating, for the record kept on each episode.

    ``$BEHAVIOR_GRAPHS_ANNOTATOR`` first so a shared deployment can name the
    person rather than the account the server happens to run as.
    """
    for var in (ENV_ANNOTATOR, "USER", "USERNAME"):
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_segment_id() -> str:
    """A short, collision-free segment id.

    Short because it is read by a human debugging an annotation file, and
    random because an id must survive its segment being dragged past a
    neighbour — anything derived from the frames or the ordering would not.
    """
    return uuid.uuid4().hex[:8]


# ── Model ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Segment:
    """One annotated behavior: frames ``[start, end)``, optionally named.

    Frozen: every edit goes through :class:`EpisodeAnnotation`, which is the
    only thing that can keep the invariants (sorted, in range, non-overlapping)
    true. A method here that moved an edge could not see the neighbours.
    """

    id: str
    start: int
    end: int
    label: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start

    def contains(self, frame: int) -> bool:
        return self.start <= frame < self.end

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "start": int(self.start), "end": int(self.end)}
        if self.label:
            out["label"] = self.label
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Segment:
        return cls(
            id=str(raw.get("id") or new_segment_id()),
            start=int(raw["start"]),
            end=int(raw["end"]),
            label=str(raw.get("label") or ""),
        )


@dataclass
class EpisodeAnnotation:
    """Every annotated subtask on one episode, plus who said so and when.

    The mutating methods enforce what the file format promises — sorted, in
    range, non-overlapping — and report whether they changed anything, so a
    caller can skip a write and an ignored edit can be surfaced instead of
    silently swallowed. None of them touch the disk; :func:`save` does that.
    """

    dataset: str
    episode: int
    num_frames: int
    fps: float = 10.0
    success: bool | None = None
    #: The task/instruction string recorded with the episode, when there is one.
    #: Copied in rather than looked up on read: an annotation should still say
    #: what was being attempted after the rollout metadata has moved on.
    task: str = ""
    segments: list[Segment] = field(default_factory=list)
    notes: str = ""
    #: Someone has watched this episode and is done with it. The only thing that
    #: distinguishes "the whole episode is one subtask" from "never opened".
    reviewed: bool = False
    annotator: str = ""
    updated_at: str = ""
    schema_version: int = SCHEMA_VERSION

    # ── Queries ──────────────────────────────────────────────────────────

    @property
    def boundaries(self) -> list[int]:
        """The interior frame boundaries the segments imply, ascending.

        What a segmentation metric compares against: every segment edge strictly
        inside the episode, with the shared edge of two adjacent segments counted
        once. Frame 0 and frame ``num_frames`` are excluded — they are where the
        recording starts and stops, not decisions anyone made.

        Note the difference from WGO-Bench's convention (``WGOEpisode.
        boundaries``), which takes only the *starts* of the second segment
        onwards. The two agree exactly on a contiguous annotation that runs the
        whole episode. They diverge where ours has a gap: the frame a subtask
        stops at, with nothing following it until later, is a real event here and
        is scored, whereas WGO-Bench treats its single trailing gap as the end of
        the annotated span (:attr:`annotated_end`) instead.
        """
        edges = {s.start for s in self.segments} | {s.end for s in self.segments}
        return sorted(f for f in edges if 0 < f < self.num_frames)

    @property
    def annotated_start(self) -> int:
        """First frame any subtask covers; ``0`` when nothing is annotated."""
        return self.segments[0].start if self.segments else 0

    @property
    def annotated_end(self) -> int:
        """Last frame any subtask covers, exclusive.

        The end of what an annotator vouched for. A predicted boundary past it
        has nothing to be right or wrong about, so an evaluation should clip
        there rather than charge it as a false positive — the same role
        ``WGOEpisode.annotated_end`` plays.
        """
        return self.segments[-1].end if self.segments else self.num_frames

    def seconds(self, frame: int) -> float:
        """*frame* as a time offset, for a metric that scores with a tolerance.

        Through :attr:`fps`, which is the *control* rate the collector recorded —
        so this is policy-step time, the axis the annotation was made on. It is
        not the clip's wall-clock time for a low-Hz VLA, where the video runs at
        real speed while the policy stepped at a few Hz; the editor maps the
        cursor to video time separately for exactly that reason.
        """
        return frame / self.fps if self.fps > 0 else float(frame)

    def find(self, segment_id: str) -> Segment | None:
        return next((s for s in self.segments if s.id == segment_id), None)

    def index_of(self, segment_id: str) -> int:
        return next((i for i, s in enumerate(self.segments) if s.id == segment_id), -1)

    def at(self, frame: int) -> Segment | None:
        """The segment covering *frame*, or None when it falls in a gap."""
        return next((s for s in self.segments if s.contains(frame)), None)

    def nearest(self, frame: int) -> Segment | None:
        """The segment covering *frame*, else the one whose edge is closest.

        What "delete" and "select" act on when the cursor is not inside
        anything: an annotator who overshot the end of a pill by two frames
        means that pill, not nothing.
        """
        if not self.segments:
            return None
        inside = self.at(frame)
        if inside is not None:
            return inside
        return min(self.segments, key=lambda s: min(abs(s.start - frame), abs(s.end - frame)))

    def gaps(self) -> list[tuple[int, int]]:
        """The ``[start, end)`` spans no segment covers, ascending.

        Rendered as unfilled track in the editor and reported as coverage: on a
        long episode the thing you cannot see by eye is the two-second hole you
        left in the middle.
        """
        out: list[tuple[int, int]] = []
        cursor = 0
        for s in self.segments:
            if s.start > cursor:
                out.append((cursor, s.start))
            cursor = max(cursor, s.end)
        if cursor < self.num_frames:
            out.append((cursor, self.num_frames))
        return out

    def covered_frames(self) -> int:
        return sum(s.length for s in self.segments)

    def coverage(self) -> float:
        """Fraction of the episode inside some segment, 0.0–1.0."""
        if self.num_frames <= 0:
            return 0.0
        return self.covered_frames() / self.num_frames

    def is_contiguous(self) -> bool:
        """True when the segments tile the whole episode with no gaps."""
        return bool(self.segments) and not self.gaps()

    def is_empty(self) -> bool:
        """Nothing here anyone would miss — see :func:`save`."""
        return not self.segments and not self.notes.strip() and not self.reviewed

    def clamp(self, frame: int) -> int:
        """*frame* forced into the episode."""
        return max(0, min(int(frame), self.num_frames))

    def free_span(self, frame: int) -> tuple[int, int]:
        """The gap around *frame*, as ``(start, end)``; empty when it is covered.

        What an in/out pair is allowed to grow into, and what the ``O``-only
        workflow uses to decide where the segment it is about to create begins.
        """
        for start, end in self.gaps():
            if start <= frame < end or (frame == end == self.num_frames):
                return start, end
        return frame, frame

    # ── Mutations ────────────────────────────────────────────────────────

    def add_segment(self, start: int, end: int, label: str = "") -> Segment | None:
        """Add ``[start, end)``, trimmed to the free space around it.

        None when there is no room — the span is inverted, or it lands entirely
        inside an existing segment. Trimming rather than rejecting an overlap:
        the caller is an out-marker dropped a few frames past the neighbour it
        was chasing, and stopping at that neighbour is what was meant.
        """
        start, end = self.clamp(min(start, end)), self.clamp(max(start, end))
        if end - start < MIN_SPAN:
            # An empty span expresses nothing, so it is refused rather than
            # rounded up to one frame — an out-marker dropped on the in-marker is
            # a mis-timed keypress, and the editor says so instead of leaving a
            # one-frame pill behind for the annotator to find later.
            return None
        # Grow-into-the-gap, from the *start* side: the in-point is the
        # deliberate end of the pair (the out-point is wherever playback got
        # to), so it decides which gap this segment belongs in.
        gap_start, gap_end = self.free_span(start)
        if gap_end - gap_start < MIN_SPAN:
            return None
        start = max(start, gap_start)
        end = min(max(end, start + MIN_SPAN), gap_end)
        if end - start < MIN_SPAN:
            return None
        segment = Segment(id=new_segment_id(), start=start, end=end, label=label)
        self.segments.append(segment)
        self._resort()
        return segment

    def remove_segment(self, segment_id: str) -> Segment | None:
        """Drop one segment; None when the id is unknown."""
        i = self.index_of(segment_id)
        if i < 0:
            return None
        return self.segments.pop(i)

    def set_span(
        self,
        segment_id: str,
        start: int,
        end: int,
        *,
        roll: bool = False,
        keep_length: bool = False,
    ) -> bool:
        """Move or resize one segment, clamped to its neighbours.

        *roll* is the video-editor roll edit, and the reason dragging feels like
        moving a boundary: when the edge being moved is *shared* with an
        adjacent segment, that neighbour's edge follows instead of blocking it.
        Neighbours that merely sit nearby are still hard stops — a gap between
        two behaviors is a thing someone drew, not slack to be consumed.

        *keep_length* distinguishes the two gestures that arrive here. Dragging a
        pill's *body* means "this span, somewhere else", so it slides up against
        the neighbour it ran into with its length intact; dragging an *edge*
        means "this edge, there", so each edge is clamped on its own and the
        length is whatever survives. Without the distinction a pill dragged into
        its neighbour would arrive squashed to a single frame against it.

        False when the id is unknown, there is no room at all, or nothing moved.
        """
        i = self.index_of(segment_id)
        if i < 0:
            return False
        old = self.segments[i]
        start, end = self.clamp(min(start, end)), self.clamp(max(start, end))
        if end - start < MIN_SPAN:
            return False

        rolled = False
        if roll:
            # Before the clamp below, so the limits it reads are the ones the
            # roll has already moved. Each neighbour keeps MIN_SPAN of its own.
            if i > 0 and self.segments[i - 1].end == old.start and start != old.start:
                prev = self.segments[i - 1]
                moved = Segment(
                    id=prev.id,
                    start=prev.start,
                    end=max(prev.start + MIN_SPAN, min(start, prev.end + (start - old.start))),
                    label=prev.label,
                )
                rolled = rolled or moved != prev
                self.segments[i - 1] = moved
            if (
                i + 1 < len(self.segments)
                and self.segments[i + 1].start == old.end
                and end != old.end
            ):
                nxt = self.segments[i + 1]
                moved = Segment(
                    id=nxt.id,
                    start=min(nxt.end - MIN_SPAN, max(end, nxt.start + (end - old.end))),
                    end=nxt.end,
                    label=nxt.label,
                )
                rolled = rolled or moved != nxt
                self.segments[i + 1] = moved

        low = self.segments[i - 1].end if i > 0 else 0
        high = self.segments[i + 1].start if i + 1 < len(self.segments) else self.num_frames
        if high - low < MIN_SPAN:
            return False
        wanted = end - start
        if keep_length and wanted <= high - low:
            start = max(low, min(start, high - wanted))
            end = start + wanted
        else:
            start = max(low, min(start, high - MIN_SPAN))
            end = min(high, max(end, start + MIN_SPAN))
        new = Segment(id=old.id, start=start, end=end, label=old.label)
        if new == old and not rolled:
            # Nothing moved — not this segment, and not a neighbour a roll could
            # have taken with it. Saying otherwise costs the caller a write, a
            # fresh ``updated_at`` and an invalidated cache for an edge drag that
            # resolved to the frame it started on, which on a zoomed-out track is
            # most of the short ones.
            return False
        self.segments[i] = new
        self._resort()
        return True

    def nudge(self, segment_id: str, delta: int) -> bool:
        """Slide a whole segment by *delta* frames, keeping its length if it can."""
        segment = self.find(segment_id)
        if segment is None or delta == 0:
            return False
        return self.set_span(
            segment_id, segment.start + delta, segment.end + delta, keep_length=True
        )

    def set_label(self, segment_id: str, label: str) -> bool:
        i = self.index_of(segment_id)
        if i < 0 or self.segments[i].label == label:
            return False
        old = self.segments[i]
        self.segments[i] = Segment(id=old.id, start=old.start, end=old.end, label=label)
        return True

    def split_segment(self, segment_id: str, frame: int) -> Segment | None:
        """Cut one segment in two at *frame*; returns the new right-hand half.

        None when the frame is not strictly inside the segment with at least
        :data:`MIN_SPAN` on each side — a split that would produce an empty half
        is a mis-aimed keypress, not an edit.
        """
        i = self.index_of(segment_id)
        if i < 0:
            return None
        old = self.segments[i]
        frame = int(frame)
        if not (old.start + MIN_SPAN <= frame <= old.end - MIN_SPAN):
            return None
        right = Segment(id=new_segment_id(), start=frame, end=old.end, label=old.label)
        self.segments[i] = Segment(id=old.id, start=old.start, end=frame, label=old.label)
        self.segments.insert(i + 1, right)
        return right

    def merge_with_next(self, segment_id: str) -> bool:
        """Absorb the following segment (and any gap between) into this one.

        Keeps the earlier segment's id and label so the pill you selected is the
        one that survives, which is what the ``M`` key looks like on screen.
        """
        i = self.index_of(segment_id)
        if i < 0 or i + 1 >= len(self.segments):
            return False
        left, right = self.segments[i], self.segments[i + 1]
        self.segments[i] = Segment(
            id=left.id, start=left.start, end=right.end, label=left.label or right.label
        )
        self.segments.pop(i + 1)
        return True

    def fill_gaps(self) -> int:
        """Extend segments until they tile the episode; returns segments changed.

        The "I meant this to be contiguous" button. Each gap is split down the
        middle between its two neighbours, and a gap at either end is absorbed
        by the single segment that touches it. Nothing happens without segments
        to grow — an empty episode is not silently made into one big subtask.
        """
        if not self.segments:
            return 0
        changed = 0
        for gap_start, gap_end in self.gaps():
            before = next((i for i, s in enumerate(self.segments) if s.end == gap_start), -1)
            after = next((i for i, s in enumerate(self.segments) if s.start == gap_end), -1)
            if before >= 0 and after >= 0:
                mid = (gap_start + gap_end) // 2
                left, right = self.segments[before], self.segments[after]
                self.segments[before] = Segment(left.id, left.start, mid, left.label)
                self.segments[after] = Segment(right.id, mid, right.end, right.label)
                changed += 2
            elif before >= 0:
                left = self.segments[before]
                self.segments[before] = Segment(left.id, left.start, gap_end, left.label)
                changed += 1
            elif after >= 0:
                right = self.segments[after]
                self.segments[after] = Segment(right.id, gap_start, right.end, right.label)
                changed += 1
        self._resort()
        return changed

    def clear_segments(self) -> int:
        """Drop every segment; returns how many there were."""
        n = len(self.segments)
        self.segments = []
        return n

    def _resort(self) -> None:
        self.segments.sort(key=lambda s: (s.start, s.end))

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "dataset": self.dataset,
            "episode": int(self.episode),
            "num_frames": int(self.num_frames),
            "fps": float(self.fps),
            "success": self.success,
            "task": self.task,
            "segments": [s.to_dict() for s in self.segments],
            # Derived, written anyway: this file is meant to be readable and
            # consumable without importing this module, and the boundary list is
            # what a segmentation metric actually wants. Never read back — see
            # from_dict, which recomputes it from the segments.
            "boundaries": self.boundaries,
            "notes": self.notes,
            "reviewed": bool(self.reviewed),
            "annotator": self.annotator or annotator_name(),
            "updated_at": self.updated_at or _now(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> EpisodeAnnotation:
        version = int(raw.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            # Read it anyway: every field this build knows about means the same
            # thing in a later version (that is what the bump is for), and
            # refusing would lock an annotator out of their own work after a
            # rollback. Say so once so a genuinely incompatible file is not a
            # mystery.
            _log.warning(
                "subtask annotation for %s/ep%s has schema_version %d > %d — "
                "reading the fields this build understands",
                raw.get("dataset"),
                raw.get("episode"),
                version,
                SCHEMA_VERSION,
            )
        ann = cls(
            dataset=str(raw.get("dataset") or ""),
            episode=int(raw["episode"]),
            num_frames=int(raw.get("num_frames") or 0),
            fps=float(raw.get("fps") or 10.0),
            success=raw.get("success"),
            task=str(raw.get("task") or ""),
            segments=[Segment.from_dict(s) for s in raw.get("segments") or []],
            notes=str(raw.get("notes") or ""),
            reviewed=bool(raw.get("reviewed")),
            annotator=str(raw.get("annotator") or ""),
            updated_at=str(raw.get("updated_at") or ""),
            schema_version=version,
        )
        ann.segments = sanitised(ann.segments, ann.num_frames)
        return ann


def sanitised(segments: list[Segment], num_frames: int) -> list[Segment]:
    """*segments* forced back into the invariants, dropping what cannot fit.

    A hand-edited file, or one written by a build whose clamps differed, need
    not be sorted, in range or disjoint — and every reader downstream (the
    editor's timeline, :meth:`EpisodeAnnotation.gaps`) assumes all three. Later
    segments yield to earlier ones so the result is deterministic, and one whose
    room is entirely taken is dropped rather than collapsed to zero length.
    """
    out: list[Segment] = []
    for s in sorted(segments, key=lambda s: (s.start, s.end)):
        start = max(0, min(int(s.start), num_frames))
        end = max(0, min(int(s.end), num_frames))
        if end < start:
            start, end = end, start
        if out:
            start = max(start, out[-1].end)
        if end - start < MIN_SPAN:
            continue
        out.append(Segment(id=s.id, start=start, end=end, label=s.label))
    return out


# ── On-disk layout ───────────────────────────────────────────────────────────


def store_dir(rollout_dir: Path | str) -> Path:
    """Where *rollout_dir*'s subtask annotations live.

    The rule itself — co-located, or redirected by
    ``$BEHAVIOR_GRAPHS_ANNOTATIONS_ROOT`` under a fingerprinted directory name —
    is :func:`behavior_graphs.annotations.store.store_dir`, shared with every
    other kind of annotation in this package. It has to be one function: two
    copies could come to disagree about which dataset a store belongs to, and
    the failure would be one dataset's annotations written over another's.
    """
    return store.store_dir(rollout_dir, SUBTASKS_DIR)


def annotation_path(rollout_dir: Path | str, episode: int) -> Path:
    """The file one episode's annotation is written to.

    Four-digit episode numbers, matching ``episodes/ep0000_succ.pkl`` and
    ``media/ep0000_succ.mp4``, so an ``ls`` of the three directories lines up.
    Without the outcome suffix those carry: the outcome is data, and a rollout
    relabelled by the outcome-review tools must not orphan its annotation.
    """
    return store_dir(rollout_dir) / f"ep{int(episode):04d}.json"


def load(rollout_dir: Path | str, episode: int) -> EpisodeAnnotation | None:
    """One episode's annotation, or None when it has never been annotated.

    A corrupt file reads as None with a warning rather than raising: the caller
    is a page rendering an episode, and one bad file should cost that episode's
    annotation, not the page.
    """
    path = annotation_path(rollout_dir, episode)
    if not path.is_file():
        return None
    try:
        return EpisodeAnnotation.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError) as e:
        _log.warning("ignoring unreadable subtask annotation %s (%s)", path, e)
        return None


def save(rollout_dir: Path | str, ann: EpisodeAnnotation) -> Path | None:
    """Persist *ann*, or delete its file when there is nothing left on it.

    Deleting rather than writing an empty record is what keeps "reviewed: this
    is one single subtask" distinguishable from "never opened": both have no
    segments, and only the first leaves a file. It also means undoing every
    segment on an episode leaves the tree exactly as it was found.

    Returns the path written, or None when the file was removed.
    """
    path = annotation_path(rollout_dir, episode=ann.episode)
    if ann.is_empty():
        try:
            path.unlink(missing_ok=True)
        except OSError as e:
            _log.warning("could not remove emptied annotation %s (%s)", path, e)
        return None
    ann.annotator = ann.annotator or annotator_name()
    ann.updated_at = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, ann.to_dict())
    return path


def delete(rollout_dir: Path | str, episode: int) -> bool:
    """Remove one episode's annotation entirely. True when a file was there."""
    path = annotation_path(rollout_dir, episode)
    try:
        existed = path.is_file()
        path.unlink(missing_ok=True)
    except OSError as e:
        _log.warning("could not delete annotation %s (%s)", path, e)
        return False
    return existed


def load_all(rollout_dir: Path | str) -> dict[int, EpisodeAnnotation]:
    """``{episode: annotation}`` for every annotated episode of a dataset.

    Keyed by the episode number in the *file body*, not the filename, so a file
    someone renamed by hand cannot silently claim a different episode.
    """
    directory = store_dir(rollout_dir)
    if not directory.is_dir():
        return {}
    out: dict[int, EpisodeAnnotation] = {}
    for path in sorted(directory.iterdir()):
        if not _EP_FILE_RE.match(path.name):
            continue
        try:
            ann = EpisodeAnnotation.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError) as e:
            _log.warning("ignoring unreadable subtask annotation %s (%s)", path, e)
            continue
        out[ann.episode] = ann
    return dict(sorted(out.items()))


def vocabulary_of(anns: Mapping[int, EpisodeAnnotation], limit: int = 12) -> list[str]:
    """:func:`label_vocabulary` over annotations already in hand.

    Split out from the path-based form because the app asks four questions of
    the same store on every rerun (segments, progress, vocabulary, export) and
    reading it once for all of them is the difference between a page that keeps
    up with a keypress and one that does not.
    """
    counts: Counter[str] = Counter()
    for ann in anns.values():
        for s in ann.segments:
            if s.label.strip():
                counts[s.label.strip()] += 1
    return [label for label, _ in counts.most_common(limit)]


def label_vocabulary(rollout_dir: Path | str, limit: int = 12) -> list[str]:
    """The labels already used in this dataset, most-used first.

    Offered as one-keypress chips in the editor: over a thousand episodes the
    same dozen behavior names come back, and retyping "grasp the mug" is both
    slow and how a vocabulary ends up with three spellings of one behavior.
    """
    return vocabulary_of(load_all(rollout_dir), limit)


@dataclass(frozen=True)
class Progress:
    """How far along a dataset's annotation is — the page's header line."""

    n_files: int
    n_reviewed: int
    n_with_segments: int
    n_segments: int
    n_incomplete: int

    @property
    def mean_segments(self) -> float:
        """Mean subtasks per *reviewed* episode, or 0.0 when none are.

        Over reviewed episodes only: an episode with segments on it that nobody
        has signed off is work in progress, and counting it would make the
        number move while someone is mid-episode.
        """
        return (self.n_segments / self.n_reviewed) if self.n_reviewed else 0.0


def progress(rollout_dir: Path | str) -> Progress:
    """:class:`Progress` for a dataset, from a single scan of its store."""
    return progress_of(load_all(rollout_dir))


def progress_of(anns: Mapping[int, EpisodeAnnotation]) -> Progress:
    """:class:`Progress` over annotations already in hand — see :func:`vocabulary_of`."""
    reviewed = [a for a in anns.values() if a.reviewed]
    return Progress(
        n_files=len(anns),
        n_reviewed=len(reviewed),
        n_with_segments=sum(1 for a in anns.values() if a.segments),
        n_segments=sum(len(a.segments) for a in reviewed),
        # Signed off but full of holes: worth surfacing, because it is the one
        # failure mode a reviewed-count alone cannot show.
        n_incomplete=sum(1 for a in reviewed if not a.is_contiguous()),
    )


def export_rows(rollout_dir: Path | str) -> list[dict[str, Any]]:
    """One flat row per reviewed episode, for an evaluation to consume.

    Reviewed only, and with every view of the annotation a metric might want:
    the spans for scoring a segmentation, the boundaries for scoring change-point
    detection, and both in frames *and* seconds — the segmentation metrics in
    :mod:`behavior_graphs.wgo_bench.segmentation` score with a time tolerance, so
    handing them the conversion removes the one place a consumer could get
    ``fps`` wrong. ``annotated_start``/``annotated_end`` bound what the annotator
    vouched for, so a predicted boundary outside that span can be clipped rather
    than charged as an error.

    None of it must be handed an episode where "no segments" means "nobody
    looked", which is why this is reviewed-only.
    """
    return export_rows_of(load_all(rollout_dir), Path(rollout_dir).name)


def export_rows_of(
    anns: Mapping[int, EpisodeAnnotation], dataset_name: str
) -> list[dict[str, Any]]:
    """:func:`export_rows` over annotations already in hand — see :func:`vocabulary_of`."""
    rows = []
    for episode, ann in anns.items():
        if not ann.reviewed:
            continue
        rows.append(
            {
                "dataset": ann.dataset or dataset_name,
                "episode": episode,
                "num_frames": ann.num_frames,
                "fps": ann.fps,
                "duration_sec": round(ann.seconds(ann.num_frames), 4),
                "success": ann.success,
                "task": ann.task,
                "segments": [[s.start, s.end] for s in ann.segments],
                "segments_sec": [
                    [round(ann.seconds(s.start), 4), round(ann.seconds(s.end), 4)]
                    for s in ann.segments
                ],
                "labels": [s.label for s in ann.segments],
                "boundaries": ann.boundaries,
                "boundaries_sec": [round(ann.seconds(f), 4) for f in ann.boundaries],
                "annotated_start": ann.annotated_start,
                "annotated_end": ann.annotated_end,
                "annotated_end_sec": round(ann.seconds(ann.annotated_end), 4),
                "n_segments": len(ann.segments),
                "coverage": round(ann.coverage(), 4),
                "contiguous": ann.is_contiguous(),
                "notes": ann.notes,
                "annotator": ann.annotator,
                "updated_at": ann.updated_at,
            }
        )
    return rows
