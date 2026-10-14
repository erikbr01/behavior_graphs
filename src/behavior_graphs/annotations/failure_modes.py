"""Failure-mode annotations: which way a failed rollout actually went wrong.

The ground truth for the study's first question. Participants rank the failure
modes they think a policy exhibits, off a behavior graph or off the rollout
videos, and the two displays produce visibly different rankings — but nothing in
the recorded data says which ranking is *right*. A rollout dataset carries a
binary ``success`` per episode and nothing else about how a failure failed, so
the answers can be compared with each other and not with the truth. This is that
missing column, labelled by hand against the footage.

The unit is one episode and one mode: what a rollout is an example of. Deliberately
a single pick rather than a set — the study asks a participant to *rank* modes by
how often they occur, so the ground truth it is scored against is the frequency
of each mode over the dataset, and a frequency needs each episode counted once.
An episode that genuinely shows two failures is what :data:`OTHER` and its note
are for; it is rare enough to be worth reading afterwards rather than modelling.

The options are the study's own wordings, taken from the question bank
(``user_studies.variants.contrastive_same_page.questions.QUESTION_SETS``) so a
label here and a participant's answer are the same string and can be compared
without a mapping — plus :data:`OTHER`, for a rollout none of them describes.
This module does not import that bank: it stores whatever mode the caller passes
and records which bank it came from, so a wording the bank drops later still
reads back as itself (see :meth:`FailureLabel.is_other`) instead of vanishing.

Storage
-------
One JSON file per episode, inside the dataset it describes::

    <rollout_dir>/annotations/failure_modes/ep0000.json

Same layout, and the same ``$BEHAVIOR_GRAPHS_ANNOTATIONS_ROOT`` redirect for a
read-only mount, as the subtask annotations next door — see
:mod:`behavior_graphs.annotations.store`. One file per episode rather than one per
dataset for the reason that store gives: 97 clips is a pass two people can split
between them, and a whole-dataset file would let the second one's save drop the
first one's labels with no error and no way to notice.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from behavior_graphs.annotations import store
from behavior_graphs.core.io import atomic_write_json

_log = logging.getLogger(__name__)

#: Bumped only for a change a reader cannot handle transparently.
SCHEMA_VERSION = 1

FAILURE_MODES_DIR = "failure_modes"

#: The mode for a rollout none of the study's options describes. A sentinel
#: rather than the string "other", so it cannot collide with a bank wording and
#: so a reader can tell "the annotator chose Other" from "the annotator chose an
#: option whose wording has since changed".
OTHER = "__other__"

#: What to call :data:`OTHER` on screen and in an export.
OTHER_LABEL = "Other"

_EP_FILE_RE = re.compile(r"^ep(\d+)\.json$")


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FailureLabel:
    """One episode's failure mode.

    ``mode`` is the study's own wording for the mode, or :data:`OTHER`. ``note``
    is free text — required by the page when the mode is ``OTHER``, since "other"
    with nothing said about it is not a ground-truth label, and optional
    otherwise.

    ``question_set`` records which task's bank the wording came from. It is what
    lets a reader tell a dropped wording from a foreign one: a mode that is no
    longer among its bank's options is still a real annotation of a real
    episode, and it must read back as that rather than as an unlabelled episode.
    """

    episode: int
    mode: str
    note: str = ""
    question_set: str = ""
    #: Copied off the episode when the label is written, so an export says what
    #: was annotated without re-reading the dataset — and so a label written
    #: against a rollout later relabelled as a success is still legible as one
    #: made about a failure.
    success: bool | None = None
    #: When it was written. No annotator goes with it: one person is doing this
    #: pass, and a name on every episode is a field to keep right rather than
    #: something a reader of the ground truth would ever ask.
    updated_at: str = ""

    @property
    def is_other(self) -> bool:
        return self.mode == OTHER

    def label(self, options: Iterable[str] = ()) -> str:
        """How to name this mode on screen.

        *options* is the bank as it stands now; a mode that is not in it is
        marked rather than silently shown as though it were still on offer.
        """
        if self.is_other:
            return f"{OTHER_LABEL}: {self.note}" if self.note else OTHER_LABEL
        options = tuple(options)
        if options and self.mode not in options:
            return f"{self.mode} (no longer offered)"
        return self.mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode": int(self.episode),
            "mode": self.mode,
            "note": self.note,
            "question_set": self.question_set,
            "success": self.success,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FailureLabel:
        version = int(raw.get("schema_version") or 1)
        if version > SCHEMA_VERSION:
            raise ValueError(f"failure-mode annotation schema v{version} is from the future")  # noqa: TRY003
        mode = str(raw.get("mode") or "")
        if not mode:
            raise ValueError("failure-mode annotation has no mode")  # noqa: TRY003
        success = raw.get("success")
        return cls(
            episode=int(raw["episode"]),
            mode=mode,
            note=str(raw.get("note") or ""),
            question_set=str(raw.get("question_set") or ""),
            success=success if isinstance(success, bool) else None,
            updated_at=str(raw.get("updated_at") or ""),
        )


# ── On-disk layout ───────────────────────────────────────────────────────────


def store_dir(rollout_dir: Path | str) -> Path:
    """Where *rollout_dir*'s failure-mode annotations live."""
    return store.store_dir(rollout_dir, FAILURE_MODES_DIR)


def annotation_path(rollout_dir: Path | str, episode: int) -> Path:
    """The file one episode's label is written to.

    Four-digit episode numbers, matching ``media/ep0000_failure.mp4`` and the
    subtask store, so an ``ls`` of the three lines up. Without the outcome
    suffix those carry: an episode relabelled by the outcome-review tools must
    not orphan the failure mode somebody annotated on it.
    """
    return store_dir(rollout_dir) / f"ep{int(episode):04d}.json"


def load(rollout_dir: Path | str, episode: int) -> FailureLabel | None:
    """One episode's label, or ``None`` when it has never been annotated.

    A corrupt file reads as ``None`` with a warning rather than raising: the
    caller is a page rendering an episode, and one bad file should cost that
    episode's label, not the page.
    """
    path = annotation_path(rollout_dir, episode)
    if not path.is_file():
        return None
    try:
        return FailureLabel.from_dict(json.loads(path.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as e:
        _log.warning("ignoring unreadable failure-mode annotation %s (%s)", path, e)
        return None


def save(rollout_dir: Path | str, label: FailureLabel) -> Path:
    """Persist *label*, stamping the time it was written."""
    if not label.mode:
        raise ValueError("a failure-mode annotation needs a mode")  # noqa: TRY003
    path = annotation_path(rollout_dir, episode=label.episode)
    label.updated_at = _now()
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, label.to_dict())
    return path


def delete(rollout_dir: Path | str, episode: int) -> bool:
    """Remove one episode's label. ``True`` when a file was there.

    Deleting rather than writing a blank mode: an episode nobody has judged and
    an episode judged to be nothing in particular are different, and only the
    first should leave the progress count alone.
    """
    path = annotation_path(rollout_dir, episode)
    try:
        existed = path.is_file()
        path.unlink(missing_ok=True)
    except OSError as e:
        _log.warning("could not delete failure-mode annotation %s (%s)", path, e)
        return False
    return existed


def load_all(rollout_dir: Path | str) -> dict[int, FailureLabel]:
    """``{episode: label}`` for every annotated episode of a dataset.

    Keyed by the episode number in the *file body*, not the filename, so a file
    someone renamed by hand cannot silently claim a different episode.
    """
    directory = store_dir(rollout_dir)
    if not directory.is_dir():
        return {}
    out: dict[int, FailureLabel] = {}
    for path in sorted(directory.iterdir()):
        if not _EP_FILE_RE.match(path.name):
            continue
        try:
            label = FailureLabel.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as e:
            _log.warning("ignoring unreadable failure-mode annotation %s (%s)", path, e)
            continue
        out[label.episode] = label
    return dict(sorted(out.items()))


# ── Summaries ────────────────────────────────────────────────────────────────


def mode_counts(labels: Mapping[int, FailureLabel]) -> dict[str, int]:
    """``{mode: episodes}``, commonest first — the ground-truth distribution.

    What a participant's ranking is scored against: they are asked which modes
    occur most, so the answer is this ordering. ``OTHER`` is counted as one mode
    however many different notes it carries, because a bag of one-off notes is
    not a rank the study offered anybody.
    """
    counts = Counter(label.mode for label in labels.values())
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def export_rows(labels: Mapping[int, FailureLabel], dataset_name: str) -> list[dict[str, Any]]:
    """One row per annotated episode, ready to hand to an analysis."""
    return [
        {
            "dataset": dataset_name,
            "episode": episode,
            "mode": label.mode,
            "is_other": label.is_other,
            "note": label.note,
            "question_set": label.question_set,
            "success": label.success,
            "updated_at": label.updated_at,
        }
        for episode, label in sorted(labels.items())
    ]
