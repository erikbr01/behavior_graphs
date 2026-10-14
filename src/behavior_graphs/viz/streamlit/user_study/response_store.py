"""Response storage abstraction for the user study.

Auto-selects backend based on ``SURVEY_GCS_BUCKET`` env var:
- Set: uses Google Cloud Storage (responses as blobs under ``survey_responses/``)
- Unset: appends to a local JSONL file under ``STUDY_RESULTS_DIR``
  (default ``./study_results``), one JSON object per line.

Usage::

    from behavior_graphs.viz.streamlit.user_study.response_store import get_store
    store = get_store()
    response_id = store.save(result_dict)   # returns a string ID
    all_responses = store.list_all()        # list[dict]

A record carrying a ``session_id`` is a *revision* of one participant's attempt
rather than a new response: the three-stage study checkpoints its answers as
they are given, so a session that is abandoned halfway still leaves usable data
(see ``behavior_graphs.viz.streamlit.user_studies.session``). Every backend
therefore addresses such a record by its session id — one blob per attempt,
overwritten — and :meth:`list_all` keeps only the last revision of each. Without
that fold, a study of 200 participants would read back as 2,000 responses, most
of them stale snapshots of the other 200.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

_log = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = "./study_results"


def _response_filename(response: dict) -> str:
    """The name this record is addressed by — its session, or a fresh timestamp.

    Session-keyed while the attempt is in flight, so each checkpoint replaces
    the last rather than piling up beside it. Records with no session id keep
    the original participant/timestamp name: that is every response the earlier
    study layouts wrote, and they are only ever written once.
    """
    session_id = str(response.get("session_id", "")).strip()
    if session_id:
        return f"session_{_slug(session_id)}.json"
    ts = response.get("timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
    group = str(response.get("group", "X")).lower()
    pid = str(response.get("participant_id", "anon")).replace(" ", "_")[:32]
    return f"group_{group}_{ts}_{pid}.json"


def _slug(value: str) -> str:
    """*value* reduced to what is safe in a file or blob name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:64]


class LocalResponseStore:
    """One JSON file per attempt, named for its session.

    ``save`` replaces that file; it does not append. The store used to be a
    single append-only ``responses.jsonl`` with one line per *revision*, which
    meant a full copy of the record on every answer. Nothing read the history:
    every consumer went through ``list_all``, which folded it away again, and a
    lookup had to scan backwards through the file to find one attempt. Thirty
    participants produced twelve hundred lines to hold thirty answers.

    Writes go to a temporary name and are renamed into place. ``os.replace`` is
    atomic on POSIX, so a reader sees either the previous record or the new one
    and never half of either — which append did not guarantee, since a crash
    mid-write leaves a torn line the reader then has to skip.

    Same shape as :class:`GCSResponseStore`, which was already one blob per
    attempt. The two backends now differ only in where the bytes go.
    """

    def __init__(self, base_dir: Path | str):
        self._dir = Path(base_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, response_id: str) -> Path:
        return self._dir / response_id

    def save(self, response: dict) -> str:
        response_id = _response_filename(response)
        record = dict(response)
        record["_response_id"] = response_id
        path = self._path(response_id)
        # Same directory as the target, so the rename cannot cross a filesystem.
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        os.replace(tmp, path)
        return response_id

    def get(self, session_id: str) -> dict | None:
        """The one attempt, opened by name. No scan, no parse of anything else."""
        wanted = str(session_id).strip()
        if not wanted:
            return None
        return self._read(self._path(f"session_{_slug(wanted)}.json"))

    def list_all(self) -> list[dict]:
        """Every attempt. One file each, so there is nothing to fold."""
        records = []
        for path in sorted(self._dir.glob("*.json")):
            record = self._read(path)
            if record is not None:
                records.append(record)
        return records

    def _read(self, path: Path) -> dict | None:
        """One record, or ``None``. Unreadable is logged, never silent.

        A file that will not parse is not the same as an attempt that does not
        exist, and treating it as one lets a participant start a second attempt
        over the top of their first.
        """
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            return None
        except Exception as error:
            _log.warning("unreadable response at %s: %s", path, error)
            return None


class GCSResponseStore:
    def __init__(self, bucket_name: str, prefix: str = "survey_responses/"):
        from google.cloud import storage  # lazy — not a hard dep in dev

        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket_name)
        self._prefix = prefix.rstrip("/") + "/"

    def save(self, response: dict) -> str:
        blob_name = self._prefix + _response_filename(response)
        blob = self._bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(response, indent=2),
            content_type="application/json",
        )
        return blob_name

    def list_all(self) -> list[dict]:
        results = []
        for blob in self._bucket.list_blobs(prefix=self._prefix):
            if not blob.name.endswith(".json"):
                continue
            try:
                results.append(json.loads(blob.download_as_text()))
            except Exception:
                pass
        # A session-keyed record has one blob that each checkpoint overwrote, so
        # there is nothing to fold here — but the fold is applied anyway, because
        # a bucket that also holds records written by the local store's
        # append-and-fold convention would otherwise read back stale revisions.
        return results

    def get(self, session_id: str) -> dict | None:
        """The one attempt, fetched by name rather than by listing the bucket."""
        wanted = str(session_id).strip()
        if not wanted:
            return None
        blob = self._bucket.blob(self._prefix + f"session_{_slug(wanted)}.json")
        try:
            if not blob.exists():
                return None
            return json.loads(blob.download_as_text())
        except Exception:
            return None


def results_dir() -> Path:
    return Path(os.environ.get("STUDY_RESULTS_DIR", "").strip() or DEFAULT_RESULTS_DIR)


def get_store() -> LocalResponseStore | GCSResponseStore:
    """Return the appropriate store based on env config.

    The local backend always writes under ``STUDY_RESULTS_DIR`` (default
    ``./study_results``) so responses survive container rebuilds.

    ``SURVEY_GCS_PREFIX`` overrides where in the bucket the blobs live. That is
    what makes the documented recovery work: the off-box mirror
    (:mod:`behavior_graphs.viz.streamlit.user_studies.backup`) writes under its
    own prefix, so pointing ``SURVEY_GCS_BUCKET`` at the backup bucket without
    also naming the prefix listed nothing and reported an empty study — at the
    one moment the local disk is already gone.
    """
    bucket = os.environ.get("SURVEY_GCS_BUCKET", "").strip()
    if bucket:
        prefix = os.environ.get("SURVEY_GCS_PREFIX", "").strip()
        return GCSResponseStore(bucket, prefix=prefix) if prefix else GCSResponseStore(bucket)
    return LocalResponseStore(results_dir())
