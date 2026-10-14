"""A copy of each attempt in a bucket, so the study is not one disk.

The responses live one JSON file per attempt under ``STUDY_RESULTS_DIR``, on
the workstation serving the app. That directory *is* the study: there is no
replica, and a lost disk or a stray ``rm`` takes every participant with it.
This sends a copy off the box while the study is running.

Set ``STUDY_BACKUP_BUCKET`` to turn it on. Unset, nothing happens and nothing is
logged, which is the state a laptop or a test run wants.

The copy is written by :class:`behavior_graphs.viz.streamlit.user_study
.response_store.GCSResponseStore`, the same class the study uses when
``SURVEY_GCS_BUCKET`` makes a bucket the *primary* store. So the layout in the
backup bucket is the layout that class reads, and recovering is pointing the app
at it rather than writing a converter. One blob per attempt, overwritten, named
for the session.

Recovery needs *both* variables, because the mirror writes under its own prefix
rather than the primary store's::

    SURVEY_GCS_BUCKET=behavior-graphs-data SURVEY_GCS_PREFIX=study_results/

Naming only the bucket lists nothing and reports an empty study, which is not
what anybody wants to discover at the moment the local disk is gone.

Two things this deliberately is not:

* It is not the primary store. ``SURVEY_GCS_BUCKET`` already makes a bucket the
  only store; this is for keeping the local file *and* having a copy.
* It is not a guarantee. Every failure here is swallowed and logged. A bucket
  that is unreachable, a credential that has expired, a quota that is spent:
  none of them may take down a page a participant is halfway through, and the
  local file remains the record of what happened.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import streamlit as st

_log = logging.getLogger(__name__)

#: Bucket to mirror into. Distinct from ``SURVEY_GCS_BUCKET``, which replaces
#: the local store rather than copying it.
BACKUP_BUCKET_ENV = "STUDY_BACKUP_BUCKET"

#: Where the copies land inside the bucket. Not ``survey_responses/``, which is
#: where the same class writes when a bucket is the *primary* store: a mirror
#: sharing that prefix would be indistinguishable from a primary-store response
#: if both were ever pointed at one bucket.
BACKUP_PREFIX = "study_results/"


@st.cache_resource(show_spinner=False)
def _backup_store(bucket: str) -> Any:
    """One client per bucket per process. Building one costs a round trip."""
    from behavior_graphs.viz.streamlit.user_study.response_store import GCSResponseStore

    return GCSResponseStore(bucket, prefix=BACKUP_PREFIX)


def backup_bucket() -> str:
    """The configured bucket, or ``""`` when mirroring is off."""
    return os.environ.get(BACKUP_BUCKET_ENV, "").strip()


def mirror(record: dict, *, once_key: str) -> bool:
    """Copy one attempt to the bucket. Returns whether it was sent.

    *once_key* is a session-state key that makes this at-most-once for whatever
    the caller decides an occasion is. The three-stage layout mirrors at each
    intermission and at submit, so a participant who leaves mid-study has
    everything up to the last intermission off the box.

    The key is set only after a send succeeds, so a failure is retried at the
    next occasion. For the submit-time copy that occasion is the participant
    coming back to their link, which is the only one that outlives the websocket
    the submit happened on.

    Never raises. See the module docstring: a study page is not the place to
    discover that a bucket is unreachable.
    """
    bucket = backup_bucket()
    if not bucket or not record.get("session_id"):
        return False
    if st.session_state.get(once_key):
        return False
    try:
        _backup_store(bucket).save(record)
    except Exception as error:  # unreachable bucket, bad credentials, quota, ...
        _log.warning("could not mirror study session to %s: %s", bucket, error)
        return False
    # Only after it worked, so a failure is retried at the next intermission
    # rather than marked done and skipped for the rest of the session.
    st.session_state[once_key] = True
    return True
