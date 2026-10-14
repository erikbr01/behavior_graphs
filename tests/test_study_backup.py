"""Copying each attempt off the machine while the study runs.

The responses on the serving workstation are the study: no replica, and a lost
disk takes every participant with it. The mirror is the second copy, and
it has to be the kind of thing that cannot break a page a participant is
halfway through.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

import streamlit as st  # noqa: E402

from behavior_graphs.viz.streamlit.user_studies import backup  # noqa: E402
from behavior_graphs.viz.streamlit.user_studies.session import build_record  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    st.session_state.clear()
    monkeypatch.delenv(backup.BACKUP_BUCKET_ENV, raising=False)


class _Recorder:
    def __init__(self, fail: bool = False):
        self.saved: list[dict] = []
        self.fail = fail

    def save(self, record: dict) -> str:
        if self.fail:
            raise RuntimeError("bucket unreachable")
        self.saved.append(record)
        return "blob"


def _use(monkeypatch, store):
    monkeypatch.setenv(backup.BACKUP_BUCKET_ENV, "some-bucket")
    monkeypatch.setattr(backup, "_backup_store", lambda bucket: store)
    return store


RECORD = {"session_id": "abc", "status": "in_progress", "step": 2}


def test_it_does_nothing_when_no_bucket_is_configured():
    """The default. A laptop and a test run want no network at all."""
    assert backup.backup_bucket() == ""
    assert backup.mirror(RECORD, once_key="k") is False


def test_it_sends_the_record_when_a_bucket_is_configured(monkeypatch):
    store = _use(monkeypatch, _Recorder())

    assert backup.mirror(RECORD, once_key="k") is True
    assert store.saved == [RECORD]


def test_it_sends_once_per_occasion(monkeypatch):
    """Called on every rerun of the page it fires on, and an intermission is
    rerun by every widget on it."""
    store = _use(monkeypatch, _Recorder())

    backup.mirror(RECORD, once_key="k")
    backup.mirror(RECORD, once_key="k")
    backup.mirror(RECORD, once_key="k")

    assert len(store.saved) == 1


def test_a_later_occasion_sends_again(monkeypatch):
    store = _use(monkeypatch, _Recorder())

    backup.mirror(RECORD, once_key="intermission_2")
    backup.mirror(RECORD, once_key="intermission_4")

    assert len(store.saved) == 2


def test_a_bucket_that_fails_does_not_raise(monkeypatch):
    """A study page is not where somebody finds out a credential expired."""
    _use(monkeypatch, _Recorder(fail=True))

    assert backup.mirror(RECORD, once_key="k") is False


def test_a_failure_is_retried_at_the_next_occasion(monkeypatch):
    """Marked done only after it worked. Otherwise one flaky moment silences
    the mirror for the rest of the participant's session."""
    failing = _use(monkeypatch, _Recorder(fail=True))
    backup.mirror(RECORD, once_key="k")

    failing.fail = False
    assert backup.mirror(RECORD, once_key="k") is True


def test_a_record_with_no_session_is_not_sent(monkeypatch):
    store = _use(monkeypatch, _Recorder())

    assert backup.mirror({"status": "in_progress"}, once_key="k") is False
    assert store.saved == []


def test_the_copy_is_the_same_record_the_local_store_holds(monkeypatch):
    """Assembled by `build_record`, which `checkpoint` and `submit` also use, so
    the bucket cannot drift into holding a different shape."""
    store = _use(monkeypatch, _Recorder())
    payload = {"participant_id": "p9", "step": 2}

    backup.mirror(build_record("abc", payload, "in_progress"), once_key="k")

    sent = store.saved[0]
    assert sent["session_id"] == "abc"
    assert sent["status"] == "in_progress"
    assert sent["participant_id"] == "p9"
    assert "updated_at" in sent


def test_it_lands_under_its_own_prefix():
    """Not mixed in with `survey_responses/`, which is where the primary GCS
    backend writes when a bucket is the only store."""
    import inspect

    from behavior_graphs.viz.streamlit.user_study.response_store import GCSResponseStore

    default = inspect.signature(GCSResponseStore.__init__).parameters["prefix"].default
    assert backup.BACKUP_PREFIX != default


# ── Which pages it fires on ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, False), (1, False), (2, True), (3, False), (4, True), (5, False), (6, False)],
)
def test_the_intermissions_are_the_even_steps_between_the_ends(step, expected):
    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
        _is_intermission,
    )

    assert _is_intermission(step, 3) is expected


def test_submitting_mirrors_too():
    """The last intermission is before stage 3, so the answers that only exist
    at the end would otherwise never leave the box."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

    assert "mirrored_submit" in inspect.getsource(r._render_submit)


# ── A finished response is not written back to an unfinished one ──────────────


def test_a_checkpoint_does_not_overwrite_a_submitted_response(monkeypatch, tmp_path):
    """The app hands out a second live session on one id by design: the intro
    tells a participant they can open the link again, and doing that in a second
    tab leaves the first live. A submit in one and any answer change in the other
    used to replace the completed response with an in-progress one, which the
    analytics then hides because unfinished sessions are filtered out."""
    from behavior_graphs.viz.streamlit.user_studies import session as s

    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)
    st.session_state.clear()

    s.submit("tabA", "shared", {"participant_id": "p9", "answers": "final"})
    wrote = s.checkpoint("tabB", "shared", {"participant_id": "p9", "answers": "stale"})

    assert wrote is False
    from behavior_graphs.viz.streamlit.user_study.response_store import get_store

    stored = get_store().get("shared")
    assert stored["status"] == "submitted"
    assert stored["answers"] == "final"


def test_a_checkpoint_still_writes_while_the_attempt_is_unfinished(monkeypatch, tmp_path):
    from behavior_graphs.viz.streamlit.user_studies import session as s
    from behavior_graphs.viz.streamlit.user_study.response_store import get_store

    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)
    st.session_state.clear()

    assert s.checkpoint("t", "live", {"step": 1}) is True
    assert s.checkpoint("t", "live", {"step": 2}) is True
    assert get_store().get("live")["step"] == 2


def test_a_failed_submit_upload_is_retried_when_they_come_back(monkeypatch, tmp_path):
    """The attempt inside the submit page gets at most one extra run: once
    `_submitted` is set that page renders a line and returns, and a refresh
    starts a new websocket that has lost the record. Returning to the link is
    the occasion that outlives it."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

    resolve = inspect.getsource(r._resolve_session)
    assert 'mirror(record, once_key=f"{pfx}_mirrored_submit")' in resolve
    # And only for a finished attempt: an unfinished one is mirrored by its
    # intermissions, not by the resume gate.
    assert resolve.index("== SUBMITTED") < resolve.index("mirror(record")
