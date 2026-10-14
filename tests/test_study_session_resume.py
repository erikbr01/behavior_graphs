"""Sessions that outlive the browser: minting, checkpointing, resuming.

A three-task study runs for half an hour, and Streamlit's own session dies with
its websocket — a refresh, a slept laptop or a redeploy. So the answers are
written as they come in, against a session id the participant carries in their
URL, and coming back to that URL offers to resume.

These tests drive the real renderer with its pages stubbed out: what is under
test is the session machinery around them, and the pages need a GPU-built
behavior graph and 60 rollouts of video that this suite has no business loading.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# The module ``_SCRIPT`` stubs, imported here so the stubs can be undone — see
# the ``_unstub_render_module`` fixture.
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import (  # noqa: E402
    render as r,
)

PFX = "ts_randomized"

#: Renders the study with every page replaced by a marker, so a run exercises
#: session resolution, the draw, the checkpoint and the resume gate only.
_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

r._stage_state = lambda pfx, stage, params: {"prefix": f"{pfx}_{stage.task}"}


def _page(**kwargs):
    # Nothing on it but the way out, which is the real advance_step.
    st.write("page", kwargs["step"])
    if st.button("next", key="stub_next"):
        from behavior_graphs.viz.streamlit.user_study.survey_steps import advance_step

        advance_step(kwargs["step"], kwargs["step_key"])


r._render_step = _page

variant = get_variant("three_stage_v1")
r.render_three_stage(variant, variant.groups[0])
"""


@pytest.fixture(autouse=True)
def _unstub_render_module():
    """Put back what ``_SCRIPT`` replaces.

    ``AppTest`` runs the script in this process, so its two assignments land on
    the real module and stay there: any later test that renders the three-stage
    study — in this file or another one — got the stub pages instead of the
    study. ``tests/test_three_stage_study.py`` and ``tests/test_study_app_route.py``
    both failed that way when they happened to run after this file, which under
    ``--dist loadfile`` depends on which worker picks them up.
    """
    stage_state, render_step = r._stage_state, r._render_step
    yield
    r._stage_state, r._render_step = stage_state, render_step


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)

    def _records() -> list[dict]:
        from behavior_graphs.viz.streamlit.user_study.response_store import LocalResponseStore

        return LocalResponseStore(tmp_path).list_all()

    return _records


def _param(at: AppTest, name: str) -> str:
    """One query parameter, as the app sees it.

    ``AppTest.query_params`` hands back the raw list per name where the app's own
    ``st.query_params`` collapses it to the last value.
    """
    value = at.query_params[name]
    return str(value[-1]) if isinstance(value, list) else str(value)


def _run(session_id: str | None = None, pid: str | None = None) -> AppTest:
    at = AppTest.from_string(_SCRIPT, default_timeout=60)
    if session_id:
        at.query_params["s"] = session_id
    if pid:
        at.query_params["pid"] = pid
    at = at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_a_first_visit_mints_a_session_and_writes_it_before_anything_is_answered(store):
    """The record exists from the first page, so there is always something to
    resume — and the id is in the URL, so the browser carries the identity."""
    at = _run(pid="p42")

    session_id = at.session_state[f"{PFX}_session_id"]
    assert session_id and _param(at, "s") == session_id
    (record,) = store()
    assert record["session_id"] == session_id
    assert record["status"] == "in_progress"
    assert record["step"] == 0
    assert record["participant_id"] == "p42"
    # The draw is in the record from the start: a resumed session has to show
    # the same tasks in the same order.
    assert sorted(record["assignment"]["stage_tasks"]) == [0, 1, 2]


def test_a_record_written_outside_the_participant_app_says_it_is_a_lab_one(store):
    """This script renders the layout the way the lab app does, without the
    participant app's marker. Records written that way are previews: real ones
    only come from ``apps/study/Home.py`` (see ``tests/test_study_app_route.py``),
    and the analytics hides previews rather than averaging them in."""
    _run(pid="p42")

    (record,) = store()
    assert record["origin"] == "lab"


def test_the_participant_id_comes_from_the_link_not_from_a_text_box(store):
    """A sidebar box defaulting to "anonymous" is a box nobody fills in, and
    then hundreds of responses cannot be told apart or paid."""
    at = _run(pid="p42")
    assert not at.text_input, "no id to type when the link carried one"

    without = _run()
    assert [box.label for box in without.text_input] == ["Participant ID (pilot)"]


def test_a_rerun_that_changed_no_answer_writes_nothing_new(store):
    """Checkpointing on every rerun would write while someone reads a graph."""
    at = _run()
    before = len(store())

    at.run()

    assert len(store()) == before


def test_returning_to_the_link_offers_to_resume_or_start_over(store):
    at = _run()
    session_id = at.session_state[f"{PFX}_session_id"]
    at.session_state[f"{PFX}_step"] = 3
    at.run()

    back = _run(session_id=session_id)

    assert [header.value for header in back.header] == ["Welcome back"]
    assert [button.key for button in back.button] == [f"{PFX}_resume", f"{PFX}_restart"]
    # It says where they got to: "resume your session" alone does not tell
    # someone whether resuming is worth it. By number, like every other page.
    said = " ".join(markdown.value for markdown in back.markdown)
    assert "Task 2" in said
    # And it has not started the study underneath the question.
    assert "page" not in said


def test_resuming_comes_back_to_the_same_page_with_the_same_draw(store):
    at = _run()
    session_id = at.session_state[f"{PFX}_session_id"]
    drawn = at.session_state[f"{PFX}_assignment"]
    at.session_state[f"{PFX}_step"] = 3
    at.run()

    resumed = _run(session_id=session_id).button(key=f"{PFX}_resume").click().run()

    assert not resumed.exception
    assert resumed.session_state[f"{PFX}_step"] == 3
    assert resumed.session_state[f"{PFX}_assignment"] == drawn
    assert resumed.session_state[f"{PFX}_session_id"] == session_id


def test_starting_over_keeps_the_old_record_and_draws_again(store):
    """Study data is never deleted — a session someone walked away from is still
    evidence about where they walked away."""
    at = _run()
    first = at.session_state[f"{PFX}_session_id"]
    at.session_state[f"{PFX}_step"] = 3
    at.run()

    restarted = _run(session_id=first).button(key=f"{PFX}_restart").click().run()

    assert not restarted.exception
    second = restarted.session_state[f"{PFX}_session_id"]
    assert second != first
    by_session = {record["session_id"]: record["status"] for record in store()}
    assert by_session[first] == "abandoned"
    assert by_session[second] == "in_progress"
    assert _param(restarted, "s") == second


def test_a_finished_session_is_not_resumable(store):
    """Otherwise a participant's second pass overwrites their submission."""
    at = _run()
    session_id = at.session_state[f"{PFX}_session_id"]
    from behavior_graphs.viz.streamlit.user_study.response_store import LocalResponseStore

    results = LocalResponseStore(__import__("os").environ["STUDY_RESULTS_DIR"])
    results.save({**results.get(session_id), "status": "submitted"})

    back = _run(session_id=session_id)

    assert [header.value for header in back.header] == ["You have already completed this study"]
    # No way to resume or restart *this* attempt. The one control is the escape
    # for a different person on a shared browser, which mints its own session.
    assert [button.key for button in back.button] == [f"{PFX}_not_me"]


def test_an_unknown_session_id_starts_a_fresh_attempt_rather_than_erroring(store):
    """A mangled link is not a dead end."""
    at = _run(session_id="nosuchsession")

    assert not at.exception
    assert at.session_state[f"{PFX}_session_id"] == "nosuchsession"
    assert store()[-1]["status"] == "in_progress"


def test_time_spent_is_checkpointed_as_it_accrues_and_survives_a_resume(store):
    """A dropout's timings are as much data as their answers, and a resumed
    session must not restart the clock on the pages it already did."""
    at = _run()
    session_id = at.session_state[f"{PFX}_session_id"]

    at = at.button(key="stub_next").click().run()

    assert not at.exception
    latest = store()[-1]
    assert "0" in {str(k) for k in latest["step_durations_seconds"]}, latest[
        "step_durations_seconds"
    ]

    resumed = _run(session_id=session_id).button(key=f"{PFX}_resume").click().run()

    assert not resumed.exception
    carried = resumed.session_state[f"{PFX}_step_times"]
    assert 0 in carried and "end" in carried[0], "the introduction's time came back"


# ── Coming back after the tab was closed ──────────────────────────────────────
#
# The URL is gone at that point, so the cookie is the only handle left. What the
# resolver does with a recalled id is the interesting part: it is worth using
# only when it still points at an attempt worth showing somebody.


def _recalled(monkeypatch, cookie: str, record: dict | None) -> str:
    monkeypatch.setattr(r, "remembered_session", lambda variant, who: cookie)
    monkeypatch.setattr(r, "participant_id", lambda: "p9")
    monkeypatch.setattr(r, "load", lambda session_id: record)
    return r._recalled_session("three_stage_v1")


def test_an_unfinished_attempt_is_recalled_so_the_gate_can_offer_it(monkeypatch):
    assert _recalled(monkeypatch, "abc123", {"status": "in_progress"}) == "abc123"


def test_a_finished_attempt_is_recalled_so_they_are_told_they_finished(monkeypatch):
    """Better than silently handing them a second blank study to fill in."""
    assert _recalled(monkeypatch, "abc123", {"status": "submitted"}) == "abc123"


def test_an_abandoned_attempt_is_not_recalled(monkeypatch):
    """They pressed "Start over" on it. Recalling it would offer to resume the
    very thing they asked to leave — or, since it is no longer resumable, tell
    them they had already completed the study."""
    assert _recalled(monkeypatch, "abc123", {"status": "abandoned"}) == ""


def test_an_id_with_no_record_behind_it_is_not_recalled(monkeypatch):
    """A stale cookie against an emptied store. A clean start beats a gate
    reporting on an attempt that does not exist."""
    assert _recalled(monkeypatch, "abc123", None) == ""


def test_a_browser_that_remembered_nothing_recalls_nothing(monkeypatch):
    assert _recalled(monkeypatch, "", {"status": "in_progress"}) == ""


def test_starting_over_leaves_a_session_the_cookie_cannot_drag_them_out_of(store):
    """The cookie still points at the attempt they just abandoned, so the run
    after "Start over" has to arrive holding a new id rather than none."""
    at = _run()
    first = at.session_state[f"{PFX}_session_id"]
    at.session_state[f"{PFX}_step"] = 3
    at.run()

    back = _run(session_id=first)
    back.button(key=f"{PFX}_restart").click().run()

    second = back.session_state[f"{PFX}_session_id"]
    assert second and second != first
    # And the abandoned one is kept, not deleted: study data is never thrown away.
    statuses = {r["session_id"]: r["status"] for r in store()}
    assert statuses[first] == "abandoned"


def test_a_participant_who_already_finished_is_told_who_to_write_to():
    """The page tells them not to start again, so it has to say what to do
    instead — otherwise the only option it leaves them is the one it forbids."""
    from behavior_graphs.viz.streamlit.user_studies.resume import CONTACT_EMAIL

    at = AppTest.from_string(
        """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.resume import render_already_submitted

render_already_submitted(pfx="t")
""",
        default_timeout=30,
    ).run()

    assert not at.exception
    said = " ".join(markdown.value for markdown in at.markdown)
    assert CONTACT_EMAIL in said
    assert "@" in CONTACT_EMAIL and "." in CONTACT_EMAIL.split("@")[-1]


def test_a_finished_attempt_offers_a_way_out_for_the_next_person(store):
    """This page is reached from the browser cookie as well as the URL, and a
    bare recruitment link carries no participant id, so everyone on one browser
    shares a single bucket for seven days. Without a way forward, the next
    person at a shared machine is told they finished a study they never saw."""
    at = _run()
    session_id = at.session_state[f"{PFX}_session_id"]
    at.session_state[f"{PFX}_step"] = 3
    at.run()

    # Finish it, then come back as somebody else on the same browser.
    from behavior_graphs.viz.streamlit.user_studies.session import submit

    submit(PFX, session_id, {"participant_id": "", "step": 7})
    back = _run(session_id=session_id)

    assert [header.value for header in back.header] == ["You have already completed this study"]
    assert f"{PFX}_not_me" in [button.key for button in back.button]

    back.button(key=f"{PFX}_not_me").click().run()
    fresh = back.session_state[f"{PFX}_session_id"]
    assert fresh and fresh != session_id, "the next person gets their own attempt"
    statuses = {r["session_id"]: r["status"] for r in store()}
    assert statuses[session_id] == "submitted", "the finished attempt is untouched"
