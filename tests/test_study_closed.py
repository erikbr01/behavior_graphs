"""The participant route, once the study has closed.

The thing to get right is not the wording — it is that nothing else runs. A
visit to an old recruitment link must not start an attempt, write a record, or
offer a way in; and it must not be asked for a password first, since a closed
study has nothing to let anyone into.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.user_studies import closed  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "apps" / "study" / "Home.py"


@pytest.fixture(autouse=True)
def _participant_route(monkeypatch, tmp_path):
    """A deployment as the live one is configured, minus the closure flag."""
    monkeypatch.delenv(closed.ACCEPTING_ENV, raising=False)
    monkeypatch.delenv("APP_PASSWORD_SHA256", raising=False)
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)
    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path / "responses"))
    monkeypatch.setenv("STUDY_VARIANT", "three_stage_v1")


def _run() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_the_default_is_closed():
    """Not a flag a deployment has to remember: the container on the
    participant URL is built from this repo and restarted by hand, so a closure
    that needed configuration would still be open on any container nobody got
    to. Every spelling of "no" reads as closed, including a typo."""
    assert closed.accepting_responses({}) is False
    assert closed.accepting_responses({closed.ACCEPTING_ENV: ""}) is False
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "0"}) is False
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "false"}) is False
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "no"}) is False
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "tru"}) is False


def test_it_opens_only_when_something_says_so_on_purpose():
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "1"}) is True
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "true"}) is True
    assert closed.accepting_responses({closed.ACCEPTING_ENV: " YES "}) is True
    assert closed.accepting_responses({closed.ACCEPTING_ENV: "on"}) is True


def test_a_recruitment_link_lands_on_the_notice():
    at = _run()

    assert not at.exception
    assert [title.value for title in at.title] == ["This study has closed"]
    said = " ".join(block.value for block in at.markdown)
    assert "no longer accepting responses" in said
    assert "nothing to fill in" in said


def test_the_notice_says_the_raffle_still_reaches_them():
    """The one thing a participant might otherwise act on: they entered a
    raffle that has not been drawn. It needs nothing from them, and saying so
    is what stops the email asking."""
    at = _run()

    said = " ".join(block.value for block in at.markdown)
    assert "raffle will be drawn soon" in said
    assert "contacted by email" in said
    assert "Nothing further is needed from you" in said


def test_the_notice_says_who_to_ask():
    at = _run()

    said = " ".join(block.value for block in at.markdown)
    assert f"[{closed.CONTACT_EMAIL}](mailto:{closed.CONTACT_EMAIL})" in said
    assert closed.CONTACT_EMAIL == "erbauer@stanford.edu"
    # Once. The link's text is the address, so it is readable and selectable
    # whether or not the click opens anything -- printing it again beside
    # itself, which this page did at first, only reads as a mistake.
    assert said.count(closed.CONTACT_EMAIL) == 2  # the label and the mailto target
    assert f"({closed.CONTACT_EMAIL})" not in said


def test_the_study_itself_is_not_rendered_behind_the_notice():
    """`st.stop()`, not a notice drawn above the introduction: a page that
    still holds the study invites a reader to scroll past the notice, and the
    layout would write its record on the way."""
    at = _run()

    assert not at.button  # no "Start", no "Next"
    assert not at.radio and not at.text_input
    assert len(at.title) == 1
    # And nothing under the three things a participant needs: what becomes of
    # the responses is the record's business, not theirs.
    assert not at.caption


def test_a_visit_writes_no_record(tmp_path):
    """The gate is ahead of `mark_participant_route`, so a closed visit leaves
    nothing at all -- not an abandoned attempt, which is what a record written
    on the first page load would look like a month from now."""
    responses = tmp_path / "responses"

    _run()

    assert not responses.exists() or list(responses.iterdir()) == []


def test_a_closed_study_does_not_ask_for_a_password_first(monkeypatch):
    """Ahead of `password_gate` on purpose: a participant who cannot get in
    should be told the study is over, not asked to prove they belong to it."""
    # sha256("hunter2"), so the gate would be armed if it ran.
    monkeypatch.setenv(
        "APP_PASSWORD_SHA256",
        "f52fbd32b2b3b86ff88ef6c490628285f482af15ddcb29541f94bcf526a3f6c7",
    )

    at = _run()

    assert [title.value for title in at.title] == ["This study has closed"]
    assert not at.text_input  # no password box


def test_a_variant_that_no_longer_runs_still_says_the_same_thing(monkeypatch):
    """The gate is ahead of the variant resolution, so a deployment pinned to a
    study that has been retired shows the notice rather than a configuration
    error aimed at whoever deployed it."""
    monkeypatch.setenv("STUDY_VARIANT", "no_such_variant_v9")

    at = _run()

    assert [title.value for title in at.title] == ["This study has closed"]
    assert not at.error


def test_the_flag_reopens_the_route(monkeypatch):
    """Reopening has to be possible -- for a dry run, or for a second round --
    and this is the proof that the notice is a gate and not a replacement: with
    the flag set, the app gets far enough to resolve the variant, which is
    where an unknown one is reported."""
    monkeypatch.setenv(closed.ACCEPTING_ENV, "1")
    monkeypatch.setenv("STUDY_VARIANT", "no_such_variant_v9")

    at = _run()

    assert not at.title or at.title[0].value != "This study has closed"
    assert any("no_such_variant_v9" in error.value for error in at.error)
