"""Round trip for the user-study response store.

The store is what Survey Analytics reads, so a regression here is invisible
until someone opens the page after a study session.

One JSON file per attempt, replaced in place. It used to be a single
append-only JSONL with a line per revision; nothing read the history, and a
lookup had to scan backwards for one attempt among twelve hundred lines.
"""

from __future__ import annotations

import json

from behavior_graphs.viz.streamlit.user_study.response_store import (
    LocalResponseStore,
    get_store,
    results_dir,
)


def _response(pid: str, group: str, effort: int) -> dict:
    return {
        "participant_id": pid,
        "group": group,
        "timestamp": f"20260811_0900{effort:02d}",
        "nasa_tlx": {"effort": effort},
    }


def test_saved_responses_read_back(tmp_path):
    store = LocalResponseStore(tmp_path)
    store.save(_response("p01", "A", 4))
    store.save(_response("p02", "B", 7))

    rows = store.list_all()
    assert [r["participant_id"] for r in rows] == ["p01", "p02"]
    assert [r["nasa_tlx"]["effort"] for r in rows] == [4, 7]
    # Every record carries the id save() handed back, which is its filename.
    assert all(r["_response_id"].endswith(".json") for r in rows)
    assert {p.name for p in tmp_path.glob("*.json")} == {r["_response_id"] for r in rows}


def test_list_all_is_empty_before_anything_is_saved(tmp_path):
    assert LocalResponseStore(tmp_path).list_all() == []


def test_an_unreadable_file_does_not_lose_the_others(tmp_path):
    store = LocalResponseStore(tmp_path)
    store.save(_response("p01", "A", 4))
    (tmp_path / "session_broken.json").write_text("{not json")
    store.save(_response("p02", "B", 7))

    assert sorted(r["participant_id"] for r in store.list_all()) == ["p01", "p02"]


def test_an_unreadable_attempt_is_not_reported_as_absent(tmp_path):
    """`get` returning None makes the app start a second attempt over the top of
    the first. A file that will not parse is not an attempt that is not there,
    so it is logged rather than swallowed."""
    import logging

    store = LocalResponseStore(tmp_path)
    (tmp_path / "session_abc.json").write_text("{not json")

    logger = logging.getLogger("behavior_graphs.viz.streamlit.user_study.response_store")
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger.addHandler(handler)
    try:
        assert store.get("abc") is None
    finally:
        logger.removeHandler(handler)

    assert records, "an unreadable record has to say so"


def test_a_checkpoint_replaces_the_attempt_rather_than_piling_up(tmp_path):
    """The three-stage study checkpoints as answers come in, so one attempt is
    written many times. Ten checkpoints from ten participants must not read back
    as a hundred responses, and must not leave a hundred files."""
    store = LocalResponseStore(tmp_path)
    for effort in (1, 2, 3):
        store.save({**_response("p01", "A", effort), "session_id": "abc", "step": effort})
    store.save({**_response("p02", "B", 9), "session_id": "def", "step": 1})
    store.save(_response("p03", "A", 5))  # a one-shot response from an older layout

    rows = store.list_all()

    assert sorted(r["participant_id"] for r in rows) == ["p01", "p02", "p03"]
    latest = next(r for r in rows if r["participant_id"] == "p01")
    assert latest["nasa_tlx"]["effort"] == 3, "the last write wins"
    assert len(list(tmp_path.glob("*.json"))) == 3, "one file per attempt, not per answer"


def test_one_attempt_can_be_fetched_by_its_session_id(tmp_path):
    """What the resume gate reads before offering to continue a session, and
    what a checkpoint reads before writing. Opened by name: no scan."""
    store = LocalResponseStore(tmp_path)
    store.save({**_response("p01", "A", 1), "session_id": "abc", "step": 1})
    store.save({**_response("p01", "A", 2), "session_id": "abc", "step": 4})

    assert store.get("abc")["step"] == 4
    assert store.get("nosuchsession") is None
    assert store.get("") is None


def test_a_write_is_never_seen_half_finished(tmp_path):
    """Written to a temporary name and renamed, which is atomic on POSIX. An
    append could leave a torn line for the reader to skip."""
    store = LocalResponseStore(tmp_path)
    store.save({"session_id": "abc", "step": 1})
    store.save({"session_id": "abc", "step": 2})

    # No leftovers, and nothing half-written for `list_all` to trip on.
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["session_abc.json"]
    assert json.loads((tmp_path / "session_abc.json").read_text())["step"] == 2


def test_a_session_keyed_record_is_addressed_by_its_session(tmp_path):
    """One file per attempt, so a checkpoint replaces the last rather than
    piling up beside it — the same shape as the GCS backend."""
    from behavior_graphs.viz.streamlit.user_study.response_store import _response_filename

    assert _response_filename({"session_id": "abc123"}) == "session_abc123.json"
    # A record from a layout that saves once keeps the old participant name.
    assert _response_filename(_response("p01", "A", 4)).startswith("group_a_")


def test_a_session_id_cannot_escape_the_results_directory(tmp_path):
    """The id comes off the query string, and it names a file."""
    store = LocalResponseStore(tmp_path / "results")
    store.save({"session_id": "../../escaped", "step": 1})

    assert not (tmp_path / "escaped.json").exists()
    assert list((tmp_path / "results").glob("*.json"))


def test_get_store_uses_the_results_dir_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)
    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path / "results"))

    assert results_dir() == tmp_path / "results"
    store = get_store()
    assert isinstance(store, LocalResponseStore)
    store.save(_response("p01", "A", 4))
    assert list((tmp_path / "results").glob("*.json"))
