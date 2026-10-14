"""The question block a participant answers once per display method.

These checks used to be spelled as substring searches over
``inspect.getsource(render_task_questions)`` -- three of them, in two files.
That form passes whether or not the function works, fails on any rename, and
said nothing about what a participant sees. Rendering the block costs about a
second, so it is rendered.
"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (  # noqa: E402
    _TLX_ITEMS,
    QUESTION_SETS,
)

_SCRIPT = """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
    render_task_questions,
)

st.session_state["out"] = render_task_questions(key_prefix="t", question_set={qset!r})
"""


def _run(question_set: str = "set_table") -> AppTest:
    """The block on its own, with no curated clips, so option 2 is a radio."""
    return AppTest.from_string(_SCRIPT.format(qset=question_set), default_timeout=60).run()


def _complete(at: AppTest) -> bool:
    return bool(at.session_state["out"][3])


def _answer_everything_but_the_task_load(at: AppTest, question_set: str) -> AppTest:
    spec = QUESTION_SETS[question_set]
    for i in range(spec["n_ranked"]):
        at = at.selectbox(key=f"t_failure_rank_{i + 1}").set_value(spec["failure_modes"][i]).run()
    return at.radio(key="t_collection_choice").set_value(spec["collection_options"][0]).run()


@pytest.mark.parametrize("question_set", sorted(QUESTION_SETS))
def test_the_prompt_counts_the_ranks_in_words_and_draws_that_many(question_set):
    """The number in the sentence has to match the dropdowns underneath it."""
    at = _run(question_set)
    n_ranked = QUESTION_SETS[question_set]["n_ranked"]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    assert not at.exception
    ask = next(m.value for m in at.markdown if "Rank the top" in m.value)
    assert f"Rank the top {words[n_ranked]} failure modes" in ask
    assert len(at.selectbox) == n_ranked


def test_the_definition_of_a_failure_mode_is_in_the_question_that_uses_it():
    """Not a paragraph above it, which a participant reads before it means
    anything. One markdown call, so the two arrive together."""
    at = _run()

    ask = next(m.value for m in at.markdown if "Rank the top" in m.value)
    assert "A failure mode is one specific way" in ask


def test_an_untouched_block_is_not_a_complete_answer():
    at = _run()

    assert not _complete(at)
    assert [w.value for w in at.warning] == ["Please answer all questions."]


def test_the_block_will_not_continue_on_the_ranks_and_the_choice_alone():
    """The task-load ratings are part of the answer now that they can be left
    blank: an untouched slider used to report 50, and the page had no way to
    tell that apart from a chosen 50."""
    at = _answer_everything_but_the_task_load(_run(), "set_table")

    assert not _complete(at), "unrated task load must not count as answered"
    assert at.session_state["out"][1] == {
        "temporal_demand": None,
        "effort": None,
        "confidence": None,
    }


def test_rating_the_task_load_is_what_completes_it():
    at = _answer_everything_but_the_task_load(_run(), "set_table")

    at = at.radio(key="t_nasa_tlx_temporal_demand").set_value(20).run()
    assert not _complete(at), "one of the three is not all of them"
    at = at.radio(key="t_nasa_tlx_effort").set_value(80).run()
    assert not _complete(at), "confidence is asked, so it is answered too"
    at = at.radio(key="t_nasa_tlx_confidence").set_value(60).run()

    assert _complete(at)
    assert not at.warning
    assert at.session_state["out"][1] == {
        "temporal_demand": 20,
        "effort": 80,
        "confidence": 60,
    }


def test_the_answers_come_back_under_the_names_the_store_reads():
    at = _answer_everything_but_the_task_load(_run(), "set_table")
    spec = QUESTION_SETS["set_table"]
    responses, _, choice, _ = at.session_state["out"]

    assert responses["failure_mode_ranking"] == spec["failure_modes"][: spec["n_ranked"]]
    assert responses["data_collection_choice"] == spec["collection_options"][0]
    assert choice == spec["collection_options"][0]


def test_a_rank_cannot_be_spent_twice():
    """Ranking the same failure mode first and second is not an ordering."""
    at = _run("prepare_tray")
    first = QUESTION_SETS["prepare_tray"]["failure_modes"][0]
    at = at.selectbox(key="t_failure_rank_1").set_value(first).run()

    assert first not in list(at.selectbox(key="t_failure_rank_2").options)


# ── The card grid ─────────────────────────────────────────────────────────────
#
# The path a participant actually takes: with curated clips present, option 2 is
# a grid of selectable cards rather than a plain radio. It had no coverage,
# which is how an `st.rerun()` that Streamlit ignores outright survived in it.

_CARDS_SCRIPT = """
import streamlit as st
from pathlib import Path
from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
    render_task_questions,
)

st.session_state["out"] = render_task_questions(
    key_prefix="t",
    question_set="set_table",
    demo_videos_dir=Path(st.session_state["clips"]),
)
"""


@pytest.fixture()
def clips(tmp_path):
    """A stand-in for the curated DROID clips, one folder per milk position."""
    for position in ("right", "middle", "left"):
        folder = tmp_path / f"prepare_plate_milk_{position}"
        folder.mkdir()
        (folder / "example.mp4").write_bytes(b"")
    return tmp_path


def _cards(clips) -> AppTest:
    at = AppTest.from_string(_CARDS_SCRIPT, default_timeout=60)
    at.session_state["clips"] = str(clips)
    return at.run()


def test_the_grid_is_the_component_not_streamlit_radios(clips):
    """The per-card radios were the sticky part: three one-option widgets, so
    deselecting the card you left was a server decision. The component keeps the
    layout and does it in the browser."""
    at = _cards(clips)

    assert not at.exception
    # Task load only — no radio per card any more.
    assert len(at.radio) == len(_TLX_ITEMS)


def test_a_pick_from_the_component_is_what_comes_back(monkeypatch):
    """The component reports by setting its value, which reruns the app, so the
    answer arrives on the next run like any widget's."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    monkeypatch.setattr(collection_cards, "_cards", lambda **kw: "Cola left")
    views = [{"option": "Cola left", "src": "u", "poster": ""}]

    assert collection_cards.render_collection_cards(views, key="k", selected=None) == "Cola left"


def test_a_rerun_the_component_did_not_cause_keeps_the_answer(monkeypatch):
    """It reports nothing unless it was clicked, and `None` from an untouched
    component must not read as "they cleared it"."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    monkeypatch.setattr(collection_cards, "_cards", lambda **kw: None)
    views = [{"option": "Cola left", "src": "u", "poster": ""}]

    assert collection_cards.render_collection_cards(views, key="k", selected="Cola left") == (
        "Cola left"
    )


def test_options_with_no_servable_clip_are_dropped(monkeypatch, tmp_path):
    """A card is its clip. One with an empty <video> is a choice a participant
    cannot make on the evidence the question is about."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    monkeypatch.setattr(
        collection_cards, "video_src", lambda clip, **kw: None if "bad" in clip.name else "u"
    )
    monkeypatch.setattr(collection_cards, "poster_src", lambda clip, **kw: None)

    views = collection_cards.card_views(
        [("good", tmp_path / "good.mp4"), ("gone", tmp_path / "bad.mp4")], key="k"
    )

    assert [v["option"] for v in views] == ["good"]


def test_an_empty_grid_leaves_the_answer_alone(clips):
    """No clips mounted at all: the block falls back to a plain radio, and the
    component must not clear what that recorded."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    assert collection_cards.render_collection_cards([], key="k", selected="kept") == "kept"


def test_the_component_ships_its_frontend(clips):
    """`declare_component(path=...)` serves this directory; without the file the
    grid is a blank iframe and the question cannot be answered at all."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    index = collection_cards._COMPONENT_DIR / "index.html"
    assert index.is_file()
    html = index.read_text()
    # The three messages the protocol is made of.
    for message in (
        "streamlit:componentReady",
        "streamlit:setComponentValue",
        "streamlit:setFrameHeight",
    ):
        assert message in html
    # And the one line the whole component exists for.
    assert "input.name = GROUP" in html


def test_the_clips_are_passed_as_urls_not_paths(monkeypatch, tmp_path):
    """`st.video(path)` re-reads and re-hashes the file on every rerun; these
    clips are ~7.5MB each."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    seen = {}

    def _src(clip, **kwargs):
        seen["coordinates"] = kwargs.get("coordinates")
        return "https://example/clip.mp4"

    monkeypatch.setattr(collection_cards, "video_src", _src)
    monkeypatch.setattr(collection_cards, "poster_src", lambda clip, **kw: None)

    views = collection_cards.card_views([("only", tmp_path / "a.mp4")], key="kp")

    assert views[0]["src"].startswith("https://")
    assert "kp" in seen["coordinates"]


def test_the_frontend_seeds_from_python_but_does_not_follow_it(clips):
    """Streamlit sends the render args for the same run in which it reads the
    component's value, so the render that follows a click still carries the
    *previous* selection. A frontend that adopted it put the highlight back on
    the card the participant had just left — "it jumps to the one I clicked
    before". While the frame is mounted the selection lives in it; Python seeds
    it on mount and when the clips change, which is one read of `args.selected`
    and no more.
    """
    from behavior_graphs.viz.streamlit.components import collection_cards

    html = (collection_cards._COMPONENT_DIR / "index.html").read_text()
    body = html[html.index("function render(args)") :]
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("//"))
    seeds_at = code.index("args.selected")
    rebuild_ends_at = code.index("return;")

    assert seeds_at < rebuild_ends_at, "seeded outside the rebuild branch"
    assert "args.selected" not in code[rebuild_ends_at:], (
        "reading it after the rebuild branch is the stale echo coming back"
    )


def test_clips_that_cannot_be_served_fall_back_to_asking_without_them(monkeypatch, clips):
    """Clips on disk that the media layer declines (a deleted file, a manager
    that refuses) used to leave question 2 as a prompt with no options and a
    Continue disabled forever, with nothing on screen to explain it."""
    from behavior_graphs.viz.streamlit.components import collection_cards

    monkeypatch.setattr(collection_cards, "video_src", lambda clip, **kw: None)
    at = _cards(clips)
    spec = QUESTION_SETS["set_table"]

    assert not at.exception
    choice = at.radio(key="t_collection_choice")
    assert list(choice.options) == list(spec["collection_options"])

    at = choice.set_value(spec["collection_options"][0]).run()
    assert at.session_state["out"][2] == spec["collection_options"][0]
