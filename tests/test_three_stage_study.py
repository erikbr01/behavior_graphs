"""The three-stage study: three tasks, one display each, the last one chosen.

What is worth pinning here is the design rather than the rendering: that the
task order and the display assignment are actually drawn, that the last stage's
display is asked for rather than assigned, that the progress bar does not name
that display before the participant has picked it, and that the response records
enough of all three for an analysis to tell two participants apart.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.step_names import (
    UNCHOSEN,
    three_stage_step_names,
)
from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
    QUESTION_SETS,
)
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import (
    render as _render_module,
)
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
    MODALITIES,
    _exit_step,
    _stage_modalities,
    _stage_step,
    _submit_step,
    draw_assignment,
)


@pytest.fixture(autouse=True)
def _unstub_render_module():
    """Put back what ``_STRANDED_SCRIPT`` replaces.

    ``AppTest`` runs the script in this process, so its assignment to
    ``r._render_stage`` lands on the real module and stays there. Anything later
    that reads it saw the stub instead: the two fragment tests below passed
    alone and failed in the suite, depending on ordering. Same hazard and same
    fix as ``tests/test_study_session_resume.py``.
    """
    render_stage = _render_module._render_stage
    yield
    _render_module._render_stage = render_stage


VARIANT_ID = "three_stage_v1"


def test_the_variant_is_the_three_tasks_in_one_group():
    """One group, because the draw is what distinguishes two participants.

    The contrastive layouts counterbalance by group (A sees videos first, B the
    graph); here the order is drawn per session, so a second group would be a
    second name for the same condition.
    """
    variant = get_variant(VARIANT_ID)

    assert [(stage.task, stage.session_name, stage.question_set) for stage in variant.stages] == [
        ("set_table", "set_table_contrastive", "set_table"),
        ("prepare_tray", "prepare_tray_contrastive", "prepare_tray"),
        ("cooking", "cooking_contrastive", "cooking"),
    ]
    assert [(group.id, group.order) for group in variant.groups] == [("randomized", [])]


def test_every_stage_names_a_task_a_session_and_a_question_bank_that_exist():
    """A stage whose session or question bank is a typo renders an empty page."""
    import yaml

    from behavior_graphs.viz.streamlit.user_study.sessions import sessions_dir

    for stage in get_variant(VARIANT_ID).stages:
        assert stage.question_set in QUESTION_SETS, stage.task
        session_file = sessions_dir() / f"{stage.session_name}.yaml"
        assert session_file.is_file(), stage.session_name
        # The questions ask what the robot got wrong, which is unanswerable
        # without being told what it was supposed to do.
        assert yaml.safe_load(session_file.read_text()).get("task_description"), stage.task
        assert stage.label and stage.label != stage.task, stage.task


def test_the_display_settings_hide_what_the_study_hides():
    """Same stimulus as the contrastive layout, or the two are not comparable."""
    params = get_variant(VARIANT_ID).graph_params

    assert params.get("show_behavior_annotations") is False
    assert params.get("show_camera_labels") is False
    assert float(params.get("video_lead_seconds", 0)) > 0
    assert params.get("shuffle_rollout_indices") is True


def test_the_draw_shuffles_the_tasks_and_splits_the_two_displays():
    """Both displays appear once across the assigned stages, in a drawn order."""
    orders, first_displays, button_orders = set(), set(), set()
    for seed in range(200):
        drawn = draw_assignment(3, random.Random(seed))
        button_orders.add(tuple(drawn["choice_order"]))
        assert sorted(drawn["stage_tasks"]) == [0, 1, 2]
        assigned = drawn["modalities"][:-1]
        assert sorted(assigned) == sorted(MODALITIES)
        # The last stage is the participant's, so the draw leaves it open.
        assert drawn["modalities"][-1] is None
        orders.add(tuple(drawn["stage_tasks"]))
        first_displays.add(assigned[0])

    assert len(orders) == 6, "every task order should be reachable"
    assert first_displays == set(MODALITIES), "either display should be able to go first"
    # The modal's two buttons too: the stage exists to measure which display
    # someone reaches for, so a display that is always the left button would put
    # a primacy bias straight onto the result.
    assert button_orders == {("videos", "graph"), ("graph", "videos")}


def test_the_chosen_display_lands_in_the_last_stage():
    drawn = {"stage_tasks": [0, 1, 2], "modalities": ["videos", "graph", None]}

    assert _stage_modalities(drawn, None) == ["videos", "graph", None]
    assert _stage_modalities(drawn, "graph") == ["videos", "graph", "graph"]


def test_the_steps_alternate_stages_and_intermissions():
    """The step indices and the step names have to describe the same flow.

    The durations are keyed by index and named from this list, so an off-by-one
    between them relabels every step of every response.
    """
    names = three_stage_step_names(["videos", "graph", None])

    assert _stage_step(1) == 1 and _stage_step(2) == 3 and _stage_step(3) == 5
    # Submit lives on the closing page, so that is the last step there is.
    assert _exit_step(3) == 6 and _submit_step(3) == _exit_step(3)
    assert len(names) == _exit_step(3) + 1
    assert names[0] == "Introduction"
    assert names[_stage_step(2) - 1] == "Intermission"
    assert names[_exit_step(3)] == "Closing questions"
    assert names[-1] == "Closing questions"


def test_the_progress_bar_does_not_name_a_display_nobody_has_picked_yet():
    """The bar is drawn above the modal, so naming a display would pre-empt it."""
    before = three_stage_step_names(["videos", "graph", None])
    after = three_stage_step_names(["videos", "graph", "graph"])

    assert before[-2] == f"Task 3 ({UNCHOSEN})"
    assert "behavior graph" not in before[-3]
    # Once picked, the bar says which — and that it was theirs to pick.
    assert after[-2] == f"Task 3 (behavior graph · {UNCHOSEN})"


def test_the_steps_number_the_tasks_and_never_name_them():
    """A participant reads "Task 2", never "Prepare the tray".

    The tasks come in a drawn order, so a name says which task someone drew and
    nothing about how far through they are — and it makes a timing chart's
    x-axis a category of one participant.
    """
    names = three_stage_step_names(["videos", "graph", "graph"])

    assert [n for n in names if n.startswith("Task ")] == [
        "Task 1 (rollout videos)",
        "Task 2 (behavior graph)",
        f"Task 3 (behavior graph · {UNCHOSEN})",
    ]
    said = " ".join(names).lower()
    assert not any(word in said for word in ("table", "tray", "cook"))
    assert three_stage_step_names([]) == []


# ── Choosing the last stage's display ─────────────────────────────────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

#: The intermission before the last stage: what is done, and the two buttons.
#: It was a modal on the stage page, on the grounds that the choice had to be
#: made before any evidence was on screen. This page has no evidence on it
#: either, and it costs neither a dialog a participant can dismiss into a dead
#: end nor a page underneath whose only job is to reopen it.
_CHOICE_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
    _render_intermission,
)
from behavior_graphs.viz.streamlit.user_study.survey_steps import record_step_entry

record_step_entry(4, "t_step")
st.session_state.setdefault("t_step", 4)
_render_intermission(
    "t",
    4,
    "t_step",
    done_number=2,
    choice_order=st.session_state.get("order", ["videos", "graph"]),
)
"""


def _choices(at) -> list[str]:
    return [button.key for button in at.button if button.key.startswith("t_choose_")]


def test_the_choice_page_offers_both_displays_and_nothing_else():
    """No Continue beside them: picking one *is* continuing, and a third control
    would let somebody reach the last stage with no display chosen."""
    at = AppTest.from_string(_CHOICE_SCRIPT, default_timeout=30).run()

    assert not at.exception
    assert _choices(at) == ["t_choose_videos", "t_choose_graph"]
    assert [button.key for button in at.button] == _choices(at)


def test_the_choice_is_made_before_any_evidence_is_on_screen():
    """A participant who has already glimpsed the graph behind the question is
    not choosing between two displays any more. The page is an intermission, so
    neither evidence view is rendered — both name themselves in a header."""
    at = AppTest.from_string(_CHOICE_SCRIPT, default_timeout=30).run()

    assert [header.value for header in at.header] == ["Intermission"]


def test_it_still_says_the_previous_task_is_done_and_saved():
    at = AppTest.from_string(_CHOICE_SCRIPT, default_timeout=30).run()
    said = " ".join(markdown.value for markdown in at.markdown)

    assert "Task 2 is complete" in said
    assert "saved" in said


def test_picking_a_display_records_it_and_moves_on():
    at = AppTest.from_string(_CHOICE_SCRIPT, default_timeout=30).run()

    at = at.button(key="t_choose_graph").click().run()

    assert at.session_state["t_chosen_modality"] == "graph"
    assert at.session_state["t_step"] == 5


def test_the_buttons_are_drawn_in_the_order_they_were_given():
    """Which display sits on the left is drawn per participant: a fixed left
    button would put a primacy bias straight onto the result this exists for."""
    at = AppTest.from_string(_CHOICE_SCRIPT, default_timeout=30)
    at.session_state["order"] = ["graph", "videos"]
    at = at.run()

    assert _choices(at) == ["t_choose_graph", "t_choose_videos"]


def test_the_choice_is_timed_on_its_own_page():
    """Otherwise stage 3's duration is deliberation plus reading, and the two
    stages with no choice in front of them are not comparable with it. The
    timing comes off the intermission's clock, and the stage's own starts when
    it is entered."""
    at = AppTest.from_string(_CHOICE_SCRIPT, default_timeout=30).run()

    at = at.button(key="t_choose_videos").click().run()

    assert at.session_state["t_choice_seconds"] >= 0
    assert 5 not in at.session_state["t_step_times"], "the stage clock has not started yet"


_STRANDED_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

# The stage page would need a graph and 60 rollouts; the point is that it is
# never reached, so a loud stand-in is enough to catch it if it is.
r._render_stage = lambda **kwargs: st.error("the stage rendered with no display")

variant = get_variant("three_stage_v1")
r._render_step(
    pfx="t",
    step_key="t_step",
    step=5,
    variant=variant,
    group=variant.groups[0],
    participant="p9",
    session_id="s1",
    ordered=list(variant.stages),
    modalities=["videos", "graph", None],
    n_stages=3,
    choice_order=["videos", "graph"],
    checkpoint_now=lambda: None,
)
"""


def test_a_stage_reached_with_no_display_asks_again_instead_of_rendering():
    """Reachable by a stored session whose step got past the choice without one
    — a resumed record, or a step edited in the URL. A stage rendered with no
    display is not a page, so the choice is put back in front of them."""
    at = AppTest.from_string(_STRANDED_SCRIPT, default_timeout=30).run()

    assert not at.exception
    assert not at.error, [error.value for error in at.error]
    assert _choices(at) == ["t_choose_videos", "t_choose_graph"]


# ── Loading the three tasks ───────────────────────────────────────────────────

_INTRO_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

# The sessions themselves are not under test here: what is, is whether a stage
# that cannot load stops the study on the page where stopping is free.
r._stage_state = lambda pfx, stage, params: (
    None if stage.task == st.session_state.get("broken") else {"prefix": "x"}
)

variant = get_variant("three_stage_v1")
r.render_three_stage(variant, variant.groups[0])
"""


def test_a_task_that_cannot_load_stops_the_study_at_the_introduction():
    """Not three pages later, with two stages' answers already given.

    The stage pages are the only place the answers live until the submit page
    writes them, and a stage that cannot render has no Continue button — so a
    stage 2 discovered broken on arrival dead-ends a participant and loses
    everything they had done. Every stage is therefore loaded up front.
    """
    at = AppTest.from_string(_INTRO_SCRIPT, default_timeout=30)
    at.session_state["broken"] = "cooking"
    at = at.run()

    assert not at.exception
    assert any("could not be loaded" in error.value for error in at.error)
    assert not [button for button in at.button if button.key.endswith("intro_continue")]


def test_all_three_loading_lets_the_study_start():
    at = AppTest.from_string(_INTRO_SCRIPT, default_timeout=30).run()

    assert not at.exception
    assert not at.error
    assert [button.key for button in at.button] == ["ts_randomized_intro_continue"]


# ── Coming back to a half-answered page ───────────────────────────────────────

_RESTORE_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
    render_task_questions,
    restore_task_answers,
)

restore_task_answers(
    key_prefix="q",
    question_set="prepare_tray",
    responses=st.session_state["stored"],
    tlx=st.session_state["stored_tlx"],
)
_responses, _tlx, _choice, complete = render_task_questions(key_prefix="q", question_set="prepare_tray")
st.session_state["given_back"] = (_responses, _tlx, complete)
"""

_STORED = {
    "failure_mode_ranking": [
        "The robot manages to slide out the tray, but fails to pick up the bowl.",
        "The robot fails to slide out the tray.",
        "The robot manages to slide out the tray, pick up the bowl, but fails to place the bowl on the tray.",
    ],
    "data_collection_choice": "Bowl in the middle of the shelf, fork on top of the shelf",
}
_STORED_TLX = {"temporal_demand": 60, "effort": 20, "confidence": 40}


def test_a_checkpointed_answer_comes_back_into_the_widgets_that_asked_for_it():
    """Resuming has to return the page as they left it, not an empty one — and
    the ranks have to come back in the order they were given, since the order is
    the answer."""
    at = AppTest.from_string(_RESTORE_SCRIPT, default_timeout=30)
    at.session_state["stored"] = _STORED
    at.session_state["stored_tlx"] = _STORED_TLX
    at = at.run()

    assert not at.exception
    responses, tlx, complete = at.session_state["given_back"]
    assert responses == _STORED
    assert tlx == _STORED_TLX
    # Restored means answered: the Continue button is not disabled on a page the
    # participant had already finished.
    assert complete


def test_an_answer_the_bank_no_longer_offers_is_dropped_rather_than_restored():
    """After an edit to the question bank, a stored answer to the old wording is
    not among the options — and Streamlit raises on a value that is not."""
    at = AppTest.from_string(_RESTORE_SCRIPT, default_timeout=30)
    at.session_state["stored"] = {
        "failure_mode_ranking": ["The robot fails at something we no longer ask about."],
        "data_collection_choice": "An arrangement nobody offers",
    }
    at.session_state["stored_tlx"] = {"effort": 55}  # not one of the scale's points

    at = at.run()

    assert not at.exception
    responses, tlx, complete = at.session_state["given_back"]
    assert responses["failure_mode_ranking"] == [None, None, None]
    assert responses["data_collection_choice"] is None
    assert tlx == {"temporal_demand": None, "effort": None, "confidence": None}
    assert not complete


# ── The response ──────────────────────────────────────────────────────────────

_SUBMIT_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
    _render_submit,
    _stage_modalities,
)

variant = get_variant("three_stage_v1")
group = variant.groups[0]
pfx = f"ts_{group.id}"
assignment = st.session_state[f"{pfx}_assignment"]
ordered = [variant.stages[i] for i in assignment["stage_tasks"]]
modalities = _stage_modalities(assignment, st.session_state.get(f"{pfx}_chosen_modality"))
_render_submit(
    variant=variant,
    group=group,
    pfx=pfx,
    step_key=f"{pfx}_step",
    step=7,
    participant="p9",
    session_id="testsession01",
    ordered=ordered,
    modalities=modalities,
    n_stages=len(ordered),
    choice_order=["graph", "videos"],
)
"""

_PFX = "ts_randomized"


def _submitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run the submit page over a seeded session and return what it stored."""
    monkeypatch.setenv("STUDY_RESULTS_DIR", str(tmp_path))
    monkeypatch.delenv("SURVEY_GCS_BUCKET", raising=False)

    at = AppTest.from_string(_SUBMIT_SCRIPT, default_timeout=30)
    at.session_state[f"{_PFX}_assignment"] = {
        "stage_tasks": [2, 0, 1],
        "modalities": ["graph", "videos", None],
    }
    at.session_state[f"{_PFX}_chosen_modality"] = "videos"
    at.session_state[f"{_PFX}_choice_seconds"] = 12.5
    for task in ("cooking", "set_table", "prepare_tray"):
        at.session_state[f"{_PFX}_{task}_responses"] = {
            "failure_mode_ranking": [f"{task} first", f"{task} second"],
            "data_collection_choice": f"{task} option",
        }
        at.session_state[f"{_PFX}_{task}_tlx_responses"] = {"effort": 40, "temporal_demand": 60}
        at.session_state[f"{_PFX}_{task}_choice"] = f"{task} option"
    at.session_state[f"{_PFX}_exit_answers"] = {
        "experience": "Some",
        "remarks": "the graph was quicker to skim",
    }

    at = at.run()
    assert not at.exception
    at = at.button(key=f"{_PFX}_submit").click().run()
    assert not at.exception

    from behavior_graphs.viz.streamlit.user_study.response_store import LocalResponseStore

    stored = LocalResponseStore(tmp_path).list_all()
    assert len(stored) == 1
    return stored[0]


def test_the_response_records_the_draw_the_answers_and_the_choice(tmp_path, monkeypatch):
    """Without the draw, a response says which task was answered but not what
    any of it was a comparison against."""
    saved = _submitted(tmp_path, monkeypatch)

    assert saved["study_variant"] == VARIANT_ID
    assert saved["session_id"] == "testsession01"
    assert saved["status"] == "submitted"
    assert saved["task_order"] == ["cooking", "set_table", "prepare_tray"]
    assert saved["stage_modalities"] == {
        "cooking": "graph",
        "set_table": "videos",
        "prepare_tray": "videos",
    }
    assert saved["modality_assignment"] == {
        "cooking": "assigned",
        "set_table": "assigned",
        "prepare_tray": "chosen",
    }
    # The stage-3 answer, on its own key as well as inside the maps above.
    assert saved["chosen_modality"] == "videos"
    assert saved["evidence_order"] == ["graph", "videos", "videos"]
    # And what the choice was made against: which button was where, and how long
    # it took. A pick made in two seconds off the left button is weaker evidence
    # than one made in twenty.
    assert saved["choice_button_order"] == ["graph", "videos"]
    assert saved["modality_choice_seconds"] == 12.5


def test_the_answers_are_keyed_by_task_not_by_display(tmp_path, monkeypatch):
    """A display is not what a set of answers is about here, and two
    participants read the same task through different ones."""
    saved = _submitted(tmp_path, monkeypatch)

    assert set(saved["understanding_responses"]) == {"cooking", "set_table", "prepare_tray"}
    assert saved["understanding_responses"]["cooking"]["data_collection_choice"] == "cooking option"
    assert saved["nasa_tlx"]["set_table"] == {"effort": 40, "temporal_demand": 60}
    assert saved["data_collection_priorities"]["prepare_tray"] == "prepare_tray option"
    assert saved["robot_data_collection_experience"] == "Some"
    assert saved["remarks"] == "the graph was quicker to skim"
    assert saved["sessions"]["cooking"] == "cooking_contrastive"


def test_the_stored_step_names_stay_comparable_across_participants(tmp_path, monkeypatch):
    """The timing chart names its x-axis from these, so a name carrying this
    participant's own task order would be a category of one."""
    saved = _submitted(tmp_path, monkeypatch)

    assert saved["step_names"] == three_stage_step_names(["graph", "videos", "videos"])
    assert not any("Cook" in name for name in saved["step_names"])
    assert len(saved["step_names"]) == _submit_step(3) + 1


# ── The intermission ──────────────────────────────────────────────────────────

_INTERMISSION_SCRIPT = """
import streamlit as st

from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
    _render_intermission,
)

_render_intermission("t", 2, "t_step", done_number=1)
"""


def _intermission() -> tuple[str, list[str]]:
    at = AppTest.from_string(_INTERMISSION_SCRIPT, default_timeout=30).run()
    assert not at.exception
    return (
        " ".join(markdown.value for markdown in at.markdown),
        [header.value for header in at.header],
    )


def test_the_intermission_says_what_is_done_and_that_it_is_recorded():
    said, headers = _intermission()

    assert headers == ["Intermission"]
    assert "Task 1 is complete" in said
    assert "saved" in said


def test_the_intermission_says_nothing_about_what_comes_next():
    """It used to announce the next task and the display it would be read in,
    which told a participant nothing they were not about to be shown — and in
    the last intermission it pre-announced the choice that is this layout's
    result, on the page immediately before the modal that collects it.
    """
    said, _ = _intermission()

    assert "Task 2" not in said
    assert "Next" not in said
    for display in ("rollout videos", "behavior graph", "you choose"):
        assert display not in said.lower()


# ── Submitting ────────────────────────────────────────────────────────────────


def test_submit_is_on_the_closing_page_and_not_a_page_of_its_own():
    """The page it used to have was headed "Review & Submit" and reviewed
    nothing: this study has no way back, so its only content was the button
    that left it."""
    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
        _exit_step,
        _submit_step,
    )

    for n_stages in (2, 3, 4):
        assert _submit_step(n_stages) == _exit_step(n_stages)


def test_a_stored_step_past_the_end_resumes_on_the_last_page():
    """An attempt saved while Submit was still its own step names a step this
    layout no longer has, and resuming it would render nothing at all."""
    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

    variant = get_variant("three_stage_v1")
    at = AppTest.from_string(
        """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

variant = get_variant("three_stage_v1")
r._rehydrate("t", "t_step", st.session_state["record"], list(variant.stages))
""",
        default_timeout=30,
    )
    at.session_state["record"] = {"step": 7, "assignment": None}
    at = at.run()

    assert not at.exception
    assert at.session_state["t_step"] == r._exit_step(len(variant.stages))


# ── Framing the two displays evenly ───────────────────────────────────────────
#
# The last stage asks which display a participant reaches for when nobody
# assigns one, which is the result this layout exists to produce. Copy that
# presents one of the two as the study's own instrument answers that question
# before it is asked.


def test_neither_display_gets_the_active_verb():
    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
        _PHASE_IMPERATIVE,
    )

    verbs = {label.split()[0] for label in _PHASE_IMPERATIVE.values()}
    assert len(verbs) == 1, f"one verb for both, got {_PHASE_IMPERATIVE}"


def test_each_display_gets_a_clause_about_what_it_lets_you_do():
    """The videos used to get a definition and the graph a capability."""
    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage.render import (
        _PHASE_CLAUSE,
    )

    for modality, clause in _PHASE_CLAUSE.items():
        assert "which" in clause, f"{modality} is described, not defined"
    shorter, longer = sorted(len(c) for c in _PHASE_CLAUSE.values())
    assert longer / shorter < 1.6, "one display is given noticeably more of a case"


def test_the_pair_is_named_in_the_order_the_participant_drew():
    """Always naming the videos first is a fixed order under a drawn one."""
    said = {}
    for order in (["videos", "graph"], ["graph", "videos"]):
        at = AppTest.from_string(
            """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

variant = get_variant("three_stage_v1")
r._render_intro(list(variant.stages), st.session_state["order"])
""",
            default_timeout=30,
        )
        at.session_state["order"] = order
        at = at.run()
        assert not at.exception
        text = " ".join(m.value for m in at.markdown)
        said[order[0]] = text.index("rollout videos") < text.index("behavior graph")

    assert said["videos"] is True and said["graph"] is False


def test_the_introduction_calls_neither_of_them_interactive():
    at = AppTest.from_string(
        """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

variant = get_variant("three_stage_v1")
r._render_intro(list(variant.stages), ["videos", "graph"])
""",
        default_timeout=30,
    ).run()

    assert "interactive" not in " ".join(m.value for m in at.markdown)


def test_the_introduction_says_who_to_contact():
    """A participant who hits trouble mid-study should not have to finish it to
    find an address. The already-completed page has one; the first page did not.
    """
    from behavior_graphs.viz.streamlit.user_studies.resume import CONTACT_EMAIL

    at = AppTest.from_string(
        """
import streamlit as st
from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

variant = get_variant("three_stage_v1")
r._render_intro(list(variant.stages), ["videos", "graph"])
""",
        default_timeout=30,
    ).run()

    assert not at.exception
    assert CONTACT_EMAIL in " ".join(markdown.value for markdown in at.markdown)


# ── Answering does not redraw the evidence, and is still saved ────────────────


def test_the_questions_are_a_fragment():
    """Without one, every radio was a full app rerun that re-rendered the whole
    page: 250–730ms on a video stage, which reads as the page reloading under
    the pointer. Nothing was re-fetching; it was the repaint."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

    source = inspect.getsource(r._render_stage)
    assert "@st.fragment" in source
    # The Continue button has to be inside it: it is gated on `complete`, and a
    # fragment cannot update a control outside itself.
    assert source.index("@st.fragment") < source.index('"Continue →"')


def test_the_fragment_checkpoints_for_itself():
    """The parent's checkpoint runs at the end of `render_three_stage`, which a
    fragment rerun never reaches. Without this an answer would only be written
    when the participant left the page, and a participant who abandons mid-stage
    is exactly the case the save-as-you-go promise exists for."""
    import inspect

    from behavior_graphs.viz.streamlit.user_studies.variants.three_stage import render as r

    stage = inspect.getsource(r._render_stage)
    assert "checkpoint_now()" in stage
    # Written before the way out is offered, not after.
    assert stage.index("checkpoint_now()") < stage.index('"Continue →"')

    parent = inspect.getsource(r.render_three_stage)
    assert "def _checkpoint_now()" in parent
    assert "checkpoint_now=_checkpoint_now" in parent
