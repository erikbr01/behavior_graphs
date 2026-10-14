"""Three tasks in a row, one display each, and the last display is the
participant's own pick.

Where the contrastive layout shows one task through both displays, this one
shows three tasks through one display apiece:

  Introduction
  Stage 1  — a task, shown as videos *or* as the behavior graph, + its questions
  Intermission
  Stage 2  — a second task, shown as whichever display stage 1 was not, + questions
  Intermission
  Stage 3  — the third task, shown as whichever display the participant picks in
             a modal, + questions
  Closing questions
  Submit

Three things are drawn per participant rather than fixed by group: which task
goes in which stage (a permutation of the three), which of the two displays
stage 1 gets (stage 2 gets the other), and which side of the modal each display
sits on. Stage 3's display is neither drawn nor assigned — the modal asks, and
the answer is recorded, because *which display someone reaches for when nobody
assigns one* is its own result, which is also why its buttons are not always in
the same order.

Task order and display assignment therefore live in the response, not in the
group id: there is one group here, and the draw is what distinguishes two
participants in it. Everything else — the evidence pages, the per-task question
banks, the closing questions, the response store — is shared with the
contrastive layout, so the two studies' answers are directly comparable.

Answers are written as they are given rather than at the end. Three tasks is
half an hour, and a layout that saved only on submit lost everything to a
refresh or a slept laptop; instead every change is checkpointed onto a
server-side record addressed by an id in the participant's URL, so an abandoned
session still leaves the stages that were finished, and a returning participant
is offered the choice of resuming it (see
:mod:`behavior_graphs.viz.streamlit.user_studies.session`).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import streamlit as st

from behavior_graphs.viz.streamlit.data import resolve_data_path
from behavior_graphs.viz.streamlit.user_studies.backup import mirror
from behavior_graphs.viz.streamlit.user_studies.browser_session import (
    remember,
    remembered_session,
)
from behavior_graphs.viz.streamlit.user_studies.evidence import (
    render_phase_view,
    render_task_restatement,
)
from behavior_graphs.viz.streamlit.user_studies.exit_questions import (
    exit_answers,
    render_exit_questions,
)
from behavior_graphs.viz.streamlit.user_studies.resume import (
    CONTACT_EMAIL,
    RESTART,
    RESUME,
    render_already_submitted,
    render_resume_gate,
)
from behavior_graphs.viz.streamlit.user_studies.session import (
    IN_PROGRESS,
    SESSION_PARAM,
    SUBMITTED,
    abandon,
    adopt_session,
    build_record,
    checkpoint,
    is_resumable,
    load,
    origin,
    participant_id,
    participant_route,
    prolific_fields,
    start_session,
    submit,
    url_session_id,
)
from behavior_graphs.viz.streamlit.user_studies.step_names import three_stage_step_names
from behavior_graphs.viz.streamlit.user_studies.types import (
    StudyGroupSpec,
    StudyStageSpec,
    StudyVariantSpec,
)
from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
    COUNT_WORDS,
    render_task_questions,
    restore_task_answers,
)
from behavior_graphs.viz.streamlit.user_study.sessions import (
    load_session_artifacts,
    seed_common_state,
    sessions_dir,
)
from behavior_graphs.viz.streamlit.user_study.strategies import load_study_config
from behavior_graphs.viz.streamlit.user_study.survey_steps import (
    advance_step,
    get_step_durations,
    record_step_entry,
    record_step_exit,
    render_progress_bar,
    seed_step_durations,
    step_elapsed,
)

#: The two displays, and what to call each of them in a sentence.
MODALITIES = ("videos", "graph")
_PHASE_DEFINITE = {"videos": "the rollout videos", "graph": "the behavior graph"}
# One verb for both. "Watch" against "Explore" made one display sound passive
# and the other capable, which is a thumb on the scale of the very question the
# last stage asks.
_PHASE_IMPERATIVE = {
    "videos": "Review the rollout videos",
    "graph": "Review the behavior graph",
}

#: Each display named as a noun, for the sentence that introduces the pair.
_PHASE_NOUN = {"videos": "**rollout videos**", "graph": "a **behavior graph**"}

#: One clause each, saying what the display lets you do. Two earlier versions
#: leaned: first the videos got a definition ("the individual attempts, as
#: clips") while the graph got a capability, which reads as one of them being
#: the tool and the other the raw material; then the videos "show each attempt
#: in full", which implies the graph shows something less than that. It does
#: not — clicking a node plays the clips behind it — so the difference is how
#: the attempts are arranged, not how much of them you get.
#:
#: So: one verb, the same grammatical shape, and one further capability each,
#: with outcome named on both sides rather than only on the videos'.
_PHASE_CLAUSE = {
    "videos": (
        "the **rollout videos**, which show the attempts one by one and can be "
        "narrowed to the successes or the failures"
    ),
    "graph": (
        "the **behavior graph**, which shows the attempts together as the steps "
        "the robot took and where they branched, with each branch's outcome"
    ),
}

#: How a stage got its display: drawn by the study, or picked by the participant.
ASSIGNED = "assigned"
CHOSEN = "chosen"

_STEP_INTRO = 0


def _stage_step(stage_number: int) -> int:
    """Step index of stage *stage_number* (1-based); an intermission follows each."""
    return 2 * stage_number - 1


def _exit_step(n_stages: int) -> int:
    return 2 * n_stages


def _is_intermission(step: int, n_stages: int) -> bool:
    """Whether *step* is one of the pages between two stages.

    Stages sit on the odd steps and their intermissions on the even ones, so
    this is the even steps between the introduction and the closing page.
    """
    return step > _STEP_INTRO and step % 2 == 0 and step < _exit_step(n_stages)


def _submit_step(n_stages: int) -> int:
    """The same page as the closing questions, which is where Submit lives.

    It used to be a step of its own, headed "Review & Submit". It reviewed
    nothing — there is no way back through this study — so it was a page whose
    only content was the button that leaves it.
    """
    return _exit_step(n_stages)


# ── The draw ──────────────────────────────────────────────────────────────────


def draw_assignment(n_stages: int, rng: random.Random) -> dict:
    """A participant's task order and the display each stage is read in.

    ``stage_tasks`` is a permutation of the stage indices in the variant config,
    and ``modalities`` is one display per stage: the two displays in a drawn
    order for the first two stages, then ``None`` for the last, which the
    participant picks for themselves. A pure function of *rng* so it can be
    tested without a browser; the caller draws once and stores the result.
    """
    stage_tasks = rng.sample(range(n_stages), n_stages)
    displays = list(MODALITIES)
    rng.shuffle(displays)
    # The last stage is the choice, so it is not part of the assignment. Any
    # earlier stage beyond the two displays repeats them in the drawn order,
    # which only matters if a config ever lists more than three tasks.
    assigned = [displays[i % len(displays)] for i in range(n_stages - 1)]
    return {
        "stage_tasks": stage_tasks,
        "modalities": [*assigned, None],
        # Which side of the modal each display's button is on, drawn too. The
        # last stage's whole result is which display someone reaches for when
        # nobody assigns one, and a fixed left button would put a primacy bias
        # straight onto it — the one confound the rest of this draw exists to
        # keep out.
        "choice_order": rng.sample(list(MODALITIES), len(MODALITIES)),
    }


def _assignment(pfx: str, n_stages: int) -> dict:
    """The draw for this session, made once and kept for the rest of it."""
    key = f"{pfx}_assignment"
    if key not in st.session_state:
        st.session_state[key] = draw_assignment(n_stages, random.Random())
    return st.session_state[key]


def _stage_modalities(assignment: dict, chosen: str | None) -> list[str | None]:
    """The display per stage, with the participant's pick filled into the last."""
    modalities = list(assignment["modalities"])
    if modalities:
        modalities[-1] = chosen
    return modalities


def _choice_order(assignment: dict) -> list[str]:
    """The order to draw the modal's two buttons in.

    Falls back to the declared order for a draw made before the order was part
    of it — a session that is already underway when this is deployed keeps
    working rather than raising on the stage it has reached.
    """
    order = assignment.get("choice_order")
    if not isinstance(order, list) or sorted(order) != sorted(MODALITIES):
        return list(MODALITIES)
    return order


# ── Loading one stage's task ──────────────────────────────────────────────────


def _stage_state(pfx: str, stage: StudyStageSpec, variant_graph_params: dict) -> dict | None:
    """Rollouts, graph and copy for one stage, loaded once per session.

    Each stage is a different session yaml, so the artifacts are keyed by task
    rather than by variant: three tasks' indices, labels and graphs coexist in
    one participant's session state, and a stage is only loaded when they first
    reach it. ``None`` when the session could not be loaded — the caller reports
    it rather than rendering half a page.
    """
    spfx = f"{pfx}_{stage.task}"
    if st.session_state.get(f"{spfx}_loaded_session") != stage.session_name:
        sess, config_path, run, mp4_dir, index, errors = load_session_artifacts(
            sessions_dir() / f"{stage.session_name}.yaml",
            data_path=resolve_data_path(),
        )
        if errors or run is None:
            for error in errors:
                st.sidebar.error(f"{stage.label}: {error}")
            return None
        if sess.get("shuffle", False):
            random.shuffle(index["episodes"])
        seed_common_state(spfx, sess, load_study_config(config_path), mp4_dir, index)
        st.session_state[f"{spfx}_labels"] = run.labels
        st.session_state[f"{spfx}_metadata"] = run.metadata
        st.session_state[f"{spfx}_graph"] = run.graph
        st.session_state[f"{spfx}_run_dir"] = str(run.run_dir)
        # The session yaml's own `graph:` block on top of the variant's, so a
        # per-task setting (prepare_tray's shallower `min_branch`) stays with the
        # task and is not restated once per study that shows it. A stage entry in
        # the variant yaml overrides both.
        st.session_state[f"{spfx}_graph_params"] = {
            **variant_graph_params,
            **sess.get("graph", {}),
            **stage.graph_params,
        }
        st.session_state[f"{spfx}_loaded_session"] = stage.session_name

    return {
        "prefix": spfx,
        "index": st.session_state[f"{spfx}_index"],
        "mp4_dir": Path(st.session_state[f"{spfx}_mp4_dir"]),
        "labels": st.session_state[f"{spfx}_labels"],
        "metadata": st.session_state[f"{spfx}_metadata"],
        "run_dir": st.session_state[f"{spfx}_run_dir"],
        "task_description": st.session_state.get(f"{spfx}_task_description", ""),
        "hide_episode_index": bool(st.session_state.get(f"{spfx}_hide_episode_index", False)),
        "graph_params": st.session_state[f"{spfx}_graph_params"],
        "demo_videos_dir": st.session_state.get(f"{spfx}_demo_videos_dir", ""),
    }


# ── The session behind the URL ────────────────────────────────────────────────


def _clear_session_state(pfx: str) -> None:
    """Forget everything about this attempt, for a participant starting over.

    Includes the loaded rollouts and graphs, which are cheap to fetch again
    (they come from a shared cache) and expensive to get wrong: a draw from the
    old attempt left behind here would put the new one's stages in the old
    one's order.
    """
    for key in [k for k in list(st.session_state.keys()) if str(k).startswith(f"{pfx}_")]:
        del st.session_state[key]


def _progress_label(record: dict) -> str:
    """Where a stored attempt got to, in words a participant can act on.

    "Resume your session" alone does not tell someone whether resuming is worth
    it; "Task 2 (rollout videos)" does. Rebuilt from the record rather than read
    from its stored ``step_names``, which an attempt saved before the names
    changed would carry in the old wording.
    """
    tasks = [str(task) for task in record.get("task_order") or []]
    if tasks:
        by_task = record.get("stage_modalities") or {}
        names = three_stage_step_names([by_task.get(task) for task in tasks])
        step = record.get("step")
        if names and isinstance(step, int) and 0 <= step < len(names):
            return str(names[step])
    stored = record.get("step_names")
    step = record.get("step")
    if isinstance(stored, list) and isinstance(step, int) and 0 <= step < len(stored):
        return str(stored[step])
    return "the beginning"


def _rehydrate(pfx: str, step_key: str, record: dict, stages: list[StudyStageSpec]) -> None:
    """Restore a stored attempt into this Streamlit session.

    Both halves of it: the plain keys the submit page reads, and the widget
    values behind them, so the page a participant left comes back filled in
    rather than blank. This runs before any question widget is created —
    Streamlit refuses to set a widget's state once it exists.
    """
    assignment = record.get("assignment")
    if isinstance(assignment, dict) and assignment.get("stage_tasks"):
        st.session_state[f"{pfx}_assignment"] = assignment
    st.session_state[f"{pfx}_chosen_modality"] = record.get("chosen_modality")
    st.session_state[f"{pfx}_choice_seconds"] = record.get("modality_choice_seconds")
    st.session_state[f"{pfx}_timestamp"] = record.get("timestamp")

    understanding = record.get("understanding_responses") or {}
    tlx = record.get("nasa_tlx") or {}
    priorities = record.get("data_collection_priorities") or {}
    by_task = {stage.task: stage for stage in stages}
    for task, answers in understanding.items():
        st.session_state[f"{pfx}_{task}_responses"] = answers
    for task, scores in tlx.items():
        st.session_state[f"{pfx}_{task}_tlx_responses"] = scores
    for task, choice in priorities.items():
        st.session_state[f"{pfx}_{task}_choice"] = choice
    for task, stage in by_task.items():
        if stage.question_set:
            restore_task_answers(
                key_prefix=f"{pfx}_{task}_questions",
                question_set=stage.question_set,
                responses=understanding.get(task),
                tlx=tlx.get(task),
            )

    # The closing page, through the same copy its own submit page reads.
    experience = record.get("robot_data_collection_experience")
    remarks = str(record.get("remarks") or "")
    st.session_state[f"{pfx}_exit_answers"] = {"experience": experience, "remarks": remarks}
    if experience is not None:
        st.session_state[f"{pfx}_experience"] = experience
    if remarks:
        st.session_state[f"{pfx}_remarks"] = remarks

    seed_step_durations(step_key, record.get("step_durations_seconds") or {})
    step = record.get("step")
    # Clamped: an attempt stored before Submit moved onto the closing page names
    # a step past the end of the layout, and resuming it would render nothing.
    st.session_state[step_key] = (
        min(int(step), _exit_step(len(stages))) if isinstance(step, int) else _STEP_INTRO
    )


def _resolve_session(
    pfx: str, step_key: str, stages: list[StudyStageSpec], *, variant_id: str
) -> str | None:
    """This browser's session id, or ``None`` when a gate is on screen instead.

    Five cases, in the order they are checked: a rerun of a session we are
    already serving; a bare link, where the browser is asked whether it kept an
    id from a closed tab; a first visit, which mints an id and publishes it to
    the URL; a return to a link whose record is finished, which says so; and a
    return to a link whose record is not, which asks whether to resume it.
    """
    local = str(st.session_state.get(f"{pfx}_session_id") or "")
    from_url = url_session_id()

    if local and (not from_url or from_url == local):
        return local
    if not from_url and participant_route():
        # A closed tab takes the URL with it; the cookie is what is left. Only
        # for a participant: the lab app renders this same layout, and a
        # researcher opening it to read the wording should get a clean run, not
        # whichever attempt they previewed last week.
        from_url = _recalled_session(variant_id)
    if not from_url:
        return start_session(pfx)

    record = load(from_url)
    if record is None:
        # An id nothing was ever stored under — a mangled link, or a store that
        # has since been emptied. A fresh attempt beats an error page.
        adopt_session(pfx, from_url)
        return from_url
    if not is_resumable(record):
        if str(record.get("status")) == SUBMITTED:
            # The one real retry for the submit-time upload. The attempt inside
            # `_render_submit` gets at most one extra run, because once
            # `{pfx}_submitted` is set that page renders a line and returns, and
            # a refresh starts a new websocket that has lost the record. Coming
            # back to the link lands here with the stored record in hand, which
            # is exactly what should be in the bucket. `mirror` marks the key
            # only on success, so this is a no-op once it has worked.
            mirror(record, once_key=f"{pfx}_mirrored_submit")
        if render_already_submitted(pfx=pfx):
            # Somebody else on a shared browser. A new id, and the cookie is
            # rewritten to it on the way through, so the finished attempt is
            # left alone and this person is not locked out behind it.
            _clear_session_state(pfx)
            st.query_params.pop(SESSION_PARAM, None)
            start_session(pfx)
            st.rerun()
        return None

    decision = render_resume_gate(pfx=pfx, progress=_progress_label(record))
    if decision == RESUME:
        _rehydrate(pfx, step_key, record, stages)
        adopt_session(pfx, from_url)
        st.rerun()
    if decision == RESTART:
        abandon(from_url, record)
        _clear_session_state(pfx)
        st.query_params.pop(SESSION_PARAM, None)
        # Minted here rather than left to the next run: the browser still holds
        # a cookie pointing at the attempt just abandoned, and a run that came
        # back with no id would recall it and offer to resume the very thing
        # they asked to leave. A new id in the URL beats the cookie, and
        # `remember` overwrites it on the way through.
        start_session(pfx)
        st.rerun()
    return None


def _recalled_session(variant_id: str) -> str:
    """The browser's remembered session id, when it still points at something.

    An abandoned attempt is one they already chose to walk away from, and an id
    with no record behind it is no attempt at all — both are better served by a
    clean start than by the resume gate, which would offer to continue the first
    and say "you have already completed this study" about the second.
    """
    recalled = remembered_session(variant_id, participant_id())
    if not recalled:
        return ""
    record = load(recalled)
    if record is None:
        return ""
    return recalled if str(record.get("status", IN_PROGRESS)) in (IN_PROGRESS, SUBMITTED) else ""


def _response(
    *,
    variant: StudyVariantSpec,
    group: StudyGroupSpec,
    pfx: str,
    step_key: str,
    step: int,
    participant: str,
    ordered: list[StudyStageSpec],
    modalities: list[str | None],
    choice_order: list[str],
) -> dict:
    """The response as it stands, whether or not the study is finished.

    One builder for the checkpoints and for the submission, so a session that is
    abandoned half-way through reads back in exactly the shape a completed one
    does — with fewer stages filled in — and the analytics needs no second
    reader for partial data.
    """
    tasks = [stage.task for stage in ordered]
    experience, remarks = exit_answers(pfx)
    # Keyed by task rather than by display: a task is what a set of answers is
    # about, and which display it was read in varies per participant here. The
    # `stage_modalities` map below is how the two are joined back up.
    return {
        "participant_id": participant,
        # Participant link, or a researcher rendering the same layout in the lab
        # app: this record exists from the first page load, so previews land in
        # the store next to real attempts unless they say which they are.
        "origin": origin(),
        **({"prolific": prolific_fields()} if prolific_fields() else {}),
        "group": group.id,
        "study_variant": variant.id,
        # Where they are, so a resumed session comes back to the same page.
        "step": step,
        # The draw, spelled out: without it a response says which task was
        # answered but not what any of it was a comparison against. Stored whole
        # as well, because resuming has to reproduce it exactly.
        "assignment": st.session_state.get(f"{pfx}_assignment"),
        "task_order": tasks,
        "task_labels": {stage.task: stage.label for stage in ordered},
        "sessions": {stage.task: stage.session_name for stage in ordered},
        # Under the name every layout since the order-effect study uses, so
        # the analytics reads one key for "which displays, in what order".
        "evidence_order": [m for m in modalities if m is not None],
        "stage_modalities": {
            stage.task: modality
            for stage, modality in zip(ordered, modalities, strict=True)
            if modality is not None
        },
        "modality_assignment": {
            stage.task: (CHOSEN if i == len(ordered) - 1 else ASSIGNED)
            for i, stage in enumerate(ordered)
        },
        # The stage-3 answer, on its own key as well as inside the maps
        # above: which display someone reaches for unprompted is a result of
        # this layout, not a detail of one of its stages.
        "chosen_modality": modalities[-1] if modalities else None,
        # Which side each option was on, and how long the choice took. Both
        # are about the answer above rather than about a stage: a pick made
        # in two seconds off the left button is weaker evidence than one made
        # in twenty, and the sides were drawn so that can be checked.
        "choice_button_order": list(choice_order),
        "modality_choice_seconds": st.session_state.get(f"{pfx}_choice_seconds"),
        "session": " → ".join(stage.session_name for stage in ordered),
        "understanding_responses": {
            stage.task: st.session_state.get(f"{pfx}_{stage.task}_responses", {})
            for stage in ordered
        },
        "nasa_tlx": {
            stage.task: st.session_state.get(f"{pfx}_{stage.task}_tlx_responses", {})
            for stage in ordered
        },
        "data_collection_priorities": {
            stage.task: st.session_state.get(f"{pfx}_{stage.task}_choice") for stage in ordered
        },
        "robot_data_collection_experience": experience,
        "remarks": remarks,
        # When the attempt started, not when it ended: a record exists from the
        # first page now, and `submitted_at` is what says it finished.
        "timestamp": _started_at(pfx),
        "step_durations_seconds": get_step_durations(step_key),
        # Named as well as numbered: the durations are keyed by step index,
        # and an index means nothing once the page order has been edited. The
        # same list the progress bar draws, so a participant and a timing chart
        # call a step the same thing. Task-free by design — see
        # :func:`three_stage_step_names` — with `task_order` above saying which
        # task each numbered one was.
        "step_names": three_stage_step_names(modalities),
    }


def _started_at(pfx: str) -> str:
    """When this attempt was first written, in the store's timestamp format."""
    key = f"{pfx}_timestamp"
    stamped = st.session_state.get(key)
    if not stamped:
        stamped = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state[key] = stamped
    return str(stamped)


# ── Pages ─────────────────────────────────────────────────────────────────────


def render_three_stage(variant: StudyVariantSpec, group: StudyGroupSpec) -> None:
    """Render the three-stage study for one participant."""
    if not variant.stages:
        st.error(f"{variant.id} lists no stages — see configs/viz/study_variants/.")
        return

    pfx = f"ts_{group.id}"
    step_key = f"{pfx}_step"
    stages = variant.stages
    n_stages = len(stages)

    # Before anything else: whose attempt is this, and is it a new one? A gate
    # on screen (resume, or "you already finished") means there is no study to
    # render this run.
    session_id = _resolve_session(pfx, step_key, stages, variant_id=variant.id)
    if session_id is None:
        return

    # The URL carries the participant, so hundreds of responses are not all
    # "anonymous". The text box remains for piloting by hand, where there is no
    # recruitment link to arrive on.
    participant = participant_id()
    # Only in the lab app. On the participant route the sidebar is hidden, so
    # this block is a box nobody can type in and two captions nobody should read
    # — and if the CSS hiding it ever lapses, it is the participant's own id and
    # session id that show up on screen.
    if not participant_route():
        st.sidebar.header("Study configuration")
        if participant:
            st.sidebar.caption(f"Participant `{participant}`")
        else:
            participant = st.sidebar.text_input(
                "Participant ID (pilot)", value="", key=f"{pfx}_pid"
            )
        st.sidebar.caption(f"Session `{session_id}`")

    # Kept in the browser as well as in the URL, so closing the tab is not the
    # end of the attempt: the bare recruitment link finds it again. Written
    # after the id is settled, which includes the id "start over" just minted.
    if participant_route():
        remember(variant.id, participant, session_id)

    assignment = _assignment(pfx, n_stages)
    ordered = [stages[i] for i in assignment["stage_tasks"]]
    chosen = st.session_state.get(f"{pfx}_chosen_modality")
    modalities = _stage_modalities(assignment, chosen)
    step = st.session_state.get(step_key, _STEP_INTRO)

    render_progress_bar(step, three_stage_step_names(modalities))
    record_step_entry(step, step_key)

    # Every stage's session is loaded before the study starts, and on every
    # entry path into it, on a page where a failure costs nothing. Loaded only
    # on arrival, a stage 2 whose rollouts are missing dead-ends a participant
    # who has already answered stage 1: a stage that cannot render has no
    # Continue button to get past. It also warms the caches, so the later stages
    # open without a wait.
    if any(_stage_state(pfx, stage, variant.graph_params) is None for stage in ordered):
        st.error(
            "One of this study's tasks could not be loaded. See the sidebar. "
            "Your answers so far are saved; please tell the experimenter rather "
            "than continuing."
        )
        return

    def _checkpoint_now() -> None:
        """Write the answers as they now stand, from wherever we are.

        Handed to the stage's question fragment, which a rerun of the parent
        never reaches. Everything it needs is closed over rather than threaded
        through six parameters, and it builds the payload the same way the
        end-of-run checkpoint below does, so the two cannot describe an attempt
        differently.
        """
        if st.session_state.get(f"{pfx}_submitted"):
            return
        checkpoint(
            pfx,
            session_id,
            _response(
                variant=variant,
                group=group,
                pfx=pfx,
                step_key=step_key,
                step=st.session_state.get(step_key, step),
                participant=participant,
                ordered=ordered,
                modalities=_stage_modalities(
                    assignment, st.session_state.get(f"{pfx}_chosen_modality")
                ),
                choice_order=_choice_order(assignment),
            ),
        )

    _render_step(
        pfx=pfx,
        step_key=step_key,
        step=step,
        variant=variant,
        group=group,
        participant=participant,
        session_id=session_id,
        ordered=ordered,
        modalities=modalities,
        n_stages=n_stages,
        choice_order=_choice_order(assignment),
        checkpoint_now=_checkpoint_now,
    )

    # Last thing on every run: the answers as they now stand. Only a run that
    # changed one writes (see `session.checkpoint`), and a submitted response is
    # never written over with an in-progress one.
    if not st.session_state.get(f"{pfx}_submitted"):
        current = st.session_state.get(step_key, step)
        payload = _response(
            variant=variant,
            group=group,
            pfx=pfx,
            step_key=step_key,
            step=current,
            participant=participant,
            ordered=ordered,
            modalities=modalities,
            choice_order=_choice_order(assignment),
        )
        checkpoint(pfx, session_id, payload)
        # And off the box, at each intermission. The local file is the study and
        # it has no replica; a participant who stops halfway has everything up
        # to the last intermission somewhere else. Off by default, and every
        # failure is swallowed — see :mod:`..backup`.
        if _is_intermission(current, n_stages):
            mirror(
                build_record(session_id, payload, IN_PROGRESS),
                once_key=f"{pfx}_mirrored_{current}",
            )


def _render_step(
    *,
    pfx: str,
    step_key: str,
    step: int,
    variant: StudyVariantSpec,
    group: StudyGroupSpec,
    participant: str,
    session_id: str,
    ordered: list[StudyStageSpec],
    modalities: list[str | None],
    n_stages: int,
    choice_order: list[str],
    checkpoint_now: Callable[[], None],
) -> None:
    """Whichever page the participant is on. One caller, so it can checkpoint."""
    if step == _STEP_INTRO:
        _render_intro(ordered, choice_order)
        if st.button("Continue →", type="primary", key=f"{pfx}_intro_continue"):
            advance_step(_STEP_INTRO, step_key)
        return

    if step >= _exit_step(n_stages):
        # The questions and the button that ends the study, on one page.
        answered = render_exit_questions(pfx, step_key, _exit_step(n_stages), show_continue=False)
        _render_submit(
            answered=answered,
            variant=variant,
            group=group,
            pfx=pfx,
            step_key=step_key,
            step=step,
            participant=participant,
            session_id=session_id,
            ordered=ordered,
            modalities=modalities,
            n_stages=n_stages,
            choice_order=choice_order,
        )
        return

    # Everything in between is a stage page or the intermission after it.
    stage_number = (step + 1) // 2
    stage = ordered[stage_number - 1]
    if step == _stage_step(stage_number) + 1:
        _render_intermission(
            pfx,
            step,
            step_key,
            done_number=stage_number,
            # The intermission before the last stage is where its display is
            # picked; every earlier one just says what is done.
            choice_order=choice_order if stage_number == n_stages - 1 else None,
        )
        return

    modality = modalities[stage_number - 1]
    if modality is None:
        # The last stage, with nothing picked. Reachable by a stored session
        # whose step got past the choice without one — a stage rendered with no
        # display is not a page, so the choice is offered again instead.
        _render_intermission(
            pfx,
            step - 1,
            step_key,
            done_number=stage_number - 1,
            choice_order=choice_order,
        )
        return

    _render_stage(
        pfx=pfx,
        step=step,
        step_key=step_key,
        stage=stage,
        stage_number=stage_number,
        n_stages=n_stages,
        modality=modality,
        variant=variant,
        checkpoint_now=checkpoint_now,
    )


def _render_intro(ordered: list[StudyStageSpec], choice_order: list[str]) -> None:
    """What the session holds, without describing the tasks themselves.

    Each task is described on its own stage page (see
    :func:`behavior_graphs.viz.streamlit.user_studies.evidence
    .render_task_restatement`), where it is what a participant needs while
    answering rather than three paragraphs they read once and click past.
    """
    n = len(ordered)
    # Counted in words, like the question banks' own prompts: "3 different robot
    # tasks" reads like a form field.
    tasks_word = COUNT_WORDS.get(n, n)
    assigned_word = COUNT_WORDS.get(n - 1, n - 1)
    # Named in the order this participant drew, and with no verb on either. The
    # pair used to read "watching rollout videos and exploring an interactive
    # behavior graph", which gave one of them the active verb, the adjective,
    # and second place in the sentence.
    first, second = (_PHASE_NOUN[modality] for modality in choice_order)
    st.markdown(f"""
This study compares two ways of understanding a robot policy's behavior:
{first} and {second}.

### What you will do

You will look at **{tasks_word} different robot tasks**, one after another. Each task has
its own page, and each page shows you one of the two display methods above,
followed by questions about two things: what you understand the policy to be
doing, and which demonstrations you would collect to improve it.

For the first {assigned_word} tasks the display method is chosen for you, and it is a
different one each time. For the last task, **you choose** which of the two you
would rather use.

### What to consider

- Robot policies are trained to imitate expert demonstrations from teleoperators.
- By evaluating them, we aim to understand how the policies perform and subsequently improve them with added data that addresses the failures we observe.
- Robot policies may fail in lots of different ways! Look out for shared patterns in failed trajectories that are shared among failures.

Your responses are saved as you go. If you close this page, opening the link
again will offer to continue from where you stopped.

If anything goes wrong, or you have a question about the study, get in touch at
{CONTACT_EMAIL}.
""")
    # The tasks themselves are not listed, here or anywhere a participant
    # reads: they are "Task 1", "Task 2", "Task 3" throughout, in the order this
    # participant meets them, and the progress bar above numbers them the same.


def _render_intermission(
    pfx: str,
    step: int,
    step_key: str,
    *,
    done_number: int,
    choice_order: list[str] | None = None,
) -> None:
    """Between two stages: the one that is done, and that it is recorded.

    Nothing about what comes next, except when what comes next is the
    participant's own decision. The page used to announce the next task and the
    display it would be read in, which told a participant nothing they were
    about to be shown anyway.

    *choice_order* turns this into the page that collects the last stage's
    display, with the two buttons in the order drawn for this participant. The
    choice used to be a modal on the stage page itself, on the grounds that it
    had to be made before any evidence was on screen — but this page has no
    evidence on it either, and it costs a dialog that a participant could
    dismiss into a dead end (fixed once already) and a page underneath whose
    only purpose was to reopen it.

    By number rather than by name, like every other page: the stages carry a
    drawn order, so "Task 2" says where in the study someone is where "Prepare
    the tray" says only which task they drew.
    """
    st.header("Intermission")
    st.markdown(f"Task {done_number} is complete. Your responses so far have been saved.")
    if choice_order is None:
        if st.button(
            "Continue to the next task →", type="primary", key=f"{pfx}_intermission_{step}"
        ):
            advance_step(step, step_key)
        return

    st.markdown(
        f"""For **Task {done_number + 1}**, you decide how to review the policy's
attempts. You can review {_PHASE_CLAUSE[choice_order[0]]}, or
{_PHASE_CLAUSE[choice_order[1]]}.

Pick whichever you would rather use. You will review the task with the one you
pick, and then answer the same kind of questions as before."""
    )
    for column, modality in zip(st.columns(2), choice_order, strict=True):
        with column:
            if st.button(
                _PHASE_IMPERATIVE[modality],
                key=f"{pfx}_choose_{modality}",
                width="stretch",
            ):
                st.session_state[f"{pfx}_chosen_modality"] = modality
                # How long the choice took, kept off the stage's own clock: a
                # pick made in two seconds is weaker evidence than one made in
                # twenty, and the stage duration should be time spent with the
                # display like the other stages'.
                deliberated = step_elapsed(step, step_key)
                if deliberated is not None:
                    st.session_state[f"{pfx}_choice_seconds"] = round(deliberated, 1)
                advance_step(step, step_key)


def _render_stage(
    *,
    pfx: str,
    step: int,
    step_key: str,
    stage: StudyStageSpec,
    stage_number: int,
    n_stages: int,
    modality: str,
    variant: StudyVariantSpec,
    checkpoint_now: Callable[[], None],
) -> None:
    """One stage: its display, then its task's questions."""
    state = _stage_state(pfx, stage, variant.graph_params)
    if state is None:
        # Numbered here too: this is a page a participant can land on, and the
        # sidebar error beside it (hidden from them) carries the task name.
        st.error(f"Failed to load Task {stage_number}. Check the sidebar for errors.")
        return

    st.subheader(f"Task {stage_number} of {n_stages}")
    if stage_number == n_stages:
        st.caption(f"You chose to review this task with {_PHASE_DEFINITE.get(modality, modality)}.")

    def _restate_task() -> None:
        render_task_restatement(
            phase=modality,
            task_description=state["task_description"],
            index=state["index"],
            mp4_dir=state["mp4_dir"],
            graph_params=state["graph_params"],
            question_set=stage.question_set,
        )

    render_phase_view(
        modality,
        state["prefix"],
        state["index"],
        state["mp4_dir"],
        state["labels"],
        state["metadata"],
        state["hide_episode_index"],
        state["graph_params"],
        state["run_dir"],
        before_evidence=_restate_task,
    )

    st.divider()
    st.header("Questions")
    if not stage.question_set:
        st.error(f"{stage.task} has no question_set — see configs/viz/study_variants/.")
        return

    # A fragment, so answering does not redraw the evidence above it. Without
    # one, every radio was a full app rerun that re-rendered the whole page:
    # measured at 250–730ms on a video stage, which reads as the page reloading
    # under the pointer. Nothing was re-fetching — the twelve video elements
    # kept their identity across a click — it was the repaint.
    #
    # The fragment carries the Continue button, because that is what `complete`
    # gates and a fragment cannot update a control outside itself. `advance_step`
    # ends in an app-scoped `st.rerun()`, so leaving the stage still redraws the
    # page, which is the one time it should.
    @st.fragment
    def _questions() -> None:
        responses, tlx, choice, can_continue = render_task_questions(
            key_prefix=f"{state['prefix']}_questions",
            question_set=stage.question_set,
            demo_videos_dir=Path(state["demo_videos_dir"]) if state["demo_videos_dir"] else None,
        )
        st.session_state[f"{pfx}_{stage.task}_responses"] = responses
        st.session_state[f"{pfx}_{stage.task}_tlx_responses"] = tlx
        st.session_state[f"{pfx}_{stage.task}_choice"] = choice
        # The parent's checkpoint runs at the end of `render_three_stage`, which
        # a fragment rerun never reaches. Called here instead, so an answer is
        # still written the moment it is given: the layout promises responses
        # are saved as you go, and a participant who abandons mid-stage is the
        # case that promise exists for.
        checkpoint_now()
        if st.button(
            "Continue →",
            type="primary",
            disabled=not can_continue,
            key=f"{pfx}_{stage.task}_continue",
        ):
            advance_step(step, step_key)

    _questions()


def _render_submit(
    *,
    variant: StudyVariantSpec,
    group: StudyGroupSpec,
    pfx: str,
    step_key: str,
    step: int,
    participant: str,
    session_id: str,
    ordered: list[StudyStageSpec],
    modalities: list[str | None],
    n_stages: int,
    choice_order: list[str],
    answered: bool = True,
) -> None:
    st.divider()
    st.markdown("Press submit to record your answers as complete.")
    if st.session_state.get(f"{pfx}_submitted"):
        st.success("Your answers have been submitted. Thank you.")
        # A second chance on this websocket. The real retry is on the returning
        # path in `_resolve_session`, which re-derives the record from the store
        # and therefore survives the refresh this branch does not.
        stored = st.session_state.get(f"{pfx}_submitted_record")
        if stored:
            mirror(stored, once_key=f"{pfx}_mirrored_submit")
        return

    if st.button("Submit", type="primary", disabled=not answered, key=f"{pfx}_submit"):
        record_step_exit(_submit_step(n_stages), step_key)
        result = {
            **_response(
                variant=variant,
                group=group,
                pfx=pfx,
                step_key=step_key,
                step=step,
                participant=participant,
                ordered=ordered,
                modalities=modalities,
                choice_order=choice_order,
            ),
            "submitted_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        try:
            submit(pfx, session_id, result)
        # Unlike a checkpoint, this failure is the participant's business: they
        # are pressing Submit and are owed the truth about whether it worked.
        except Exception as error:  # noqa: BLE001 — surfaced, not swallowed
            st.error(
                "Your answers could not be submitted just now. Please press "
                "Submit again; if it keeps failing, tell the experimenter. "
                f"Nothing you did is lost. ({error})"
            )
            return
        # Read by the checkpoint at the end of the run, so a submitted response
        # is not immediately written over with an in-progress one.
        # After the response is safely stored, and to its own file: the address
        # The last intermission is before stage 3, so without this the answers
        # that only exist at the end would never leave the box.
        submitted_record = build_record(session_id, result, SUBMITTED)
        # Kept so a failed upload can be retried on a later run (above).
        st.session_state[f"{pfx}_submitted_record"] = submitted_record
        mirror(submitted_record, once_key=f"{pfx}_mirrored_submit")
        st.session_state[f"{pfx}_submitted"] = True
        # No record id on screen: it is a filename, it means nothing to the
        # person reading it, and it is the same message either way — see the
        # already-submitted branch above.
        st.success("Your answers have been submitted. Thank you.")
        st.download_button(
            "Download your response",
            data=json.dumps(result, indent=2),
            file_name=f"three_stage_{group.id}_{result['submitted_at']}.json",
            mime="application/json",
        )
        st.balloons()
