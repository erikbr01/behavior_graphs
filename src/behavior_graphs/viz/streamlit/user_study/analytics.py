"""Tidy tables out of stored study responses, whatever layout wrote them.

Five generations of the study are in the results store, and no two of them
record an answer the same way:

============================  ============================================
``production_group_ab``       ``app_group_a`` / ``app_group_b``. Group is the
                              *condition* (A = no graph, B = graph), the whole
                              session is rated once, and the answers are
                              ``allocations`` or ``ranked_choices`` plus three
                              ``likert_*`` blocks. Writes no ``study_variant``.
``order_effect_v1``           Both modalities, one after the other. Group is
                              the *order*. Per-phase answers live in top-level
                              keys named after the phase —
                              ``survey_after_graph``,
                              ``nasa_tlx_after_videos``, …
``contrastive_*_v1``          Both modalities with the questions repeated after
                              each. Per-phase answers are nested *inside*
                              ``understanding_responses`` / ``nasa_tlx`` /
                              ``data_collection_priorities``, and the free-text
                              box is ``remarks`` rather than ``notes``.
``three_stage_v1``            Three tasks, one modality each, the last one the
                              participant's own pick. Same nested blocks as the
                              contrastive layout, but keyed by *task* rather
                              than by modality — the modality a task was read
                              in varies per participant and is recorded in
                              ``stage_modalities``.
============================  ============================================

The clash worth knowing about: ``nasa_tlx`` is ``{dimension: score}`` in the
production layout and ``{phase: {dimension: score}}`` in the contrastive one.
Reading the first as the second silently yields nothing, which is what the
analytics page used to do — it drew an empty TLX panel for every contrastive
response it had.

The second clash: the same nested key is a modality in the contrastive layout
and a task in the three-stage one. A row therefore carries both a ``task`` and a
``phase``, resolved through ``stage_modalities`` — colouring a chart by "phase"
when the key was a task name would have put "cooking" in a legend of modalities.

One more thing the three-stage layout changed: it writes a record as the answers
come in rather than once at the end, so the store holds unfinished attempts too.
``status`` says which an attempt is, every row carries it, and :func:`completed`
is what a result should be computed over — a session abandoned at stage one is
data, but it is not a participant. For the same reason a record says where it
came from: rendering that layout in the lab app writes one too, and
:func:`from_participants` is what leaves those previews out.

So the page does not read response dicts. It reads these functions, each of
which returns a list of flat row dicts ready for a DataFrame, with ``phase``
empty for the layouts that rate a session once. Anything a layout does not
record simply produces no rows.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

#: Responses from ``app_group_a`` / ``app_group_b`` carry no ``study_variant``:
#: they predate the variant registry. They are still a version, and this is the
#: id of the registry entry that runs them (``production_group_ab.yaml``).
PRODUCTION_VERSION = "production_group_ab"

#: The layout the study reports: three tasks, two displays assigned and the
#: third the participant's own pick. The store also holds every layout in the
#: table above, which asked different questions of different people, so a
#: result is computed over :func:`for_version` of this and not over the store.
THREE_STAGE_VERSION = "three_stage_v1"

#: Where a phase's answers hide in the order-effect layout: one top-level key
#: per phase, the phase name suffixed onto a block name.
_SUFFIXED_BLOCKS = {
    "survey_after_": "understanding",
    "nasa_tlx_after_": "nasa_tlx",
    "data_collection_priorities_after_": "data_collection",
}

#: The production layout's numeric blocks, and what to call them on a chart.
LIKERT_BLOCKS = {
    "likert_strategy": "Strategy selection",
    "likert_final": "Final assessment",
    "likert_graph": "Graph interaction",
}

#: NASA-TLX dimensions in the order the instrument asks them, so a chart's
#: x-axis is not alphabetical.
TLX_DIMENSIONS = (
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance",
    "effort",
    "frustration",
    # Not a TLX dimension, but asked by the same block and stored in the same
    # map, so a chart that left it out would silently drop an answer.
    "confidence",
)

#: The closing question's answers, weakest first. Here rather than beside the
#: radio that asks it because a chart of the answers has to be ordered by the
#: scale — alphabetical puts "A great deal" next to "A little" — and the two
#: orders drifting apart would relabel the axis without anyone noticing.
EXPERIENCE_SCALE = (
    "None at all",
    "A little",
    "Some",
    "A good deal",
    "A great deal",
)

TLX_LABELS = {
    "mental_demand": "Mental demand",
    "physical_demand": "Physical demand",
    "temporal_demand": "Temporal demand",
    "performance": "Performance",
    "effort": "Effort",
    "frustration": "Frustration",
    "confidence": "Confidence",
}

#: Which way a rated dimension improves — ``"down"`` where a low answer is the
#: better display, ``"up"`` where a high one is.
#:
#: Every load item runs low-is-better, ``performance`` included: its scale is
#: labelled Perfect to Failure, so a high score is a participant saying they did
#: badly (:mod:`behavior_graphs.viz.streamlit.user_study.nasa_tlx`, whose question
#: text says so outright). ``confidence`` is the one that runs the other way.
#:
#: Here rather than in the chart that draws it because *both* the app's box
#: plots and the paper figures need it, and the direction is a property of the
#: instrument. A chart that puts these dimensions on one axis has to say which
#: way each of them improves, not say it once: a reader who takes the load
#: items' arrow for all of them reads the confidence panel backwards.
BETTER_DIRECTION = {**{key: "down" for key in TLX_LABELS}, "confidence": "up"}


#: What a record's ``status`` means, and the order a chart should show them in.
#: Unset is a response from a layout that only ever wrote on submit, so it is a
#: completed one.
STATUSES = ("submitted", "in_progress", "abandoned")


def status(response: dict) -> str:
    """``submitted`` / ``in_progress`` / ``abandoned``.

    The three-stage layout checkpoints its answers as they are given, so the
    store now holds unfinished attempts alongside finished ones. Every panel that
    counts participants or averages an answer has to know which it is looking at:
    pooling a session abandoned at stage one with a completed one silently
    reweights the result.
    """
    value = str(response.get("status") or "").strip()
    if value in STATUSES:
        return value
    # No status at all: written by a layout that saved once, at the end.
    return "submitted"


def is_complete(response: dict) -> bool:
    return status(response) == "submitted"


#: Where a record came from — see ``user_studies.session.origin``.
PARTICIPANT_ORIGIN = "participant"
LAB_ORIGIN = "lab"


def origin(response: dict) -> str:
    """``participant`` for a recruitment-link session, ``lab`` for a preview.

    The three-stage layout writes a record on the first page load, so opening
    that layout in the lab app to check a wording leaves a record too — with no
    participant id and stopped on the introduction, which is exactly what a real
    dropout looks like. Anything that does not say is a participant: every
    response written before this field existed is one, and a filter that guesses
    the other way would hide collected data.
    """
    return (
        LAB_ORIGIN
        if str(response.get("origin") or "").strip() == LAB_ORIGIN
        else PARTICIPANT_ORIGIN
    )


def from_participants(responses: Iterable[dict]) -> list[dict]:
    """Everything except the lab app's previews."""
    return [r for r in responses if origin(r) == PARTICIPANT_ORIGIN]


def study_version(response: dict) -> str:
    """Which layout wrote this response."""
    return str(response.get("study_variant") or PRODUCTION_VERSION)


def version_counts(responses: Iterable[dict]) -> dict[str, int]:
    """``{version: number of responses}``, most responses first.

    Ordered by count so the dropdown opens on the study actually being run
    rather than on whichever id sorts first.
    """
    counts: dict[str, int] = {}
    for response in responses:
        version = study_version(response)
        counts[version] = counts.get(version, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def for_version(responses: Iterable[dict], version: str | None) -> list[dict]:
    """The responses written by *version*, or all of them when it is ``None``."""
    if version is None:
        return list(responses)
    return [r for r in responses if study_version(r) == version]


def completed(responses: Iterable[dict]) -> list[dict]:
    """Only the finished attempts — what a result should be computed over."""
    return [r for r in responses if is_complete(r)]


def status_counts(responses: Iterable[dict]) -> dict[str, int]:
    """``{status: how many}``, in the order :data:`STATUSES` lists them."""
    counts = {name: 0 for name in STATUSES}
    for response in responses:
        counts[status(response)] = counts.get(status(response), 0) + 1
    return {name: n for name, n in counts.items() if n}


def phases(response: dict) -> list[str]:
    """The modalities this participant saw, in the order they saw them.

    Empty for the production layout, which shows one condition and never names
    it in the response.
    """
    order = response.get("evidence_order") or response.get("modality_order") or []
    return [str(phase) for phase in order]


def stage_modalities(response: dict) -> dict[str, str]:
    """``{task: modality}`` — which display each task was read in.

    Only the three-stage layout varies that within a participant; ``{}``
    everywhere else, which is what makes it usable as the test for "is this
    nested key a task or a modality".
    """
    raw = response.get("stage_modalities")
    if not isinstance(raw, dict):
        return {}
    return {str(task): str(modality) for task, modality in raw.items() if modality}


def task_order(response: dict) -> list[str]:
    """The tasks this participant saw, in the order they saw them; ``[]`` if one."""
    order = response.get("task_order")
    return [str(task) for task in order] if isinstance(order, list) else []


def task_labels(response: dict) -> dict[str, str]:
    """What this response calls each of its tasks; ``{}`` for a one-task layout.

    Stored per response rather than looked up from a config, so a response stays
    readable after the task it names has been relabelled — the same reason
    :func:`step_labels` prefers what was written at submit time.
    """
    labels = response.get("task_labels")
    if not isinstance(labels, dict):
        return {}
    return {str(task): str(label) for task, label in labels.items()}


def _nested(response: dict, key: str) -> tuple[str, str]:
    """``(task, phase)`` for one key of a nested answer block.

    The contrastive layout keys those blocks by modality and the three-stage one
    by task, and both are bare strings — so the task list is what tells them
    apart. A key that names a task resolves to the modality that task was read
    in, which is the value every per-modality chart on the page is grouped by.
    """
    by_task = stage_modalities(response)
    if key in by_task:
        return key, by_task[key]
    return "", key


def group_label(response: dict) -> str:
    """The group, said in a way that does not mean two different things.

    "A" is the no-graph condition in the production layout and the
    videos-first *order* in every layout since — same letter, different
    variable. Where the response records an order, the label carries it.
    """
    group = str(response.get("group", "?"))
    order = phases(response)
    if not order:
        return group
    return f"{group} ({' → '.join(order)})"


def participant_rows(responses: Iterable[dict]) -> list[dict]:
    """One row per response: who, which layout, which group, how long."""
    rows = []
    for response in responses:
        durations = response.get("step_durations_seconds")
        if not isinstance(durations, dict):
            durations = {}
        # This row is built before any panel, so a bad value here costs the
        # whole page rather than one chart.
        total = sum(float(v) for v in durations.values() if isinstance(v, int | float))
        rows.append(
            {
                "participant": participant(response),
                "version": study_version(response),
                "group": str(response.get("group", "?")),
                "group_label": group_label(response),
                "status": status(response),
                # Which page they were on when the record was last written —
                # ``None`` for a layout that saved only on submit.
                "step": response.get("step") if isinstance(response.get("step"), int) else None,
                "phases": " → ".join(phases(response)),
                # Empty for the one-task layouts, where the session column
                # already says which task it was.
                "tasks": " → ".join(task_order(response)),
                "session": str(response.get("session", "")),
                "timestamp": str(response.get("timestamp", "")),
                "experience": experience(response),
                "minutes": round(total / 60, 1) if total else None,
                "remarks": remarks(response),
            }
        )
    return rows


def participant(response: dict) -> str:
    """Who gave this response, by the best identifier it carries.

    The study recruits over a bare link, which carries no participant id --
    see ``user_studies/browser_session.cookie_name`` -- so ``participant_id``
    is empty on every anonymous session, and falling back to one shared
    "anon" makes a whole cohort look like one person. That has been caught
    four times, each time in a different panel and never loudly:

    - the Participants tile read 1 beside Responses 30;
    - failure-mode scoring concatenated every session's ranking into one per
      task and reported coverage above 100% (see
      :func:`failure_scoring.score_rows`);
    - the per-task task-load tables counted one person per cell however many
      had rated it, which is why their rows carry ``response_id``;
    - the adjusted regret model's join onto these rows matched nothing at all,
      so its panel silently did not draw.

    Each was fixed where it surfaced, and this function kept returning "anon"
    for the next one to find. It is the one reading now.

    Every record the three-stage layout writes has a session id, which is the
    identifier that actually distinguishes those responses. ``session:`` keeps
    it legible as one rather than passing for a person's own label.

    Only a response carrying neither id is "anon" -- with nothing left to tell
    two of those apart, one group is the honest answer rather than a lossy one.
    """
    given = str(response.get("participant_id", "") or "").strip()
    if given:
        return given
    session = str(response.get("session_id", "") or "").strip()
    return f"session:{session}" if session else "anon"


def experience(response: dict) -> str:
    """The robot-data-collection experience answer, "" when unasked."""
    return str(response.get("robot_data_collection_experience") or "")


def remarks(response: dict) -> str:
    """The free-text box, under either of the two names it has had."""
    return str(response.get("remarks") or response.get("notes") or "")


def _identity(response: dict) -> dict:
    return {
        # The same reading as `participant_rows` above, which for most of this
        # module's life it was not: every row builder here goes through the one
        # helper now, so a panel cannot be the one that still sees "anon".
        "participant": participant(response),
        # The attempt a row came from, which is a different question from who
        # gave it — one person sitting the study twice is two response ids
        # against one participant. It arrived here as a way to count people
        # while ``participant`` still collapsed every anonymous session onto
        # "anon"; that is fixed above, so both are now the count they say they
        # are, and a panel labelled "Sessions" is counting attempts on purpose
        # rather than for want of anything better. The store writes
        # ``_response_id`` on save; a record that predates it (or was never
        # saved) falls back to its timestamp.
        "response_id": str(response.get("_response_id", "") or response.get("timestamp", "")),
        "version": study_version(response),
        "group": str(response.get("group", "?")),
        "group_label": group_label(response),
        # On every row, not just the participant table: a page that filters its
        # charts to finished sessions has to be able to.
        "status": status(response),
    }


def _suffixed(response: dict, prefix: str) -> list[tuple[str, dict]]:
    """``[(phase, block)]`` for the order-effect layout's per-phase keys."""
    found = []
    for key, value in response.items():
        if key.startswith(prefix) and isinstance(value, dict | list):
            found.append((key[len(prefix) :], value))
    return found


def tlx_rows(responses: Iterable[dict]) -> list[dict]:
    """Task-load scores, one row per (participant, phase, dimension).

    Reads all three shapes: ``{dimension: score}`` rated once,
    ``{phase: {dimension: score}}`` rated per phase, and one
    ``nasa_tlx_after_<phase>`` key per phase.
    """
    rows = []
    for response in responses:
        ident = _identity(response)
        labels = task_labels(response)
        blocks: list[tuple[str, str, dict]] = []
        block = response.get("nasa_tlx") or {}
        if isinstance(block, dict) and block:
            # Per-phase when every value is itself a mapping of scores; a
            # dimension's score is a number, so the two cannot be confused.
            if all(isinstance(v, dict) for v in block.values()):
                blocks.extend(
                    (*_nested(response, str(key)), scores) for key, scores in block.items()
                )
            else:
                blocks.append(("", "", block))
        blocks.extend(
            (*_nested(response, phase), b) for phase, b in _suffixed(response, "nasa_tlx_after_")
        )
        for task, phase, scores in blocks:
            if not isinstance(scores, dict):
                continue
            for dimension, score in scores.items():
                if not isinstance(score, int | float):
                    continue
                rows.append(
                    {
                        **ident,
                        "task": task,
                        # Blank rather than the key for a layout that rates a
                        # session once: a chart grouped by task would otherwise
                        # draw one category called "".
                        "task_label": labels.get(task, task) if task else "",
                        "phase": phase,
                        "dimension": TLX_LABELS.get(str(dimension), str(dimension)),
                        "dimension_key": str(dimension),
                        "score": float(score),
                    }
                )
    return rows


def rating_rows(responses: Iterable[dict]) -> list[dict]:
    """Numeric agree/disagree answers: the order-effect understanding survey
    and the production layout's three Likert blocks."""
    rows = []
    for response in responses:
        ident = _identity(response)
        blocks: list[tuple[str, str, dict]] = [
            (phase, "understanding", answers)
            for phase, answers in _suffixed(response, "survey_after_")
            if isinstance(answers, dict)
        ]
        for key, label in LIKERT_BLOCKS.items():
            answers = response.get(key)
            if isinstance(answers, dict) and answers:
                blocks.append(("", label, answers))
        for phase, block, answers in blocks:
            for question, score in answers.items():
                if not isinstance(score, int | float):
                    continue
                rows.append(
                    {
                        **ident,
                        "phase": phase,
                        "block": block,
                        "question": str(question),
                        "score": float(score),
                    }
                )
    return rows


def ranking_rows(responses: Iterable[dict]) -> list[dict]:
    """Ordered answers, one row per (participant, phase, rank).

    Three of them: the contrastive failure-mode ranking, the order-effect
    data-collection priorities, and the production layout's ``ranked_choices``.
    Rank 1 is the participant's first pick in all three.
    """
    rows = []
    for response in responses:
        ident = _identity(response)
        blocks: list[tuple[str, str, str, Sequence]] = []
        understanding = response.get("understanding_responses")
        if isinstance(understanding, dict):
            # Per phase now; the layout's first build wrote this block flat, so a
            # value here can be the answer itself rather than a phase's answers
            # (`{"failure_mode_ranking": [...]}`). Without the isinstance the
            # `.get` on that list took the whole page down, not just this panel.
            for key, answers in understanding.items():
                if not isinstance(answers, dict):
                    continue
                ranked = answers.get("failure_mode_ranking")
                if isinstance(ranked, list):
                    blocks.append((*_nested(response, str(key)), "failure_mode", ranked))
        for phase, ranked in _suffixed(response, "data_collection_priorities_after_"):
            if isinstance(ranked, list):
                blocks.append((*_nested(response, phase), "data_collection", ranked))
        legacy = response.get("ranked_choices")
        if isinstance(legacy, list):
            blocks.append(("", "", "strategy", legacy))
        for task, phase, block, ranked in blocks:
            for rank, option in enumerate(ranked, start=1):
                if option is None:
                    continue
                rows.append(
                    {
                        **ident,
                        "task": task,
                        "phase": phase,
                        "block": block,
                        "rank": rank,
                        "option": str(option),
                    }
                )
    return rows


def choice_rows(responses: Iterable[dict]) -> list[dict]:
    """Single-pick answers: which data to collect next, per phase.

    Read from ``data_collection_priorities`` when it maps a phase to one
    option, and otherwise from the per-phase ``data_collection_choice`` inside
    ``understanding_responses`` — the contrastive layout writes both, and a
    response saved mid-refactor may have only one.
    """
    rows = []
    for response in responses:
        ident = _identity(response)
        picked: dict[str, str] = {}
        understanding = response.get("understanding_responses")
        if isinstance(understanding, dict):
            for key, answers in understanding.items():
                if not isinstance(answers, dict):
                    continue
                choice = answers.get("data_collection_choice")
                if isinstance(choice, str) and choice:
                    picked[str(key)] = choice
        priorities = response.get("data_collection_priorities")
        if isinstance(priorities, dict):
            for key, choice in priorities.items():
                if isinstance(choice, str) and choice:
                    picked.setdefault(str(key), choice)
        assignment = response.get("modality_assignment")
        for key, choice in picked.items():
            task, phase = _nested(response, key)
            rows.append(
                {
                    **ident,
                    "task": task,
                    "phase": phase,
                    "block": "data_collection",
                    "choice": choice,
                    # ``assigned`` or ``chosen``, as :func:`stage_rows` reports
                    # it, and empty for the layouts that assign nothing. A
                    # graph-versus-videos comparison needs it: the three-stage
                    # layout's last stage is the participant's own pick, so
                    # pooling it in makes the display a *result* of the
                    # participant rather than something drawn for them.
                    "assignment": (
                        str(assignment.get(key, "")) if isinstance(assignment, dict) else ""
                    ),
                }
            )
    return rows


def stage_rows(responses: Iterable[dict]) -> list[dict]:
    """One row per stage of a multi-task layout: task, display, and how it got it.

    The three-stage layout's own record: which task a participant met in which
    stage, which display it was shown as, and whether that display was drawn for
    them (``assigned``) or picked by them (``chosen``). No rows for the layouts
    that show one task, which have nothing to say here.
    """
    rows = []
    for response in responses:
        tasks = task_order(response)
        if not tasks:
            continue
        ident = _identity(response)
        by_task = stage_modalities(response)
        assignment = response.get("modality_assignment")
        assignment = assignment if isinstance(assignment, dict) else {}
        labels = task_labels(response)
        for stage, task in enumerate(tasks, start=1):
            rows.append(
                {
                    **ident,
                    "stage": stage,
                    "task": task,
                    "task_label": str(labels.get(task, task)),
                    "phase": by_task.get(task, ""),
                    # "assigned" or "chosen" — the distinction the last stage
                    # exists to make, so an unrecorded one is left blank rather
                    # than assumed to be either.
                    "assignment": str(assignment.get(task, "")),
                }
            )
    return rows


def allocation_rows(responses: Iterable[dict]) -> list[dict]:
    """Budget split across strategies — the production layout's other answer."""
    rows = []
    for response in responses:
        ident = _identity(response)
        allocations = response.get("allocations")
        if not isinstance(allocations, dict):
            continue
        for strategy, demos in allocations.items():
            if isinstance(demos, int | float):
                rows.append({**ident, "strategy": str(strategy), "allocation": float(demos)})
    return rows


def step_labels(response: dict) -> list[str]:
    """What this layout calls each of its steps, in order.

    Prefers the list the response stored at submit time, so a response stays
    readable after its layout has been reordered or renamed, and falls back to
    reconstructing it from the version and the evidence order — which is how
    responses written before ``step_names`` existed get named at all.
    """
    from behavior_graphs.viz.streamlit.user_studies.step_names import step_names

    stored = response.get("step_names")
    if isinstance(stored, list) and stored:
        return [str(name) for name in stored]
    return step_names(study_version(response), phases(response))


def progress_label(response: dict) -> str:
    """The name of the page this record was last written on.

    For the funnel: "how many stopped at Stage 2" is the readable version of a
    ``step`` index, and it is what says whether a study is too long or one task
    is doing the damage.
    """
    step = response.get("step")
    if not isinstance(step, int):
        return "unknown"
    names = step_labels(response)
    return str(names[step]) if 0 <= step < len(names) else f"step {step}"


def progress_rows(responses: Iterable[dict]) -> list[dict]:
    """One row per unfinished attempt: how far it got, and its status.

    Only unfinished ones: a completed attempt got to the end by definition, and
    counting it here would bury the drop-offs this table exists to show.
    """
    rows = []
    for response in responses:
        if is_complete(response):
            continue
        rows.append(
            {
                **_identity(response),
                "step": response.get("step") if isinstance(response.get("step"), int) else -1,
                "stopped_on": progress_label(response),
                "updated_at": str(response.get("updated_at", "")),
            }
        )
    return rows


def duration_rows(responses: Iterable[dict]) -> list[dict]:
    """Seconds spent on each step, for every layout — they all record it.

    The stored durations are keyed by step *index*; ``step`` is the name of
    that step, falling back to ``Step 3`` for a layout whose names are not
    known. ``step_index`` is kept so a chart can order the steps by the
    sequence a participant met them in rather than alphabetically.
    """
    rows = []
    for response in responses:
        ident = _identity(response)
        durations = response.get("step_durations_seconds")
        if not isinstance(durations, dict):
            continue
        names = step_labels(response)
        for step, seconds in durations.items():
            if not isinstance(seconds, int | float):
                continue
            try:
                index = int(step)
            except (TypeError, ValueError):
                index = -1
            named = names[index] if 0 <= index < len(names) else f"Step {step}"
            rows.append(
                {
                    **ident,
                    "step_index": index,
                    "step": named,
                    "seconds": float(seconds),
                }
            )
    return rows


#: A step that *is* one task's stage, and which stage it is. Both spellings:
#: the layout calls them "Task 1 (behavior graph)" now and called them
#: "Stage 1 (…)" when the first responses were collected, and each response
#: carries the names it was written with.
_STAGE_STEP = re.compile(r"^(?:task|stage)\s*(\d+)\b", re.IGNORECASE)


def task_duration_rows(responses: Iterable[dict]) -> list[dict]:
    """Seconds each participant spent on each task, for the per-task layouts.

    :func:`duration_rows` keys time by step, which is what a chart of the
    layout's own pages wants and is the wrong key for a question about tasks: a
    task arrives at a different step for every participant, because the stage
    order is drawn. This resolves the step back to the task that was on it, via
    the stage number in the step's name.

    One row per (participant, task) that has a recorded duration. A stage
    nobody reached produces no row rather than a zero — the store holds
    unfinished attempts, and an abandoned session did not spend no time on
    stage three.
    """
    rows = []
    for response in responses:
        tasks = task_order(response)
        if not tasks:
            continue
        durations = response.get("step_durations_seconds")
        if not isinstance(durations, dict):
            continue
        ident = _identity(response)
        labels = task_labels(response)
        by_task = stage_modalities(response)
        # Keyed by step index, as ``step_durations_seconds`` is — by a string
        # index once the record has been through JSON, and by an int before it.
        elapsed: dict[int, float] = {}
        for step, value in durations.items():
            try:
                elapsed[int(step)] = float(value)
            except (TypeError, ValueError):
                continue
        stage_of_step: dict[int, int] = {}
        for index, name in enumerate(step_labels(response)):
            if match := _STAGE_STEP.match(str(name)):
                stage_of_step.setdefault(int(match.group(1)), index)
        for stage, task in enumerate(tasks, start=1):
            index = stage_of_step.get(stage)
            seconds = elapsed.get(index) if index is not None else None
            if seconds is None:
                continue
            rows.append(
                {
                    **ident,
                    "stage": stage,
                    "task": task,
                    "task_label": labels.get(task, task),
                    "phase": by_task.get(task, ""),
                    "seconds": seconds,
                }
            )
    return rows


def exit_answer_rows(responses: Iterable[dict]) -> list[dict]:
    """The closing questions: how much robot-data experience, and any remarks.

    One row per response that answered either — the experience question is only
    asked by the contrastive layout, and the remarks box is optional in all of
    them.
    """
    rows = []
    for response in responses:
        told_us = experience(response)
        wrote = remarks(response)
        if not told_us and not wrote:
            continue
        rows.append({**_identity(response), "experience": told_us, "remarks": wrote})
    return rows


def phase_disagreements(responses: Iterable[dict]) -> list[dict]:
    """Per participant: did their answer change between the two modalities?

    The question the two-phase layouts exist to ask. Only meaningful for a
    response with two phases, so single-condition responses produce no rows.
    """
    rows = []
    for response in responses:
        order = phases(response)
        if len(order) != 2:
            continue
        ident = _identity(response)
        first, second = order
        by_block: dict[str, dict[str, str]] = {}
        for row in choice_rows([response]):
            by_block.setdefault(row["block"], {})[row["phase"]] = row["choice"]
        # Where the answer was a ranking, the comparison is over the top pick:
        # whether the participant's *first* priority moved is the readable
        # version of "did the ranking change".
        for row in ranking_rows([response]):
            if row["rank"] != 1:
                continue
            by_block.setdefault(f"top_{row['block']}", {})[row["phase"]] = row["option"]
        for block, answers in by_block.items():
            if first not in answers or second not in answers:
                continue
            rows.append(
                {
                    **ident,
                    "block": block,
                    "first_phase": first,
                    "second_phase": second,
                    "first_answer": answers[first],
                    "second_answer": answers[second],
                    "changed": answers[first] != answers[second],
                }
            )
    return rows
