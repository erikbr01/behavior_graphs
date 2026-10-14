"""What each numbered step of a study layout is called.

Two readers need these names: the progress bar a participant sees, and the
analytics page's "time per step" chart, which otherwise plots ``0 1 2 3 4 5``
and leaves whoever is reading it to remember which page step 3 was.

They live here rather than inline in each renderer so the two cannot drift —
a chart labelled from a stale copy of the list is worse than one labelled with
numbers. Names for the two-phase layouts depend on the group: step 1 is the
rollout videos for the group that saw them first and the graph for the group
that did not, so a layout's names are a function of its evidence order.

Submitting also records the list into the response, so a response stays readable
after the layout it came from is renamed or reordered — see
``behavior_graphs.viz.streamlit.user_study.analytics.step_labels``, which prefers
what was stored and falls back to these.
"""

from __future__ import annotations

from behavior_graphs.viz.streamlit.user_study.survey_steps import STEP_LABELS

#: What a modality is called in a step name, as against in a response key.
PHASE_STEP_LABELS = {"videos": "Rollout videos", "graph": "Behavior graph"}


def _phase(phase: str) -> str:
    return PHASE_STEP_LABELS.get(phase, phase.title())


def contrastive_step_names(order: list[str]) -> list[str]:
    """The contrastive layout: each modality is followed by its own questions.

    ``[]`` unless the order names both modalities: padding a short one would
    label both phase steps with the same modality, and an absent one would
    assert videos-first — mislabelling every graph-first participant's chart.
    A number beats a wrong word here too.
    """
    if len(order) != 2:
        return []
    first, second = order
    return [
        "Introduction",
        f"{_phase(first)} + questions",
        "Intermission",
        f"{_phase(second)} + questions",
        "Closing questions",
        "Submit",
    ]


def order_effect_step_names(order: list[str]) -> list[str]:
    """The order-effect layout: both modalities, each with a survey after it.

    ``[]`` without both modalities, for the reason in
    :func:`contrastive_step_names`.
    """
    if len(order) != 2:
        return []
    first, second = order
    return [
        "Introduction",
        _phase(first),
        "Survey 1",
        _phase(second),
        "Survey 2",
        "Submit",
    ]


#: What the last stage's display is called before the participant has picked it.
UNCHOSEN = "your choice"


def three_stage_step_names(modalities: list[str | None]) -> list[str]:
    """The three-stage layout: one task per stage, each with its own questions.

    *modalities* is the display each stage is read in, in stage order — the
    first ones assigned, the last one the participant's own pick, which is
    ``None`` until they make it (the progress bar is drawn before the choice,
    and naming a display there would pre-empt it).

    The tasks are numbered, never named. A participant is told "Task 2", not
    "Prepare the tray": the tasks come in a drawn order, so a name says which
    task someone happens to be on and nothing about how far through they are,
    and a timing chart labelled with one is a category of one participant.
    Which task went where is ``task_order`` on the response.
    """
    if not modalities:
        return []
    names = ["Introduction"]
    last = len(modalities) - 1
    for i, modality in enumerate(modalities):
        # The last stage's display was chosen rather than assigned, and the
        # difference is the point of that stage.
        if i == last:
            display = UNCHOSEN if modality is None else f"{_phase(modality).lower()} · {UNCHOSEN}"
        else:
            display = _phase(modality).lower() if modality else UNCHOSEN
        names.append(f"Task {i + 1} ({display})")
        if i < last:
            names.append("Intermission")
    # No "Submit" step: the button sits on the closing page, which used to be
    # followed by a "Review & Submit" page that reviewed nothing, since there is
    # no way back through this study.
    names.append("Closing questions")
    return names


def step_names(version: str, order: list[str] | None = None) -> list[str]:
    """Step names for a study version, ``[]`` when the version is unknown.

    Unknown rather than guessed: a layout added later would otherwise have its
    steps labelled with another layout's names, and a number is a better label
    than a wrong word.
    """
    order = list(order or [])
    if version.startswith("three_stage"):
        # Reconstructed from the modalities alone: the task order is in the
        # response too, but under its own key, and this fallback only runs for a
        # response that stored no names at all.
        return three_stage_step_names(order)
    if version.startswith("contrastive_"):
        return contrastive_step_names(order)
    if version.startswith("order_effect"):
        return order_effect_step_names(order)
    if version.startswith("production"):
        # app_group_a / app_group_b drive the progress bar off this list.
        return list(STEP_LABELS)
    return []
