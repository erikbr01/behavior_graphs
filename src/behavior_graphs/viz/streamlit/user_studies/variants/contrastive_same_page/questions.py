"""Task-specific ranking questions for the contrastive study variants."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import streamlit as st

from behavior_graphs.viz.streamlit.components.collection_cards import (
    card_views,
    render_collection_cards,
)
from behavior_graphs.viz.streamlit.user_study.nasa_tlx import RATING_POINTS, render_nasa_tlx
from behavior_graphs.viz.streamlit.user_study.task_media import (
    collection_video_candidates,
)

# Two of the six NASA-TLX dimensions. The block is answered twice per
# participant, once per display method, and the rest say little about reading a
# graph versus watching clips: physical demand is a mouse either way,
# performance asks them to grade an answer they cannot check, frustration
# duplicates effort at this length, and mental demand and effort were two
# wordings of the same question.
#: Confidence is not a TLX dimension (see ``nasa_tlx``); it is asked here, under
#: the two load items, because how sure somebody is of the answers they just
#: gave is what a display is meant to change and neither load item measures it.
_TLX_ITEMS = ("temporal_demand", "effort", "confidence")


class QuestionSet(TypedDict):
    failure_modes: list[str]
    #: How many of them to rank. Fewer options need fewer ranks: set_table has
    #: four failure modes, and ranking three of four is most of a full ordering
    #: of a list a participant has just read.
    n_ranked: int
    collection_prompt: str
    collection_options: list[str]


#: Spelled out, because "Rank the top 2" reads like a form field. Shared with
#: the study intros, which count tasks for the same reason.
COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _plain_collection_choice(key_prefix: str, options: list[str]) -> str | None:
    """Question 2 without the footage: one radio over the option names.

    Used when no clips are mounted at all, and when clips exist but none can be
    served. Same key either way, so the parent and the response builder never
    have to ask which path drew it.
    """
    return st.radio(
        "Data collection choice",
        options=options,
        index=None,
        key=f"{key_prefix}_collection_choice",
    )


def _render_collection_cards(
    *,
    key_prefix: str,
    question_set: str,
    previews: list[tuple[int, str, list[Path]]],
    all_options: list[str],
) -> str | None:
    """The preview grid, drawn by its own component. Returns the chosen option.

    Was three single-option ``st.radio`` widgets, one per card, with a callback
    to keep them exclusive — which put the deselect on the far side of a
    websocket round trip and made switching cards feel sticky. The component
    keeps the per-card layout and does the deselect in the browser; see
    :mod:`behavior_graphs.viz.streamlit.components.collection_cards` for the
    measurements and for why the two all-Streamlit layouts each fell short.
    """
    usable = [(option, paths[0]) for _index, option, paths in previews if paths]
    if not usable:
        return None
    views = card_views(usable, key=key_prefix)
    if not views:
        # Clips were found on disk but none can be served (a deleted file, a
        # media manager that declined). Rendering nothing here left question 2
        # with a prompt, no options, and a Continue disabled forever with
        # nothing on screen to explain it. Ask it without the footage instead,
        # which is answerable.
        # Every option, not just the ones that had a file. A deployment with
        # two of three folders mounted would otherwise show a radio with one
        # choice, and record a forced answer as a preference. There is no
        # footage on screen either way, so there is nothing to restrict it to.
        return _plain_collection_choice(key_prefix, list(all_options))
    choice_key = f"{key_prefix}_collection_choice"
    picked = render_collection_cards(
        views, key=key_prefix, selected=st.session_state.get(choice_key)
    )
    # On the plain key the parent and the response builder already read, so the
    # component path and the no-clips fallback below are indistinguishable to
    # everything downstream.
    st.session_state[choice_key] = picked
    return picked


QUESTION_SETS: dict[str, QuestionSet] = {
    "set_table": {
        "failure_modes": [
            "The robot fails to pick up the pumpkin.",
            "The robot manages to pick up the pumpkin, but fails to place it on the plate.",
            "The robot manages to pick and place the pumpkin, but fails to grasp the chocolate milk.",
            "The robot manages to pick and place the pumpkin and grasp the chocolate milk, but fails to place the chocolate milk.",
        ],
        "n_ranked": 2,
        "collection_prompt": "You are in charge of collecting new data to improve the policy. Choose one initial object arrangement for additional data.",
        # One per chocolate-milk position, which is the only difference this
        # task can answer. The six earlier options crossed the milk position
        # with the order of the other two objects — a distinction the curated
        # footage does not make, since it is filed by milk position alone, so
        # two options a participant read as different were illustrated by the
        # same clips.
        #
        # Each label describes the arrangement in the clips it is shown with,
        # read off the recordings rather than composed from a rule: the cupcake
        # and the cola are not in the same order in all three, and a label that
        # tidied them into one would describe an arrangement nobody recorded.
        # `collection_video_candidates` maps these to the folders in order.
        "collection_options": [
            "Cupcake left, cola middle, chocolate milk right",
            "Cola left, chocolate milk middle, cupcake right",
            "Chocolate milk left, cola middle, cupcake right",
        ],
    },
    "prepare_tray": {
        "failure_modes": [
            "The robot fails to slide out the tray.",
            "The robot manages to slide out the tray, but fails to pick up the bowl.",
            "The robot manages to slide out the tray, pick up the bowl, but fails to place the bowl on the tray.",
            "The robot manages to slide out the tray, pick up the bowl, place it, but fails to pick up the fork.",
            "The robot manages to slide out the tray, pick up the bowl, place it, pick up the fork, but fails to place the fork on the tray.",
        ],
        "n_ranked": 3,
        "collection_prompt": "You are in charge of collecting new data to improve the policy. Choose one initial object arrangement for additional data.",
        "collection_options": [
            "Bowl and fork on top of the shelf",
            "Bowl in the middle of the shelf, fork on top of the shelf",
        ],
    },
    "cooking": {
        "failure_modes": [
            "The robot fails to pick up the spoon from the pot.",
            "The robot fails to place the spoon with its head on the towel.",
            "The robot fails to pick up the carrot.",
            "The robot fails to place the carrot in the pot.",
            "The robot fails to pick up the corn.",
            "The robot fails to place the corn in the pot.",
            "The robot fails to pick up the chili.",
            "The robot fails to place the chili in the pot.",
            "The robot fails to pick up the lid.",
            "The robot fails to place the lid on the pot.",
        ],
        "n_ranked": 3,
        "collection_prompt": "You are in charge of collecting new data to improve the policy. Choose one subtask for additional data.",
        "collection_options": [
            "Pick up the carrot and place it in the pot.",
            "Pick up the corn and place it in the pot.",
            "Pick up the lid and place it on the pot.",
        ],
    },
}


def restore_task_answers(
    *, key_prefix: str, question_set: str, responses: dict | None, tlx: dict | None
) -> None:
    """Put a checkpointed answer back into the widgets that asked for it.

    Widget state is addressed by key, and these are the keys
    :func:`render_task_questions` uses — so a participant who resumes a session
    finds the page as they left it rather than blank. It lives here beside the
    rendering for that reason: two copies of these key names would drift, and
    the failure would be a silently un-restored answer.

    Must run before the widgets are created; Streamlit refuses to set the state
    of a widget that has already been instantiated in the same run.

    A stored value that is not in this task's bank is dropped rather than
    written: after an edit to the bank, an answer to the old wording is no
    longer one of the options, and Streamlit raises when a key's value is not
    among them.
    """
    spec = QUESTION_SETS.get(question_set)
    if spec is None:
        return
    stored = responses or {}
    ranked = [
        mode for mode in stored.get("failure_mode_ranking") or [] if mode in spec["failure_modes"]
    ]
    for i in range(spec["n_ranked"]):
        st.session_state[f"{key_prefix}_failure_rank_{i + 1}"] = (
            ranked[i] if i < len(ranked) else None
        )
    choice = stored.get("data_collection_choice")
    if choice in spec["collection_options"]:
        st.session_state[f"{key_prefix}_collection_choice"] = choice
        # Each preview card is a one-option radio, so restoring the pick means
        # selecting that card and clearing the others.
        for index, option in enumerate(spec["collection_options"]):
            st.session_state[f"{key_prefix}_collection_video_{index}"] = (
                option if option == choice else None
            )
    for dimension, score in (tlx or {}).items():
        if dimension in _TLX_ITEMS and score in RATING_POINTS:
            st.session_state[f"{key_prefix}_nasa_tlx_{dimension}"] = score


def render_task_questions(
    *, key_prefix: str, question_set: str, demo_videos_dir: Path | None = None
) -> tuple[dict[str, object], dict[str, int], str | None, bool]:
    """Render ranking/collection questions and NASA-TLX for one modality."""
    spec = QUESTION_SETS[question_set]
    # The definition sits inside the question that uses the term, not above it:
    # "failure mode" is being used in a specific sense -- the step an attempt
    # breaks down at -- and a participant reading the options cannot infer that.
    n_ranked = spec["n_ranked"]
    st.markdown(
        f"**1. Rank the top {COUNT_WORDS.get(n_ranked, n_ranked)} failure modes "
        "that occur in these rollouts.** A failure mode is one specific way an "
        "attempt goes wrong: the step the robot breaks down at, and how far it "
        "got before that. Each option below names one."
    )
    ranked: list[str | None] = []
    rank_keys = [f"{key_prefix}_failure_rank_{i + 1}" for i in range(n_ranked)]
    used_failure_modes: set[str] = set()
    for i in range(n_ranked):
        rank_key = rank_keys[i]
        current = st.session_state.get(rank_key)
        # Repair persisted answers from older sessions before rendering the
        # dropdowns, keeping the first occurrence and clearing later repeats.
        if current in used_failure_modes:
            st.session_state[rank_key] = None
            current = None
        other_selected = {
            value
            for key in rank_keys
            if key != rank_key
            for value in [st.session_state.get(key)]
            if value is not None
        }
        failure_options = [
            mode for mode in spec["failure_modes"] if mode not in other_selected or mode == current
        ]
        choice = st.selectbox(
            f"Failure mode rank {i + 1}",
            options=[None, *failure_options],
            format_func=lambda value: "Select a failure mode…" if value is None else value,
            key=rank_key,
        )
        ranked.append(choice)
        if choice is not None:
            used_failure_modes.add(choice)

    st.markdown(f"**2. {spec['collection_prompt']}**")
    # Curated clips are mounted with the study media. Each clip is a selectable
    # card so the control sits directly beneath the footage it describes.
    previews = [
        (i, option, collection_video_candidates(question_set, i, demo_videos_dir))
        for i, option in enumerate(spec["collection_options"])
    ]
    if any(paths for _, _, paths in previews):
        collection_choice = _render_collection_cards(
            key_prefix=key_prefix,
            question_set=question_set,
            previews=previews,
            all_options=spec["collection_options"],
        )
    else:
        collection_choice = _plain_collection_choice(key_prefix, spec["collection_options"])
    # Not "Task Load": two of the three items measure load and the third is
    # confidence, which is not load at all. What they have in common is that
    # they are all about the act of answering, so the heading says that.
    st.markdown("**3. Follow-up questions**")
    tlx = render_nasa_tlx(key_prefix=f"{key_prefix}_nasa_tlx", items=_TLX_ITEMS)
    complete = all(choice is not None for choice in ranked) and len(set(ranked)) == n_ranked
    complete = complete and collection_choice is not None
    # The task-load ratings are part of the answer now that they can be left
    # blank: before, an untouched slider reported 50 and the page had no way to
    # tell that apart from a chosen 50.
    complete = complete and all(rating is not None for rating in tlx.values())
    if not complete:
        st.warning("Please answer all questions.")
    responses = {
        "failure_mode_ranking": ranked,
        "data_collection_choice": collection_choice,
    }
    return responses, tlx, collection_choice, complete
