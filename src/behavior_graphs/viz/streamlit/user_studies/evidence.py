"""What a participant looks at: the rollout videos, or the behavior graph.

Every study variant since the order-effect mockup shows the same two displays
of the same rollouts, introduced with the same words, and then asks its own
questions about them. That shared half lives here so the variants cannot drift
into showing participants subtly different pages — a difference in the copy
above the evidence is a difference in the stimulus, which is the one thing a
between-displays comparison cannot afford.

Variants own their step machines, question banks and response shapes; they call
in here for the evidence itself.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import streamlit as st

from behavior_graphs.core.node_labels import load_node_labels
from behavior_graphs.viz.streamlit.components.graph_config import tree_kwargs_from_config
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import render_trajectory_tree
from behavior_graphs.viz.streamlit.user_study.graph_intro import render_graph_intro
from behavior_graphs.viz.streamlit.user_study.task_media import render_task_progression
from behavior_graphs.viz.streamlit.user_study.video_browser import (
    render_video_browser,
    shuffled_numbers,
)

#: The rollout the cooking grid numbers 20, so a participant who wants to look
#: again can find it there. 19 while the numbering was drawn per session, which
#: made it a different clip for everybody; with the permutation fixed, 19
#: resolves to a rollout that fails, and the sentence above the video says it
#: succeeds.
_COOKING_SUCCESS_DISPLAY_NUMBER = 20


def render_task_restatement(
    *,
    phase: str,
    task_description: str,
    index: dict,
    mp4_dir: Path,
    graph_params: dict,
    question_set: str | None,
) -> None:
    """The task, how much evidence there is, and one successful rollout.

    Passed as ``before_evidence`` to :func:`render_phase_view`, so it is the
    last thing a participant reads before looking at the display: the questions
    ask what the robot got wrong, which is unanswerable without knowing what it
    was supposed to do, and the intro page carrying that is several clicks
    behind.
    """
    if not task_description:
        return
    # How much evidence it is, in the same breath. A participant asked which
    # failure modes "occur in these rollouts" is answering about a sample, and
    # 60 rollouts and 6 would warrant different confidence.
    n = len(index.get("episodes") or ())
    scope = ""
    if n and phase == "graph":
        scope = f" The graph below is built from {n} rollouts of this policy."
    elif n:
        scope = f" The clips below are {n} rollouts of this policy."
        # Said here rather than over the grid: a clip that starts mid-task
        # reads as a rollout that started mid-task, and this is where the page
        # already says what the clips are.
        lead = float(graph_params.get("video_lead_seconds", 0) or 0)
        if lead:
            scope += f" Each clip opens {lead:g} seconds before it ends."
    video_note = (
        " The video below shows a successful execution of the task." if phase == "videos" else ""
    )
    st.markdown(f"**The task.** {task_description}{scope}{video_note}")
    render_task_progression(
        task_key=question_set or "",
        index=index,
        mp4_dir=mp4_dir,
        display_number=_COOKING_SUCCESS_DISPLAY_NUMBER if question_set == "cooking" else None,
    )


def render_rollout_grid_intro() -> None:
    """The grid's key, opposite ``render_graph_intro``.

    A list rather than an annotated figure, which is the one place the two are
    still not matched in weight. It earns some of that back by naming the
    controls exactly as the interface labels them, which a drawn figure cannot:
    "Filter episodes", "Prev" and "Next" are the words on the page below.
    """
    st.subheader("How to read the rollout grid")
    st.markdown(
        "- Each cell is one rollout, captioned with its number and its outcome.\n"
        "- The two clips in a cell are the same attempt from the workspace "
        "camera and the wrist camera.\n"
        "- **Filter episodes** narrows the grid to the successes or the failures.\n"
        "- The grid shows four rollouts at a time. **Prev** and **Next** move "
        "through all of them.\n"
        "- Each clip opens shortly before the attempt ends, where the outcome is."
    )


def _display_numbers(
    index: dict, graph_params: dict, hide_episode_index: bool
) -> dict[int, int] | None:
    """What number to print on each clip, or ``None`` to print the real index.

    Mirrors :func:`~behavior_graphs.viz.streamlit.user_study.video_browser
    .render_video_browser`: a permutation only when the study both hides the
    true indices and asks for shuffled ones, because a display number is what
    replaces the real index rather than something shown beside it.
    """
    if not (hide_episode_index and graph_params.get("shuffle_rollout_indices", False)):
        return None
    return shuffled_numbers(index["episodes"])


def render_phase_view(
    phase: str,
    pfx: str,
    index: dict,
    mp4_dir: Path,
    labels,
    metadata: list[dict],
    hide_episode_index: bool,
    graph_params: dict,
    run_dir_str: str | None,
    before_evidence: Callable[[], None] | None = None,
) -> None:
    """*before_evidence* runs after the explanatory text and immediately above
    the videos/graph, which is the last thing a participant reads before
    looking at the evidence.
    """
    if phase == "videos":
        st.header("Rollout Videos")
        # The same three pieces of framing the graph gets below: what the unit
        # is, how to read the display, and a heading over the artifact itself.
        # Without them the two displays are not comparable as displays — one
        # arrives explained and the other as a folder of files, which tells a
        # participant which of the two the study is about before they have read
        # either. It does not close that gap (a behavior graph is unfamiliar and
        # video is not) but it stops the page widening it.
        st.markdown(
            "Each **rollout** (one trial of the policy attempting the task) is "
            "recorded from two cameras, one facing the workspace and one on the "
            "wrist. The grid shows every rollout on its own, labelled with the "
            "outcome it ended in, **success ✓** or **failure ✗**, so an attempt "
            "can be watched from start to finish, and the successes and the "
            "failures can be looked at separately."
        )
        render_rollout_grid_intro()
        st.subheader("This policy's rollouts")
        if before_evidence is not None:
            before_evidence()
        render_video_browser(
            mp4_dir,
            index,
            page_size=4,
            key_prefix=f"{pfx}_vbrow",
            show_true_index=not hide_episode_index,
            lead_seconds=float(graph_params.get("video_lead_seconds", 0)),
            shuffle_indices=graph_params.get("shuffle_rollout_indices", False),
            show_camera_labels=graph_params.get("show_camera_labels", True),
        )
    else:  # "graph"
        st.header("Behavior Graph")
        st.markdown(
            "Each **rollout** (one trial of the policy attempting the task) is "
            "decomposed into **behaviors**: the distinct steps the robot takes as it "
            "tries to accomplish the task. The graph aggregates behaviors from all "
            "rollouts "
            "and shows in what order they happened, where rollouts branch away from "
            "each other, and how likely each branch is to end in **success ✓** or "
            "**failure ✗**."
        )
        _node_vlm_labels = load_node_labels(Path(run_dir_str)) if run_dir_str else {}
        _cluster_names = {nid: v["label"] for nid, v in _node_vlm_labels.items()}
        render_graph_intro()
        # Says which graph the reader is now looking at: without it the example
        # above and the real thing below run together as one picture.
        st.subheader("This policy's behavior graph")
        if before_evidence is not None:
            before_evidence()

        # Scoped to a fragment: render_trajectory_tree issues
        # st.rerun(scope="fragment") internally when closing a panel, which
        # raises outside an @st.fragment-decorated function (see the matching
        # comment in app_group_b.py).
        @st.fragment
        def _dispatch() -> None:
            render_trajectory_tree(
                labels=labels,
                metadata=metadata,
                # The session's `graph:` block, applied the same way wherever
                # this graph is drawn — see `components/graph_config.py`.
                **tree_kwargs_from_config(graph_params),
                cluster_names=_cluster_names,
                node_vlm_labels=_node_vlm_labels or None,
                mp4_dir=mp4_dir,
                mp4_index=index,
                # The same numbers the rollout grid puts on the same clips.
                # Without this the graph captioned its clips with the true
                # episode index while the grid showed a shuffled one, so the
                # two displays disagreed about what a clip was called and the
                # graph handed back the ordering `shuffle_rollout_indices`
                # exists to withhold.
                episode_numbers=_display_numbers(index, graph_params, hide_episode_index),
                level="rollout",
                key_prefix=f"{pfx}_tree",
            )

        _dispatch()
