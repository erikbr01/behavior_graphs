"""The rollout-video browser: one paginated grid of players.

Episodes used to be one collapsed ``st.expander`` each, so the step that asks a
participant to "watch several rollouts" made them open and close players one at
a time — and nothing was visible until they did. A grid puts a whole page of
players on screen at once, which is the comparison the study is actually after.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import streamlit as st

from behavior_graphs.viz.streamlit.clips import prepare_entry_clips
from behavior_graphs.viz.streamlit.components.camera_views import player_views
from behavior_graphs.viz.streamlit.components.mp4_player import mp4_player
from behavior_graphs.viz.streamlit.components.pager import page_slice

# Tall enough to read the arm's motion at half the page width.
_CELL_HEIGHT_PX = 260
#: Cells across a full-width grid. Two, because a cell holds every camera view
#: of its rollout side by side: at three cells across, each view was a sixth of
#: the page and the arm was a few pixels wide.
_GRID_COLUMNS = 2


def filter_episodes(episodes: Sequence[dict], choice: str) -> list[dict]:
    """The subset of ``episodes`` the "Filter episodes" radio selects."""
    if choice == "Success only":
        return [ep for ep in episodes if ep.get("success") is True]
    if choice == "Failure only":
        return [ep for ep in episodes if ep.get("success") is False]
    return list(episodes)


def episode_status(episode: dict) -> str:
    """Short success/failure label for an episode's caption."""
    success = episode.get("success")
    if success is True:
        return "✓ Success"
    if success is False:
        return "✗ Failure"
    return "Unknown"


def shuffled_numbers(episodes: Sequence[dict]) -> dict[int, int]:
    """``{episode index: display number}`` over a permutation of 1..N.

    Shuffled so a clip's number says nothing about when it was recorded, and
    *derived*, not drawn: each episode's place in the permutation comes from a
    hash of its own index, so the same episode carries the same number for every
    participant and in every session.

    It used to be a ``random.sample`` per session, kept in session state so at
    least paging the grid did not renumber it. That made "rollout 19" a
    different clip for each participant — including the one the task page pins
    as its worked example (see :mod:`~behavior_graphs.viz.streamlit.user_study
    .task_media`) — so two participants writing about a numbered rollout were
    not writing about the same rollout.
    """
    ids = [ep["index"] for ep in episodes]
    ordered = sorted(ids, key=lambda ep_id: hashlib.blake2b(str(ep_id).encode()).digest())
    return {ep_id: number for number, ep_id in enumerate(ordered, start=1)}


def render_video_browser(
    mp4_dir: Path,
    index: dict,
    page_size: int = 4,
    key_prefix: str = "vbrow",
    show_true_index: bool = True,
    lead_seconds: float = 0.0,
    show_camera_labels: bool = True,
    shuffle_indices: bool = False,
) -> None:
    """The grid of rollout clips, paged and filterable.

    ``lead_seconds``, if set, opens every clip that many seconds before its end
    rather than at the start: what happened is decided at the end, and on these
    datasets that is a minute of working behaviour before the part worth
    watching. Every clip, not just the failures — where a clip opens would
    otherwise announce its outcome before it is watched. Saying so is the
    caller's job: a clip that starts mid-task reads as a rollout that started
    mid-task, and the sentence belongs with the rest of what the page says about
    the clips rather than as a note of its own above the grid.

    ``shuffle_indices`` draws each clip's displayed number from a permutation
    rather than numbering them 1..N down the page, so the number carries no
    ordering: sequential numbers tell a participant which clips were recorded
    next to each other, and "episode 3" beside "episode 4" reads as two related
    runs. Only meaningful with ``show_true_index=False``, which is what puts a
    display number on a clip at all. The permutation is derived from the episode
    indices rather than drawn, so a clip keeps its number across pages, sessions
    and participants — see :func:`shuffled_numbers`.
    """
    episodes = index["episodes"]
    # Fixed by position in the (possibly shuffled) full list, so a given
    # episode's display number stays the same across filter views.
    display_numbers = {ep["index"]: i for i, ep in enumerate(episodes, start=1)}
    if shuffle_indices:
        display_numbers = shuffled_numbers(episodes)

    filter_key = f"{key_prefix}_filter"
    page_key = f"{key_prefix}_page"

    filter_choice = st.radio(
        "Filter episodes",
        options=["All", "Success only", "Failure only"],
        horizontal=True,
        key=filter_key,
    )

    filtered = filter_episodes(episodes, filter_choice)

    if st.session_state.get(f"{key_prefix}_last_filter") != filter_choice:
        st.session_state[page_key] = 0
        st.session_state[f"{key_prefix}_last_filter"] = filter_choice

    page_episodes, current_page, total_pages = page_slice(filtered, page_size, page_key)
    start = current_page * page_size

    def _turn_to(page: int) -> None:
        st.session_state[page_key] = page

    def _nav_row() -> None:
        # One row, above the grid, beside the filter it works with. There used
        # to be a second copy below the clips; at four per page in two columns
        # the whole grid is about a screen, so the copy saved no scrolling and
        # cost a second pair of controls doing the same thing.
        #
        # Full-width buttons in the outer columns, so "← Prev" and "Next →" sit
        # at the two ends of the row rather than each at the left edge of its
        # own column, and the row centres them against the label itself instead
        # of a padding-top guessed at the height of a button.
        col_prev, col_label, col_next = st.columns([1, 3, 1], vertical_alignment="center")
        with col_prev:
            # on_click, not a post-hoc st.rerun: the callback lands before the
            # fragment re-runs, so the grid below is already on the new page —
            # and only the fragment re-runs, not the whole study step.
            st.button(
                "← Prev",
                key=f"{key_prefix}_prev",
                disabled=current_page == 0,
                on_click=_turn_to,
                args=(current_page - 1,),
                use_container_width=True,
            )
        with col_label:
            shown = (
                f"{start + 1}–{start + len(page_episodes)} of {len(filtered)}"
                if page_episodes
                else "none"
            )
            st.markdown(
                f"<div style='text-align:center;'>"
                f"Page {current_page + 1} / {total_pages}"
                f"<span style='color:#888;font-size:0.85em;'> &nbsp;·&nbsp; "
                f"episodes {shown}</span></div>",
                unsafe_allow_html=True,
            )
        with col_next:
            st.button(
                "Next →",
                key=f"{key_prefix}_next",
                disabled=current_page >= total_pages - 1,
                on_click=_turn_to,
                args=(current_page + 1,),
                use_container_width=True,
            )

    _nav_row()

    if not page_episodes:
        st.info("No episodes match this filter.")
        return

    # Transcode this page's clips concurrently behind one spinner rather than
    # one at a time as each player renders — every camera view of them, since
    # that is what the players put on the page.
    prepare_entry_clips(page_episodes, mp4_dir)

    for row_start in range(0, len(page_episodes), _GRID_COLUMNS):
        row = page_episodes[row_start : row_start + _GRID_COLUMNS]
        # Always ``_GRID_COLUMNS`` wide, even for a short last row, so its cells
        # stay the same size as the rows above instead of stretching.
        cols = st.columns(_GRID_COLUMNS)
        for col, ep in zip(cols, row):
            label = ep["index"] if show_true_index else display_numbers[ep["index"]]
            # Bordered cell: without it a caption sits closer to the video of
            # the row above than to its own, so the labels read off by one row.
            with col.container(border=True):
                st.caption(f"**Episode {label}** · {episode_status(ep)}")
                mp4_player(
                    **player_views(ep, mp4_dir, index),
                    label="",
                    key=f"{key_prefix}_ep{ep['index']}",
                    max_height_px=_CELL_HEIGHT_PX,
                    show_camera_labels=show_camera_labels,
                    seek_from_end_s=lead_seconds or None,
                    total_frames=ep.get("frame_count"),
                )
