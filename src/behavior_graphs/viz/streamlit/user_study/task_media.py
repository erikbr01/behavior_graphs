"""Participant-facing media helpers for the study task pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from behavior_graphs.viz.streamlit.user_study.video_browser import shuffled_numbers

#: Rendered width of the task-progression clip, in pixels.
#:
#: Also the cap for the data-collection example clips, which sit further down
#: the same page: the study footage is all 16:9, so one shared width is one
#: shared height, and an example in a wide column no longer towers over the
#: clip of the task it is an example for.
TASK_VIDEO_WIDTH = 576


def _display_numbered_episode(
    index: dict, mp4_dir: Path, *, display_number: int
) -> tuple[int, Path] | None:
    """Return the video the rollout grid numbers *display_number*.

    The numbering comes from :func:`~behavior_graphs.viz.streamlit.user_study
    .video_browser.shuffled_numbers`, so this is the same clip the grid labels
    with that number, and the same clip for every participant.
    """
    episodes = index.get("episodes", [])
    numbers = shuffled_numbers(episodes)
    episode_index = next(
        (episode_id for episode_id, number in numbers.items() if number == display_number), None
    )
    if episode_index is None:
        return None
    for episode in episodes:
        if episode.get("index") != episode_index:
            continue
        # The page introduces this clip as a successful execution of the task,
        # so a number that lands on a failure is not shown: the caller falls
        # back to a rollout that did succeed. Worth guarding rather than
        # trusting the constant — cooking is 14 successes in 60 rollouts, so
        # most numbers are failures.
        if episode.get("success") is not True:
            return None
        path = mp4_dir / str(episode.get("path", ""))
        if path.is_file():
            return int(episode["index"]), path
    return None


def _success_video(index: dict, mp4_dir: Path) -> tuple[int, Path] | None:
    """Return the first successful, readable episode video."""
    for episode in index.get("episodes", []):
        if episode.get("success") is not True:
            continue
        path = mp4_dir / str(episode.get("path", ""))
        if path.is_file():
            return int(episode["index"]), path
    return None


def render_task_progression(
    task_key: str,
    index: dict,
    mp4_dir: Path,
    *,
    display_number: int | None = None,
) -> None:
    """Show one complete successful rollout beneath the caller's task text."""
    selected = None
    if display_number is not None:
        selected = _display_numbered_episode(index, mp4_dir, display_number=display_number)
    if selected is None:
        selected = _success_video(index, mp4_dir)
    if selected is None:
        st.caption("Task progression preview will appear when a successful rollout is mounted.")
        return
    _episode_idx, path = selected
    # A fixed pixel width still sits at the left edge of whatever holds it, so
    # padding with empty side columns only centered the 576px *slot* — the
    # video inside it drifted left as the window grew. Center the container.
    with st.container(horizontal_alignment="center"):
        st.video(str(path), width=TASK_VIDEO_WIDTH)


def collection_video_candidates(
    task_key: str, option_index: int, demo_videos_dir: Path | None
) -> list[Path]:
    """Find the curated DROID clips for one data-collection option."""
    if demo_videos_dir is None or not demo_videos_dir.is_dir():
        return []
    folder_by_task = {
        "prepare_tray": ("prepare_tray_top", "prepare_tray_bottom"),
        "cooking": ("cooking_carrot", "cooking_corn", "cooking_lid"),
    }
    folder: str | None = None
    if task_key in folder_by_task:
        folders = folder_by_task[task_key]
        if option_index < len(folders):
            folder = folders[option_index]
    elif task_key == "set_table":
        # One option per chocolate-milk start position, in the order the
        # question bank lists them, and the curated footage is organized by that
        # position — so this is the identity mapping and not a lookup table that
        # can drift out of step with the options. It did: the six-option version
        # mapped "milk middle" to the right-hand footage and "milk left" to the
        # middle, so two of the six illustrated an arrangement other than the
        # one they described.
        positions = ("right", "middle", "left")
        if option_index < len(positions):
            folder = f"prepare_plate_milk_{positions[option_index]}"
    if folder is None:
        return []
    return sorted((demo_videos_dir / folder).glob("*.mp4"))
