"""Which rollout the study pages show as "the task".

The clip sits under a sentence that calls it a successful execution of the task,
and it is pinned by the number the rollout grid gives it — so it has to be the
rollout carrying that number, and it has to be one that succeeded. Neither held
while the grid's numbering was drawn per session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from behavior_graphs.viz.streamlit.user_study.task_media import (  # noqa: E402
    _display_numbered_episode,
)
from behavior_graphs.viz.streamlit.user_study.video_browser import (  # noqa: E402
    shuffled_numbers,
)


def _index(tmp_path: Path, outcomes: dict[int, bool]) -> dict:
    episodes = []
    for ep_id, ok in outcomes.items():
        name = f"ep{ep_id:04d}.mp4"
        (tmp_path / name).write_bytes(b"")
        episodes.append({"index": ep_id, "success": ok, "path": name, "views": {}})
    return {"episodes": episodes}


def test_the_same_episode_carries_the_same_number_every_time():
    """A permutation, but a derived one — not a fresh draw per session.

    Sixty episodes rather than a handful: with four, a random permutation
    repeating itself is a one-in-24 coincidence, which is not a test.
    """
    episodes = [{"index": i} for i in range(60)]

    numbers = shuffled_numbers(episodes)

    assert shuffled_numbers([dict(ep) for ep in episodes]) == numbers
    assert sorted(numbers.values()) == list(range(1, 61))
    # Shuffled, or the number says which rollouts were recorded side by side.
    assert [numbers[i] for i in range(60)] != list(range(1, 61))


def test_the_numbering_does_not_depend_on_the_order_episodes_arrive_in():
    forwards = shuffled_numbers([{"index": i} for i in range(20)])
    backwards = shuffled_numbers([{"index": i} for i in reversed(range(20))])

    assert forwards == backwards


def test_the_pinned_clip_is_the_one_the_grid_gives_that_number(tmp_path):
    outcomes = {i: True for i in range(12)}
    numbers = shuffled_numbers([{"index": i} for i in outcomes])
    wanted = next(ep_id for ep_id, number in numbers.items() if number == 5)

    selected = _display_numbered_episode(_index(tmp_path, outcomes), tmp_path, display_number=5)

    assert selected is not None
    assert selected[0] == wanted


def test_a_number_that_lands_on_a_failure_is_declined(tmp_path):
    """The caller falls back to a successful rollout rather than showing this.

    Cooking is 14 successes in 60 rollouts, so most numbers land on a failure —
    and the study page introduces the clip as the task being done.
    """
    outcomes = {i: True for i in range(12)}
    numbers = shuffled_numbers([{"index": i} for i in outcomes])
    doomed = next(ep_id for ep_id, number in numbers.items() if number == 5)
    outcomes[doomed] = False

    assert _display_numbered_episode(_index(tmp_path, outcomes), tmp_path, display_number=5) is None


def test_a_number_no_rollout_carries_is_declined(tmp_path):
    outcomes = {i: True for i in range(4)}

    assert _display_numbered_episode(_index(tmp_path, outcomes), tmp_path, display_number=9) is None
