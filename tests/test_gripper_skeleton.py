"""Skeleton subtask spans from gripper telemetry: the events, the tiling, the read.

The point of the feature is to save an annotator from placing every boundary by
hand, so what these check is that it never hands back something worse than
nothing: a trace belonging to a different episode, a pill too small to grab, or
a tiling with a hole in it.
"""

from __future__ import annotations

import numpy as np
import pytest

from behavior_graphs.annotations import gripper

h5py = pytest.importorskip("h5py")


def _trace(pattern: list[tuple[int, int]], n: int) -> np.ndarray:
    """Closedness that is 1.0 over each ``[start, end)`` of *pattern*, else 0."""
    out = np.zeros(n, dtype=np.float64)
    for start, end in pattern:
        out[start:end] = 1.0
    return out


def _write_capture(rollout_dir, name: str, values: np.ndarray, key: str | None = None) -> None:
    directory = rollout_dir / gripper.RECORDINGS_DIR / name
    directory.mkdir(parents=True)
    with h5py.File(directory / "trajectory.h5", "w") as handle:
        handle.create_dataset(key or "observation/robot_state/gripper_position", data=values)


# ── Events ───────────────────────────────────────────────────────────────────


def test_a_grasp_and_a_release_are_two_events():
    assert gripper.events(_trace([(40, 80)], 120), min_dwell=5) == [40, 80]


def test_the_event_frame_is_the_first_one_in_the_new_state():
    # Half-open spans: the boundary is where the next subtask *begins*, so a
    # gripper that is open at 39 and closed at 40 puts the edge on 40.
    trace = _trace([(40, 80)], 120)
    assert trace[39] == 0.0 and trace[40] == 1.0
    assert gripper.events(trace, min_dwell=1)[0] == 40


def test_a_rattle_shorter_than_the_dwell_is_not_an_event():
    # Three frames of closedness while the fingers settle, then the real grasp.
    trace = _trace([(20, 23), (40, 80)], 120)
    assert gripper.events(trace, min_dwell=8) == [40, 80]


def test_a_rejected_rattle_leaves_the_state_where_it_was():
    # The failure a refractory period would have: accept the rattle's first edge,
    # suppress its second, and end up believing the gripper is closed from 20 on
    # — which loses the boundary at the grasp that follows.
    assert 20 not in gripper.events(_trace([(20, 23), (40, 80)], 120), min_dwell=8)


def test_a_release_on_the_final_frames_counts_however_short_it_is():
    # It ran out of episode to persist into, not out of conviction.
    assert gripper.events(_trace([(40, 117)], 120), min_dwell=8) == [40, 117]


def test_a_threshold_a_grasp_never_reaches_finds_nothing():
    trace = _trace([(40, 80)], 120) * 0.4
    assert gripper.events(trace, closed_above=0.5, min_dwell=5) == []
    assert gripper.events(trace, closed_above=0.3, min_dwell=5) == [40, 80]


def test_a_trace_too_short_to_have_an_edge():
    assert gripper.events(np.ones(1), min_dwell=1) == []
    assert gripper.events(np.empty(0), min_dwell=1) == []


# ── Spans ────────────────────────────────────────────────────────────────────


def test_spans_tile_the_whole_episode():
    out = gripper.spans(120, [40, 80], min_span=5)
    assert out == [(0, 40), (40, 80), (80, 120)]


def test_no_events_is_one_span_over_the_episode():
    assert gripper.spans(120, [], min_span=5) == [(0, 120)]


def test_a_boundary_too_close_to_the_start_is_dropped():
    assert gripper.spans(120, [3, 60], min_span=8) == [(0, 60), (60, 120)]


def test_a_boundary_too_close_to_the_end_is_dropped():
    assert gripper.spans(120, [60, 117], min_span=8) == [(0, 60), (60, 120)]


def test_two_boundaries_closer_than_the_minimum_collapse_to_one():
    assert gripper.spans(120, [60, 63], min_span=8) == [(0, 60), (60, 120)]


def test_dropping_a_boundary_never_leaves_a_hole():
    out = gripper.spans(200, [3, 60, 63, 120, 196], min_span=8)
    assert out[0][0] == 0 and out[-1][1] == 200
    assert all(a[1] == b[0] for a, b in zip(out, out[1:], strict=False))


def test_an_episode_shorter_than_one_span_proposes_nothing():
    assert gripper.spans(4, [2], min_span=8) == []


def test_duplicate_and_unsorted_boundaries_are_tolerated():
    assert gripper.spans(120, [80, 40, 80], min_span=5) == [(0, 40), (40, 80), (80, 120)]


# ── The whole proposal ───────────────────────────────────────────────────────


def test_skeleton_converts_the_dwell_from_seconds_using_fps():
    # 0.5 s is 8 frames at 15 Hz, so a 5-frame rattle is out; at 2 Hz it is one
    # frame, so the same rattle is in.
    trace = _trace([(20, 25), (40, 80)], 120)
    assert gripper.skeleton(trace, 120, fps=15.0, min_dwell_sec=0.5) == [
        (0, 40),
        (40, 80),
        (80, 120),
    ]
    assert (20, 25) in gripper.skeleton(trace, 120, fps=2.0, min_dwell_sec=0.5)


def test_skeleton_of_an_episode_the_gripper_never_moved_on():
    assert gripper.skeleton(np.zeros(120), 120, fps=15.0) == [(0, 120)]


# ── Reading the telemetry ────────────────────────────────────────────────────


def test_trace_reads_the_capture_at_the_episode_s_sorted_position(tmp_path):
    _write_capture(tmp_path, "ep_20260804T100000", np.full(10, 0.1))
    _write_capture(tmp_path, "ep_20260804T110000", np.full(20, 0.9))
    assert gripper.trace(tmp_path, 0, num_frames=10)[0] == pytest.approx(0.1)
    assert gripper.trace(tmp_path, 1, num_frames=20)[0] == pytest.approx(0.9)


def test_trace_refuses_a_capture_of_the_wrong_length(tmp_path):
    # What an assembly that skipped an incomplete episode looks like: the sorted
    # positions shift, and every index after the gap reads someone else's
    # telemetry. Offering nothing beats putting every boundary in the wrong
    # place.
    _write_capture(tmp_path, "ep_20260804T100000", np.full(10, 0.1))
    assert gripper.trace(tmp_path, 0, num_frames=99) is None


def test_trace_without_the_length_check_takes_what_it_finds(tmp_path):
    _write_capture(tmp_path, "ep_20260804T100000", np.full(10, 0.1))
    assert gripper.trace(tmp_path, 0).size == 10


def test_trace_is_none_for_a_dataset_with_no_raw_captures(tmp_path):
    assert gripper.trace(tmp_path, 0) is None


def test_trace_is_none_past_the_last_capture(tmp_path):
    _write_capture(tmp_path, "ep_20260804T100000", np.full(10, 0.1))
    assert gripper.trace(tmp_path, 7) is None


def test_trace_is_none_when_the_capture_records_no_gripper(tmp_path):
    _write_capture(tmp_path, "ep_0", np.zeros((10, 7)), key="observation/robot_state/joint")
    assert gripper.trace(tmp_path, 0) is None


def test_trace_prefers_the_measured_state_over_the_commanded_action(tmp_path):
    # The command leads the measurement by however long the gripper takes to
    # close, so reading it would put every boundary a few frames early.
    directory = tmp_path / gripper.RECORDINGS_DIR / "ep_0"
    directory.mkdir(parents=True)
    with h5py.File(directory / "trajectory.h5", "w") as handle:
        handle.create_dataset("action/gripper_position", data=np.full(10, 0.9))
        handle.create_dataset("observation/robot_state/gripper_position", data=np.full(10, 0.1))
    assert gripper.trace(tmp_path, 0)[0] == pytest.approx(0.1)


def test_trace_falls_back_to_the_commanded_action(tmp_path):
    _write_capture(tmp_path, "ep_0", np.full(10, 0.9), key="action/gripper_position")
    assert gripper.trace(tmp_path, 0)[0] == pytest.approx(0.9)


def test_trace_is_none_for_an_unreadable_capture(tmp_path):
    directory = tmp_path / gripper.RECORDINGS_DIR / "ep_0"
    directory.mkdir(parents=True)
    (directory / "trajectory.h5").write_bytes(b"not an hdf5 file")
    assert gripper.trace(tmp_path, 0) is None
