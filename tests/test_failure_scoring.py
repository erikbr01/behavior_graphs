"""Scoring a failure-mode ranking against the failures that actually happened.

The metrics are the point of the module, so most of this is about what each one
says and — more importantly — about the two that would be *wrong together* if
either were dropped: coverage does not care what order the picks were in, and
the order score does not care whether the picks were the right ones.

One test here is a regression test rather than a specification: the study
collects anonymously, so every response in the store identifies its participant
as ``"anon"``, and anything that groups rankings back together by participant
silently merges all of them.
"""

from __future__ import annotations

import json

import pytest

from behavior_graphs.viz.streamlit.user_study import failure_scoring as fs

#: Modes short enough to read in an assertion. The real ones are sentences.
CARROT, LID, CORN, SPOON = "carrot", "lid", "corn", "spoon"

#: 46 failures, as in the cooking rollout: one dominant mode, a clear second,
#: then a tail.
TRUTH = {CARROT: 20, LID: 15, CORN: 6, SPOON: 5}
BANK = [CARROT, LID, CORN, SPOON, "pot", "towel", "plate", "bowl"]


def _response(task: str, ranked: list[str], modality: str = "graph", **extra) -> dict:
    return {
        "_response_id": extra.pop("response_id", "r1"),
        "stage_modalities": {task: modality},
        "understanding_responses": {task: {"failure_mode_ranking": ranked}},
        **extra,
    }


# ── What each number says ────────────────────────────────────────────────────


def test_the_best_possible_picks_score_at_the_ceiling():
    scored = fs.score([CARROT, LID], TRUTH, BANK)

    assert scored.top1_hit is True
    assert scored.coverage == pytest.approx(35 / 46)
    assert scored.achievable == pytest.approx(35 / 46)
    assert scored.coverage_ratio == pytest.approx(1.0)
    assert scored.order_tau == pytest.approx(1.0)


def test_coverage_ignores_the_order_and_the_order_score_does_not():
    """Both, because either alone scores one of these two rankings wrongly: the
    same two modes named in the wrong order is a reader who found the failures
    and misjudged which dominates, and that is not the same mistake as naming
    the wrong two."""
    right_order = fs.score([CARROT, LID], TRUTH, BANK)
    wrong_order = fs.score([LID, CARROT], TRUTH, BANK)
    wrong_modes = fs.score([CORN, SPOON], TRUTH, BANK)

    assert wrong_order.coverage == right_order.coverage
    assert wrong_order.order_tau == pytest.approx(-1.0)
    assert wrong_order.top1_hit is False

    assert wrong_modes.coverage < right_order.coverage
    # And its order is right, which is exactly why coverage has to be reported
    # too: it found the tail and ranked the tail correctly.
    assert wrong_modes.order_tau == pytest.approx(1.0)


def test_chance_is_the_share_of_the_bank_a_pick_covers():
    """Every mode is equally likely to be in a random pick of k, and the shares
    sum to one, so the expected coverage is exactly k over the bank size. It is
    the floor a task's score has to be read against: ten modes and four modes
    are very different questions."""
    assert fs.score([CARROT], TRUTH, BANK).chance == pytest.approx(1 / 8)
    assert fs.score([CARROT, LID, CORN], TRUTH, BANK).chance == pytest.approx(3 / 8)
    # No bank given: no claim about chance rather than a made-up one.
    assert fs.score([CARROT], TRUTH).chance == 0.0


def test_a_tie_for_the_most_common_failure_counts_for_either():
    truth = {CARROT: 10, LID: 10, CORN: 1}

    assert fs.score([CARROT], truth, BANK).top1_hit is True
    assert fs.score([LID], truth, BANK).top1_hit is True
    assert fs.score([CORN], truth, BANK).top1_hit is False


def test_there_is_no_order_to_score_when_there_is_only_one_pick():
    assert fs.score([CARROT], TRUTH, BANK).order_tau is None


def test_there_is_no_order_to_score_when_every_pick_is_equally_common():
    """``None``, not zero: zero would read as a participant who ordered them
    badly, and there was no order to get right."""
    truth = {CARROT: 5, LID: 5, CORN: 5}

    assert fs.score([CARROT, LID], truth, BANK).order_tau is None


def test_a_mode_picked_twice_cannot_take_coverage_above_everything():
    """The study will not allow it, and there is no reading of this figure in
    which coverage over 100% is not a bug."""
    scored = fs.score([CARROT, CARROT], TRUTH, BANK)

    assert scored.picks == 1
    assert scored.coverage == pytest.approx(20 / 46)


def test_nothing_to_score_is_not_a_score_of_zero():
    """A participant who answered nothing, or a dataset with no annotated
    failures, is absent from the metric rather than bottom of it."""
    assert fs.score([], TRUTH, BANK) is None
    assert fs.score([None, ""], TRUTH, BANK) is None
    assert fs.score([CARROT], {}, BANK) is None


def test_the_dominant_share_says_when_a_ranking_had_nothing_to_discriminate():
    assert fs.dominant_share({CARROT: 27, LID: 1}) == pytest.approx(27 / 28)
    assert fs.dominant_share({}) == 0.0


# ── Reading the responses ────────────────────────────────────────────────────


def test_two_anonymous_responses_are_two_rankings_and_not_one():
    """The regression test. The study collects anonymously, so every response
    in the store says ``participant_id`` is nothing at all; grouping rankings
    back together by participant merges every session into one per task and
    concatenates their picks. It does not fail loudly either — it reports
    coverage above 100%, which is how it was caught."""
    responses = [
        _response("cooking", [CARROT, LID], response_id="a"),
        _response("cooking", [CORN, SPOON], response_id="b"),
    ]

    rows = fs.score_rows(responses, {"cooking": TRUTH}, {"cooking": BANK})

    assert len(rows) == 2
    assert all(row["coverage"] <= 1.0 for row in rows)
    assert sorted(row["coverage"] for row in rows) == pytest.approx([11 / 46, 35 / 46])


def test_each_ranking_carries_the_display_its_task_was_read_in():
    """Which is the comparison the study exists to make, so it comes from the
    same resolution every other panel on the page uses."""
    responses = [
        _response("cooking", [CARROT], modality="graph", response_id="a"),
        _response("cooking", [CARROT], modality="videos", response_id="b"),
    ]

    rows = fs.score_rows(responses, {"cooking": TRUTH})

    assert {row["phase"] for row in rows} == {"graph", "videos"}


def test_a_task_with_no_annotations_is_skipped_rather_than_scored_against_nothing():
    rows = fs.score_rows([_response("set_table", [CARROT])], {"cooking": TRUTH})

    assert rows == []


def test_the_layouts_older_flat_block_does_not_take_the_panel_down():
    """An early build wrote ``understanding_responses`` flat, so a value there
    can be the answer rather than a task's answers."""
    flat = {
        "_response_id": "old",
        "understanding_responses": {"failure_mode_ranking": [CARROT, LID]},
    }

    assert fs.score_rows([flat], {"cooking": TRUTH}) == []


# ── Summarising ──────────────────────────────────────────────────────────────


def test_a_summary_averages_within_a_task_and_a_display():
    rows = fs.score_rows(
        [
            _response("cooking", [CARROT, LID], modality="graph", response_id="a"),
            _response("cooking", [CORN, SPOON], modality="graph", response_id="b"),
            _response("cooking", [CARROT, LID], modality="videos", response_id="c"),
        ],
        {"cooking": TRUTH},
        {"cooking": BANK},
    )

    summary = {(row["task"], row["phase"]): row for row in fs.summarise(rows)}

    assert summary[("cooking", "graph")]["n"] == 2
    assert summary[("cooking", "graph")]["top1_hit_rate"] == pytest.approx(0.5)
    assert summary[("cooking", "videos")]["coverage"] == pytest.approx(35 / 46)


def test_the_order_score_is_averaged_over_the_rankings_that_have_one():
    """A ranking with no order to get right contributes nothing to the mean
    rather than a zero, and the count it was taken over is reported so a reader
    can see how much the number rests on."""
    truth = {CARROT: 5, LID: 5}
    rows = fs.score_rows(
        [
            _response("cooking", [CARROT, LID], response_id="tied"),
            _response("cooking", [CARROT], response_id="single"),
        ],
        {"cooking": truth},
    )

    (summary,) = fs.summarise(rows, by=("task",))

    assert summary["n"] == 2
    assert summary["order_tau"] is None
    assert summary["order_tau_n"] == 0


# ── The ground truth on disk ─────────────────────────────────────────────────


def test_the_ground_truth_is_counted_off_the_annotation_store(tmp_path):
    store = tmp_path / "annotations" / "failure_modes"
    store.mkdir(parents=True)
    for episode, mode in enumerate([CARROT, CARROT, LID]):
        (store / f"ep{episode:04d}.json").write_text(
            json.dumps({"schema_version": 1, "episode": episode, "mode": mode})
        )

    assert fs.ground_truth(tmp_path) == {CARROT: 2, LID: 1}


def test_a_dataset_with_no_annotations_has_no_ground_truth(tmp_path):
    assert fs.ground_truth(tmp_path) == {}


def test_each_anonymous_session_is_scored_as_its_own_participant():
    """The reason this module reads responses rather than `analytics.ranking_rows`
    was that the two disagreed about who a response came from, and every
    anonymous session scored under "anon". They agree now, and this is what
    would have caught the coverage-above-100% that found the original bug."""
    from behavior_graphs.viz.streamlit.user_study import failure_scoring

    responses = [
        {
            "session_id": f"s{index}",
            "study_variant": "three_stage_v1",
            "stage_modalities": {"cooking": "graph"},
            "understanding_responses": {
                "cooking": {"failure_mode_ranking": ["fails to pick up the lid"]}
            },
        }
        for index in range(4)
    ]

    rows = failure_scoring.score_rows(
        responses, {"cooking": {"fails to pick up the lid": 3, "fails to place the corn": 1}}
    )

    assert len(rows) == 4
    assert {row["participant"] for row in rows} == {f"session:s{index}" for index in range(4)}
