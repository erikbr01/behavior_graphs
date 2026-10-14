"""Scoring a data-collection choice against the policy we retrained on it.

The question the study asks second — "which additional data would you collect?" —
has a measured right answer: every option was collected, the policy retrained on
it, and the retrained policy rolled out. These tests pin the three things that
can silently go wrong in turning that into a number.

The first is drift. The option strings live in the question bank
(``variants.contrastive_same_page.questions.QUESTION_SETS``) and the policies
live in :mod:`behavior_graphs.viz.streamlit.user_study.regret`; an edit to a
wording on one side and not the other does not raise, it just stops matching, and
the panel quietly reports regret for two tasks instead of three.

The second is a partially measured task: with one arm's rollouts missing, the
best option is a max over whatever is on disk and every regret computed against
it is understated. Such a task must produce no rows at all.

The third is the baseline. It is a policy we have a success rate for, and for one
of the three tasks it beats every option — but nobody was offered "collect
nothing", so it must never be the option regret is measured against.
"""

from __future__ import annotations

import pytest

from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
    QUESTION_SETS,
)
from behavior_graphs.viz.streamlit.user_study import figure_export, regret

# The three-stage layout's shape, trimmed to the data-collection answers: one
# task per stage, each read in one display, keyed by task rather than by display.
THREE_STAGE = {
    "participant_id": "p1",
    "group": "randomized",
    "study_variant": "three_stage_v1",
    "status": "submitted",
    "task_order": ["set_table", "prepare_tray", "cooking"],
    "task_labels": {
        "set_table": "Set the table",
        "prepare_tray": "Prepare the tray",
        "cooking": "Cook the meal",
    },
    "evidence_order": ["graph", "videos", "graph"],
    "stage_modalities": {"set_table": "graph", "prepare_tray": "videos", "cooking": "graph"},
    "understanding_responses": {
        "set_table": {"data_collection_choice": "Chocolate milk left, cola middle, cupcake right"},
        "prepare_tray": {"data_collection_choice": "Bowl and fork on top of the shelf"},
        "cooking": {"data_collection_choice": "Pick up the corn and place it in the pot."},
    },
    "timestamp": "20260901_120000",
}

#: A second participant who read the same tasks in the other display, so a
#: graph-versus-videos comparison has something on both sides.
THREE_STAGE_OTHER = {
    **THREE_STAGE,
    "participant_id": "p2",
    "stage_modalities": {"set_table": "videos", "prepare_tray": "graph", "cooking": "videos"},
    "understanding_responses": {
        # The best set_table option and the worst cooking one: regret 0 and the
        # full spread of that arm set.
        "set_table": {"data_collection_choice": "Chocolate milk left, cola middle, cupcake right"},
        "prepare_tray": {
            "data_collection_choice": "Bowl in the middle of the shelf, fork on top of the shelf"
        },
        "cooking": {"data_collection_choice": "Pick up the corn and place it in the pot."},
    },
    "timestamp": "20260901_130000",
}


def rates(**overrides: tuple[int, int]) -> dict[str, regret.SuccessRate]:
    """Stand-in success rates, defaulting to the measured ones.

    Round numbers rather than the recorded 46/60s: a test that asserts a regret
    of 30.0 should fail when the arithmetic changes, not when somebody corrects
    an outcome label on disk. Override an entry to make an arm better, worse or
    (with ``None``) unmeasured.
    """
    counts: dict[str, tuple[int, int] | None] = {
        "set_table_baseline_pretrained_dit": (50, 100),
        "set_table_plusleft_pretrained_dit": (80, 100),
        "set_table_plusmiddle_pretrained_dit": (50, 100),
        "set_table_plusright_pretrained_dit": (40, 100),
        "prepare_tray_baseline": (60, 100),
        "prepare_tray_plustop": (75, 100),
        "prepare_tray_plusbottom": (90, 100),
        # The task whose baseline beats every option.
        "cooking_baseline_dit_200": (25, 100),
        "cooking_baseline_carrot_dit": (20, 100),
        "cooking_baseline_corn_dit": (15, 100),
        "cooking_baseline_lid_dit": (24, 100),
        "set_table_baseline_v2_sync": (38, 100),
        "set_table_baseline_v2_plusleft_sync": (57, 100),
        "set_table_baseline_v2_plusmiddle_sync": (53, 100),
        "set_table_baseline_v2_plusright_sync": (47, 100),
    }
    counts.update(overrides)  # type: ignore[arg-type]
    return {
        dataset: regret.SuccessRate(dataset=dataset, successes=count[0], trials=count[1])
        for dataset, count in counts.items()
        if count is not None
    }


def test_every_option_the_study_asks_names_a_retrained_policy():
    """The drift guard: the question bank and the arm registry must agree.

    A wording edited in the bank and not here does not raise anywhere — the
    choice simply stops matching an arm and the task drops out of the panel.
    """
    for question_set, spec in QUESTION_SETS.items():
        arms = regret.task_by_key(question_set)
        assert arms is not None, f"{question_set} has no retrained policies registered"
        for option in spec["collection_options"]:
            found = regret.arm_for_option(option)
            assert found is not None, f"{question_set}: {option!r} names no retrained policy"
            assert found[0].task == question_set


def test_each_option_maps_to_its_own_policy():
    """Two options sharing a dataset would score one collection as two."""
    for task in regret.TASK_ARMS:
        assert len(set(task.datasets)) == len(task.arms)


def test_an_option_identifies_its_task_without_the_response_saying():
    """What lets the contrastive layouts be scored: they key answers by display,
    so the option string is the only thing that says which task it belongs to."""
    seen: dict[str, str] = {}
    for task in regret.TASK_ARMS:
        for arm in task.arms:
            for option in arm.options:
                assert option not in seen, f"{option!r} is both {seen.get(option)} and {task.task}"
                seen[option] = task.task


def test_regret_is_the_distance_to_the_best_option():
    rows = regret.regret_rows([THREE_STAGE], rates())
    by_task = {row["task"]: row for row in rows}

    # Picked the best set_table option: 80 of 100 against a best of 80.
    assert by_task["set_table"]["regret"] == pytest.approx(0.0)
    assert by_task["set_table"]["chose_best"]
    # Picked the worse prepare_tray option: 90 is available, 75 was taken.
    assert by_task["prepare_tray"]["regret"] == pytest.approx(15.0)
    assert not by_task["prepare_tray"]["chose_best"]
    assert by_task["prepare_tray"]["best_label"] == "Bowl in middle"
    # And the worst cooking option: 24 is available, 15 was taken.
    assert by_task["cooking"]["regret"] == pytest.approx(9.0)


def test_the_display_a_choice_was_made_off_comes_from_the_stage():
    """Which is a property of the response, not of the group — the three-stage
    layout draws the display per participant."""
    rows = regret.regret_rows([THREE_STAGE], rates())
    assert {row["task"]: row["phase"] for row in rows} == {
        "set_table": "graph",
        "prepare_tray": "videos",
        "cooking": "graph",
    }


def test_the_baseline_is_never_the_option_to_beat():
    """Cooking's baseline (25) beats all three of its arms, and the best arm is
    still the 24 — "collect nothing" was not on offer, so scoring against it
    would report regret for a choice nobody could have made."""
    rows = regret.regret_rows([THREE_STAGE], rates())
    cooking = next(row for row in rows if row["task"] == "cooking")

    assert cooking["best_rate"] == pytest.approx(24.0)
    assert cooking["baseline_rate"] == pytest.approx(25.0)
    # The column that says the choice made the policy worse than reading it and
    # collecting nothing: 15 against a baseline of 25.
    assert cooking["gain_over_baseline"] == pytest.approx(-10.0)


def test_a_task_with_an_unmeasured_arm_is_not_scored_at_all():
    """Not scored against the arms that happen to be on disk: the max would be a
    max over two of three options, and every regret under it understated."""
    partial = rates(prepare_tray_plusbottom=None)  # type: ignore[arg-type]

    rows = regret.regret_rows([THREE_STAGE], partial)

    assert {row["task"] for row in rows} == {"set_table", "cooking"}
    assert regret.unscored_datasets(partial) == {"prepare_tray": ("prepare_tray_plusbottom",)}
    assert "prepare_tray" not in {task.task for task in regret.scoreable(partial)}


def test_a_missing_baseline_costs_only_its_own_column():
    """It is context, not the reference: without it the task is still scored."""
    rows = regret.regret_rows([THREE_STAGE], rates(prepare_tray_baseline=None))  # type: ignore[arg-type]
    tray = next(row for row in rows if row["task"] == "prepare_tray")

    assert tray["regret"] == pytest.approx(15.0)
    assert tray["baseline_rate"] is None
    assert tray["gain_over_baseline"] is None


def test_an_answer_that_names_no_arm_produces_no_row():
    """The production layout's ``ranked_choices`` are strategies nobody
    retrained a policy on, and a store holds responses to wordings the bank has
    since dropped."""
    unknown = {
        **THREE_STAGE,
        "understanding_responses": {"set_table": {"data_collection_choice": "collect more data"}},
    }
    assert regret.regret_rows([unknown], rates()) == []


def test_a_dropped_wording_still_scores():
    """Six option strings named three policies before the question was cut down
    to one option per milk position; a stored answer to one of them is data."""
    old = {
        **THREE_STAGE,
        "understanding_responses": {
            "set_table": {"data_collection_choice": "Milk left, cupcake middle, cola right"}
        },
    }
    row = regret.regret_rows([old], rates())[0]

    assert row["dataset"] == "set_table_plusleft_pretrained_dit"
    assert row["regret"] == pytest.approx(0.0)


def test_pooling_is_over_answers_and_only_where_a_display_is_named():
    rows = regret.regret_rows([THREE_STAGE, THREE_STAGE_OTHER], rates())
    pooled = regret.regret_by_display(rows)

    # Three tasks each, one display per task: p1 read set_table and cooking off
    # the graph (regret 0 and 9) and p2 read prepare_tray off it (0), so the
    # pooled column is three decisions rather than one per participant.
    assert sorted(pooled["graph"]) == pytest.approx([0.0, 0.0, 9.0])
    assert sorted(pooled["videos"]) == pytest.approx([0.0, 9.0, 15.0])


def test_a_response_that_never_recorded_a_display_is_left_out_of_the_comparison():
    """The production layout showed one condition and did not record which."""
    no_phase = {
        "participant_id": "p3",
        "group": "A",
        "understanding_responses": {"": {"data_collection_choice": None}},
        "data_collection_priorities": {"": "Pick up the lid and place it on the pot."},
    }
    rows = regret.regret_rows([no_phase], rates())

    assert [row["phase"] for row in rows] == [""]
    assert regret.regret_by_display(rows) == {"graph": [], "videos": []}


def test_the_summary_pools_the_tasks_into_one_row_per_display():
    rows = regret.regret_rows([THREE_STAGE, THREE_STAGE_OTHER], rates())
    summary = {(row["task_label"], row["phase"]): row for row in regret.summary_rows(rows)}

    graph_overall = summary[("All tasks", "graph")]
    assert graph_overall["answers"] == 3
    assert graph_overall["mean_regret"] == pytest.approx(3.0)
    assert graph_overall["chose_best"] == 2
    assert graph_overall["chose_best_rate"] == pytest.approx(200 / 3)
    assert summary[("All tasks", "videos")]["mean_regret"] == pytest.approx(8.0)
    # And the tasks are still there individually, in the study's stage order.
    assert [row["task_label"] for row in regret.summary_rows(rows)][:2] == [
        "Set the table",
        "Set the table",
    ]


def test_the_tasks_are_ordered_by_the_studys_stages_not_alphabetically():
    by_task = regret.regret_by_task(regret.regret_rows([THREE_STAGE], rates()))

    assert list(by_task) == ["set_table", "prepare_tray", "cooking"]


def test_an_order_effect_ranking_scores_against_the_policy_it_showed():
    """That layout ranks option *ids* over the earlier v2 policy, so its arms
    are the v2 retrains — scoring it against the pretrained-DiT arms would
    compare a choice with a policy the participant never saw."""
    order_effect = {
        "participant_id": "p4",
        "group": "B",
        "study_variant": "order_effect_v1",
        "modality_order": ["graph", "videos"],
        "data_collection_priorities": {"graph": "milk_middle", "videos": "milk_left"},
    }
    rows = {row["phase"]: row for row in regret.regret_rows([order_effect], rates())}

    assert rows["graph"]["dataset"] == "set_table_baseline_v2_plusmiddle_sync"
    assert rows["graph"]["regret"] == pytest.approx(4.0)
    assert rows["videos"]["regret"] == pytest.approx(0.0)


def test_a_rate_needs_a_scored_rollout():
    """An unlabelled dataset has no success rate — not a rate of zero."""
    with pytest.raises(ValueError, match="scored rollout"):
        regret.SuccessRate(dataset="x", successes=0, trials=0)


@pytest.mark.parametrize("mode", list(figure_export.REGRET_FIGURE_MODES))
def test_the_regret_figure_exports_a_pdf_in_every_mode(mode):
    rows = regret.regret_rows([THREE_STAGE, THREE_STAGE_OTHER], rates())

    files = figure_export.regret_figure_files(rows, mode)

    assert files is not None
    assert files["pdf"].startswith(b"%PDF")
    assert files["png"].startswith(b"\x89PNG")


def test_the_regret_figure_is_reproducible():
    """No timestamp in the PDF, so the same responses export the same bytes —
    a figure that changes on every download is one a reviewer cannot diff."""
    rows = regret.regret_rows([THREE_STAGE], rates())

    first = figure_export.regret_figure_files(rows, "both")
    second = figure_export.regret_figure_files(rows, "both")

    assert first is not None and second is not None
    assert first["pdf"] == second["pdf"]


def test_no_regret_figure_when_no_choice_names_a_display():
    """``None`` rather than empty axes, which would read as a plotting fault."""
    no_phase = {
        "participant_id": "p3",
        "group": "A",
        "data_collection_priorities": {"": "Pick up the lid and place it on the pot."},
    }
    rows = regret.regret_rows([no_phase], rates())

    assert rows  # the choice is scored...
    assert figure_export.regret_figure(rows, "both") is None  # ...but not comparable


def test_an_unknown_figure_mode_is_a_bug_not_a_blank_figure():
    with pytest.raises(ValueError, match="regret figure mode"):
        figure_export.regret_figure([], "sideways")


# ── Shares: who asked for which data, off which display ──────────────────────


def _shares(rows, task: str, phase: str) -> dict[str, tuple[int, int]]:
    """``{dataset: (picks, answers)}`` for one task and display."""
    return {
        row["dataset"]: (row["picks"], row["answers"])
        for row in rows
        if row["task"] == task and row["phase"] == phase
    }


def test_a_share_is_of_the_answers_that_display_drew_for_that_task():
    """Not of the participants, and not of every answer in the store: a share of
    picks is only a share among the options that were on the same screen."""
    rows = regret.choice_share_rows([THREE_STAGE, THREE_STAGE_OTHER])

    # Both participants read set_table, in different displays, and both picked
    # milk left — one answer per display, so each is the whole of its own column.
    assert _shares(rows, "set_table", "graph")["set_table_plusleft_pretrained_dit"] == (1, 1)
    assert _shares(rows, "set_table", "videos")["set_table_plusleft_pretrained_dit"] == (1, 1)
    assert [row["share"] for row in rows if row["picks"]] == pytest.approx([1.0] * 6)


def test_an_option_nobody_picked_gets_a_row_of_its_own():
    """The point of the function: the interesting thing a display can do is
    leave the best option unchosen, and rows only for the picked options cannot
    say it. Zero picks, not a missing row."""
    rows = regret.choice_share_rows([THREE_STAGE])
    graph = _shares(rows, "set_table", "graph")

    assert graph["set_table_plusmiddle_pretrained_dit"] == (0, 1)
    assert graph["set_table_plusright_pretrained_dit"] == (0, 1)
    # Every arm of the task, and nothing else — the baseline was not an option.
    assert set(graph) == set(regret.task_by_key("set_table").datasets)


def test_the_baseline_is_named_on_the_row_but_is_not_one_of_them():
    """A figure drawing a bar per policy has to know which policy was never on
    offer; the arm set's own baseline rides along so it can."""
    rows = regret.choice_share_rows([THREE_STAGE])

    assert {row["baseline"] for row in rows if row["task"] == "set_table"} == {
        "set_table_baseline_pretrained_dit"
    }
    assert all(row["dataset"] != row["baseline"] for row in rows)


def test_the_stage_the_participant_chose_the_display_for_is_left_out_by_default():
    """That stage's display is an outcome of the participant, so a share pooled
    over it is a sentence about people rather than about displays."""
    chosen = {
        **THREE_STAGE,
        "modality_assignment": {
            "set_table": "assigned",
            "prepare_tray": "assigned",
            "cooking": "chosen",
        },
    }

    randomized = regret.choice_share_rows([chosen])
    assert not [row for row in randomized if row["task"] == "cooking"]

    pooled = regret.choice_share_rows([chosen], randomized_only=False)
    assert _shares(pooled, "cooking", "graph")["cooking_baseline_corn_dit"] == (1, 1)


def test_an_answer_with_no_display_on_it_is_not_a_share_of_either():
    """The production layout showed one condition and never recorded which."""
    no_phase = {
        "participant_id": "p3",
        "group": "A",
        "data_collection_priorities": {"": "Pick up the lid and place it on the pot."},
    }

    assert regret.choice_share_rows([no_phase]) == []


def test_an_answer_that_names_no_option_is_not_counted_in_the_denominator():
    """A free-text answer, or a wording the question bank has dropped and this
    module has not aliased: it cannot be placed on a policy, so it must not
    quietly shrink everybody else's share either."""
    junk = {
        **THREE_STAGE,
        "participant_id": "p4",
        "understanding_responses": {"set_table": {"data_collection_choice": "more data please"}},
    }
    rows = regret.choice_share_rows([THREE_STAGE, junk])

    assert _shares(rows, "set_table", "graph")["set_table_plusleft_pretrained_dit"] == (1, 1)


def test_the_displays_come_out_graph_first_whatever_order_they_were_answered_in():
    """The figure takes its row order from these rows, and the paper meets the
    graph on the left of every panel."""
    rows = regret.choice_share_rows([THREE_STAGE_OTHER, THREE_STAGE])
    set_table = [row["phase"] for row in rows if row["task"] == "set_table"]

    assert set_table[0] == "graph"
    assert set_table[-1] == "videos"


# ── Normalizing ──────────────────────────────────────────────────────────────


def _set_table() -> regret.TaskArms:
    return next(task for task in regret.TASK_ARMS if task.task == "set_table")


def _cooking() -> regret.TaskArms:
    return next(task for task in regret.TASK_ARMS if task.task == "cooking")


def test_the_span_is_the_best_option_minus_the_worst():
    """The denominator that makes two tasks comparable: raw points do not pool,
    because one arm set spans four times what another does."""
    assert regret.arm_span(_set_table(), rates()) == pytest.approx(40.0)  # 80 to 40
    assert regret.arm_span(_cooking(), rates()) == pytest.approx(9.0)  # 24 to 15


def test_a_task_whose_options_all_measure_the_same_has_no_scale():
    """``None``, not zero: nobody could have chosen wrongly there, and dividing
    by it would say everybody did."""
    flat = rates(
        set_table_plusleft_pretrained_dit=(50, 100),
        set_table_plusmiddle_pretrained_dit=(50, 100),
        set_table_plusright_pretrained_dit=(50, 100),
    )

    assert regret.arm_span(_set_table(), flat) is None


def test_the_gain_is_what_choosing_well_was_worth():
    """Best option minus collecting nothing — the value that was actually on the
    table in the decision the participant was asked to make."""
    assert regret.gain_over_baseline(_set_table(), rates()) == pytest.approx(30.0)  # 80 - 50


def test_a_task_no_option_improved_has_no_gain_to_forgo_a_share_of():
    """The case that has to be guarded rather than divided by: a negative
    denominator would report the worst choice as the best one. Cooking is that
    task in these fixtures — its baseline beats every option."""
    assert regret.gain_over_baseline(_cooking(), rates()) is None

    (row,) = [r for r in regret.regret_rows([THREE_STAGE], rates()) if r["task"] == "cooking"]
    assert row["regret_of_gain"] is None
    # The bounded score still works there, which is half of why there are two.
    assert row["regret_normalized"] == pytest.approx(1.0)


def test_a_gain_that_is_not_separated_from_zero_is_not_a_denominator():
    """The defect this gate exists for: cooking's best option is 16 successes in
    60 against a baseline's 14, and dividing a 5-point regret by that 3.3-point
    "gain" produced a mean of 165% of the gain forgone — which reads as a strong
    finding and is an artifact of a near-zero denominator."""
    task = _cooking()
    close = rates(cooking_baseline_dit_200=(14, 60), cooking_baseline_lid_dit=(16, 60))

    assert regret.gain_over_baseline(task, close) == pytest.approx(100 * 2 / 60)
    assert regret.gain_p_value(task, close) > regret.GAIN_ALPHA
    assert regret.measurable_gain(task, close) is None

    (row,) = [r for r in regret.regret_rows([THREE_STAGE], close) if r["task"] == "cooking"]
    assert row["regret_of_gain"] is None
    # Carried on the row anyway, so the page can say why the column is blank.
    assert row["gain"] == pytest.approx(100 * 2 / 60)
    assert row["gain_p"] == pytest.approx(regret.gain_p_value(task, close))
    # And the bounded score still scores it: best of several useless options is
    # a better read than worst of them.
    assert row["regret_normalized"] is not None


def test_a_separated_gain_is_used():
    """The three tasks whose data-collection actually worked: 18 to 28 points,
    every one of them separated from collecting nothing."""
    task = _set_table()

    assert regret.gain_p_value(task, rates()) < regret.GAIN_ALPHA
    assert regret.measurable_gain(task, rates()) == pytest.approx(30.0)


def test_chance_is_not_a_half():
    """It depends on how the options are spread, which is the whole reason it
    has to be reported beside the score: 0.5 is a good result on one task and a
    bad one on another."""
    # set_table: 0, 30 and 40 points behind the best, over a span of 40.
    assert regret.chance_normalized(_set_table(), rates()) == pytest.approx((0 + 0.75 + 1.0) / 3)
    # Evenly spaced options are the one case where it does land on a half.
    even = rates(
        set_table_plusleft_pretrained_dit=(80, 100),
        set_table_plusmiddle_pretrained_dit=(60, 100),
        set_table_plusright_pretrained_dit=(40, 100),
    )
    assert regret.chance_normalized(_set_table(), even) == pytest.approx(0.5)


def test_a_choice_worse_than_collecting_nothing_scores_over_one_of_the_gain():
    """Which is the thing the bounded score cannot say: the worst option of a
    task is 1.0 normalized whether it merely trails the best or actively undoes
    the baseline."""
    worst = _set_table().arms[-1]  # 40 of 100, against a baseline of 50
    response = {
        **THREE_STAGE,
        "understanding_responses": {"set_table": {"data_collection_choice": worst.options[0]}},
    }

    (row,) = regret.regret_rows([response], rates())

    assert row["regret"] == pytest.approx(40.0)
    assert row["regret_normalized"] == pytest.approx(1.0)
    # 40 points behind the best, of which only 30 were ever on the table.
    assert row["regret_of_gain"] == pytest.approx(40 / 30)
    assert row["worse_than_baseline"] is True


def test_the_best_option_is_zero_on_both_scales():
    (row,) = [r for r in regret.regret_rows([THREE_STAGE], rates()) if r["task"] == "set_table"]

    assert row["regret_normalized"] == pytest.approx(0.0)
    assert row["regret_of_gain"] == pytest.approx(0.0)
    assert row["worse_than_baseline"] is False


def test_a_summary_averages_the_normalized_scores_over_the_answers_that_have_them():
    rows = regret.regret_rows([THREE_STAGE, THREE_STAGE_OTHER], rates())

    by_task = {
        (row["task"], row["phase"]): row
        for row in regret.summary_rows(rows)
        if row["task"] == "set_table"
    }

    # Both participants picked the best set_table option, in either display.
    for row in by_task.values():
        assert row["mean_normalized"] == pytest.approx(0.0)
        assert row["mean_of_gain"] == pytest.approx(0.0)
        assert row["worse_than_baseline_rate"] == pytest.approx(0.0)
        # The reference is averaged over the same answers as the score.
        assert row["chance_normalized"] == pytest.approx(
            regret.chance_normalized(_set_table(), rates())
        )


def test_the_pooled_normalized_mean_skips_a_task_that_has_no_gain_scale():
    """Cooking has no `regret_of_gain` in these fixtures, so it contributes to
    the pooled bounded score and not to the pooled share-of-gain — rather than
    contributing a zero to it, which would read as a perfect choice."""
    rows = regret.regret_rows([THREE_STAGE], rates())

    pooled = next(
        row
        for row in regret.summary_rows(rows)
        if row["task_label"] == "All tasks" and row["phase"] == "graph"
    )

    # This participant read set_table and cooking on the graph: the best
    # set_table option (0 on both scales) and the worst cooking one (1.0
    # bounded, and no share-of-gain at all, since nothing improved cooking).
    assert pooled["answers"] == 2
    assert pooled["mean_normalized"] == pytest.approx(0.5)
    assert pooled["mean_of_gain"] == pytest.approx(0.0)


def test_a_none_that_pandas_turned_into_a_nan_is_still_missing():
    """A caller hands these rows over as a dataframe's records, and pandas turns
    a None in a float column into NaN on the way. An `is None` test alone lets
    that through, and one NaN takes a mean with it — the page's own regret table
    read NaN for exactly this reason."""
    assert regret.has_score(0.0) is True
    assert regret.has_score(None) is False
    assert regret.has_score(float("nan")) is False

    rows = [
        {
            "task": "t",
            "task_label": "T",
            "phase": "graph",
            "regret": 0.0,
            "chose_best": True,
            "regret_normalized": 0.0,
            "regret_of_gain": float("nan"),
            "chance_normalized": 0.5,
            "worse_than_baseline": None,
        },
        {
            "task": "t",
            "task_label": "T",
            "phase": "graph",
            "regret": 10.0,
            "chose_best": False,
            "regret_normalized": 1.0,
            "regret_of_gain": 2.0,
            "chance_normalized": 0.5,
            "worse_than_baseline": True,
        },
    ]

    (pooled,) = [row for row in regret.summary_rows(rows) if row["task_label"] == "All tasks"]

    assert pooled["mean_normalized"] == pytest.approx(0.5)
    # The NaN is skipped, not averaged in: a mean of 2.0, not of NaN.
    assert pooled["mean_of_gain"] == pytest.approx(2.0)
    assert pooled["worse_than_baseline_rate"] == pytest.approx(100.0)


def test_the_randomized_rows_are_the_assigned_stages():
    """One home for the restriction every display comparison is made under —
    it was spelled out in four places, and a page that spelled it differently
    would export a figure covering different evidence from its neighbours."""
    rows = [
        {"phase": "graph", "assignment": "assigned", "regret": 1.0},
        {"phase": "videos", "assignment": "chosen", "regret": 2.0},
        # No assignment recorded: the layouts that never assigned a display had
        # no free choice to contaminate the comparison with, so this counts.
        {"phase": "graph", "regret": 3.0},
    ]

    assert [row["regret"] for row in regret.randomized_rows(rows)] == [1.0, 3.0]


def test_the_restriction_is_what_the_figures_and_the_claims_apply():
    """Not a near-copy each: the same function, so none of them can drift."""
    rows = regret.regret_rows([THREE_STAGE, THREE_STAGE_OTHER], rates())
    chosen = [{**row, "assignment": "chosen"} for row in rows]

    assert regret.randomized_rows(chosen) == []
    assert regret.randomized_rows(rows) == rows  # the fixtures assign nothing
