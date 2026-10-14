"""Reading a stored response written by any of the study's five layouts.

The layouts disagree about where an answer goes, and one of the disagreements is
silent: ``nasa_tlx`` is ``{dimension: score}`` in the production layout and
``{phase: {dimension: score}}`` in the contrastive one, so reading the first as
the second yields nothing rather than an error. That is exactly what the
analytics page did before it went through this module — an empty task-load panel
over a store full of contrastive responses.

The fixtures below are the shapes each layout's ``_render_submit`` actually
writes, trimmed to the keys under test.
"""

from __future__ import annotations

from behavior_graphs.viz.streamlit.user_study import analytics

PRODUCTION = {
    "participant_id": "p1",
    "group": "A",
    "session": "set_table_baseline",
    "ranked_choices": ["more milk poses", "more pumpkin poses", None],
    "nasa_tlx": {"mental_demand": 60, "effort": 40},
    "likert_strategy": {"q1": 4, "q2": 2},
    "likert_final": {"q3": 5},
    "notes": "harder than it looked",
    "timestamp": "20260801_120000",
    "step_durations_seconds": {"0": 30.0, "1": 90.0},
}

PRODUCTION_ALLOCATING = {
    "participant_id": "p2",
    "group": "B",
    "allocations": {"strategy_a": 200, "strategy_b": 300},
    "likert_graph": {"q4": 3},
    "timestamp": "20260801_130000",
}

ORDER_EFFECT = {
    "participant_id": "p3",
    "group": "B",
    "study_variant": "order_effect_v1",
    "modality_order": ["graph", "videos"],
    "survey_after_graph": {"understood_failures": 5},
    "survey_after_videos": {"understood_failures": 3},
    "nasa_tlx_after_graph": {"mental_demand": 30},
    "nasa_tlx_after_videos": {"mental_demand": 70},
    "data_collection_priorities_after_graph": ["milk_left", "milk_middle", None],
    "data_collection_priorities_after_videos": ["milk_middle", "milk_left", None],
    "notes": "",
    "timestamp": "20260810_090000",
}

CONTRASTIVE = {
    "participant_id": "p4",
    "group": "A",
    "study_variant": "contrastive_set_table_v1",
    "evidence_order": ["videos", "graph"],
    "understanding_responses": {
        "videos": {
            "failure_mode_ranking": ["fails to grasp the milk", "fails to place the milk"],
            "data_collection_choice": "Milk left, cola middle, cupcake right",
        },
        "graph": {
            "failure_mode_ranking": ["fails to place the milk", "fails to grasp the milk"],
            "data_collection_choice": "Milk left, cola middle, cupcake right",
        },
    },
    "nasa_tlx": {
        "videos": {"temporal_demand": 40, "effort": 20},
        "graph": {"temporal_demand": 40, "effort": 40},
    },
    "data_collection_priorities": {
        "videos": "Milk left, cola middle, cupcake right",
        "graph": "Milk left, cola middle, cupcake right",
    },
    # A value the radio can produce: it offers EXPERIENCE_SCALE and nothing else,
    # and the page orders the chart by that scale.
    "robot_data_collection_experience": "A good deal",
    "remarks": "the graph was easier to skim",
    "timestamp": "20260823_185114",
    "step_durations_seconds": {"0": 151.1, "1": 1001.3},
}

THREE_STAGE = {
    "participant_id": "p5",
    "group": "randomized",
    "study_variant": "three_stage_v1",
    # Three tasks, one display each, in an order drawn for this participant --
    # so the same nested key that is a modality in CONTRASTIVE is a task here.
    "task_order": ["cooking", "set_table", "prepare_tray"],
    "task_labels": {
        "cooking": "Cook the meal",
        "set_table": "Set the table",
        "prepare_tray": "Prepare the tray",
    },
    "evidence_order": ["graph", "videos", "videos"],
    "stage_modalities": {"cooking": "graph", "set_table": "videos", "prepare_tray": "videos"},
    "modality_assignment": {
        "cooking": "assigned",
        "set_table": "assigned",
        "prepare_tray": "chosen",
    },
    "chosen_modality": "videos",
    "session": "cooking_contrastive → set_table_contrastive → prepare_tray_contrastive",
    "sessions": {
        "cooking": "cooking_contrastive",
        "set_table": "set_table_contrastive",
        "prepare_tray": "prepare_tray_contrastive",
    },
    "understanding_responses": {
        "cooking": {
            "failure_mode_ranking": ["fails to pick up the lid", "fails to place the corn"],
            "data_collection_choice": "Pick up the lid and place it on the pot.",
        },
        "set_table": {
            "failure_mode_ranking": ["fails to grasp the milk"],
            "data_collection_choice": "Milk left, cola middle, cupcake right",
        },
        "prepare_tray": {
            "failure_mode_ranking": ["fails to pick up the fork"],
            "data_collection_choice": "Bowl and fork on top of the shelf",
        },
    },
    "nasa_tlx": {
        "cooking": {"temporal_demand": 30, "effort": 30},
        "set_table": {"temporal_demand": 50, "effort": 60},
        "prepare_tray": {"temporal_demand": 40, "effort": 40},
    },
    "data_collection_priorities": {
        "cooking": "Pick up the lid and place it on the pot.",
        "set_table": "Milk left, cola middle, cupcake right",
        "prepare_tray": "Bowl and fork on top of the shelf",
    },
    "robot_data_collection_experience": "A little",
    "remarks": "",
    "timestamp": "20260827_101500",
    "step_durations_seconds": {"0": 60.0, "1": 300.0, "5": 240.0},
    "step_names": [
        "Introduction",
        "Stage 1 (behavior graph)",
        "Intermission",
        "Stage 2 (rollout videos)",
        "Intermission",
        "Stage 3 (rollout videos · your choice)",
        "Closing questions",
        "Submit",
    ],
}

ALL = [PRODUCTION, PRODUCTION_ALLOCATING, ORDER_EFFECT, CONTRASTIVE, THREE_STAGE]


def test_a_response_without_a_variant_is_the_production_layout():
    """Those predate the variant registry; they are not "unknown"."""
    assert analytics.study_version(PRODUCTION) == analytics.PRODUCTION_VERSION
    assert analytics.study_version(CONTRASTIVE) == "contrastive_set_table_v1"


def test_the_versions_are_offered_commonest_first():
    counts = analytics.version_counts(ALL)

    assert list(counts) == [
        analytics.PRODUCTION_VERSION,
        "contrastive_set_table_v1",
        "order_effect_v1",
        "three_stage_v1",
    ]
    assert counts[analytics.PRODUCTION_VERSION] == 2


def test_a_response_that_does_not_say_where_it_came_from_is_a_participant():
    """Every response written before the field existed is one, and guessing the
    other way would hide collected data behind a filter nobody asked for."""
    assert analytics.origin(PRODUCTION) == analytics.PARTICIPANT_ORIGIN
    assert analytics.origin(dict(THREE_STAGE, origin="")) == analytics.PARTICIPANT_ORIGIN
    assert analytics.origin(dict(THREE_STAGE, origin="participant")) == (
        analytics.PARTICIPANT_ORIGIN
    )


def test_the_lab_apps_own_previews_are_separable_from_participants():
    """The three-stage layout writes a record on the first page load, so opening
    it on the User Study page to read the wording leaves one — with no
    participant id, stopped on the introduction, which is what a real dropout
    looks like."""
    preview = dict(THREE_STAGE, participant_id="", origin="lab", status="in_progress", step=0)

    assert analytics.origin(preview) == analytics.LAB_ORIGIN
    assert analytics.from_participants([THREE_STAGE, preview]) == [THREE_STAGE]


def test_selecting_a_version_selects_only_its_responses():
    picked = analytics.for_version(ALL, "order_effect_v1")

    assert [r["participant_id"] for r in picked] == ["p3"]
    assert len(analytics.for_version(ALL, None)) == len(ALL)


def test_the_group_label_says_which_a_it_is():
    """ "A" is the no-graph condition in one layout and videos-first in another."""
    assert analytics.group_label(PRODUCTION) == "A"
    assert analytics.group_label(CONTRASTIVE) == "A (videos → graph)"
    assert analytics.group_label(ORDER_EFFECT) == "B (graph → videos)"


def test_task_load_is_read_out_of_all_three_places_it_is_written():
    rows = analytics.tlx_rows(ALL)

    def scores(participant: str) -> dict:
        return {
            (r["phase"], r["dimension_key"]): r["score"]
            for r in rows
            if r["participant"] == participant
        }

    # Rated once for the whole session: no phase.
    assert scores("p1") == {("", "mental_demand"): 60.0, ("", "effort"): 40.0}
    # One top-level key per phase.
    assert scores("p3") == {("graph", "mental_demand"): 30.0, ("videos", "mental_demand"): 70.0}
    # Nested under the phase — the shape that used to read as empty.
    assert scores("p4") == {
        ("videos", "temporal_demand"): 40.0,
        ("videos", "effort"): 20.0,
        ("graph", "temporal_demand"): 40.0,
        ("graph", "effort"): 40.0,
    }


def test_a_missing_answer_produces_no_row_rather_than_a_hole():
    assert analytics.tlx_rows([{"participant_id": "x", "group": "A"}]) == []
    assert analytics.ranking_rows([{"nasa_tlx": {"effort": 10}}]) == []
    assert analytics.allocation_rows([CONTRASTIVE]) == []


def test_rankings_keep_the_rank_and_drop_the_blanks():
    rows = analytics.ranking_rows(ALL)

    # Keyed by task as well as phase: a task-keyed layout and a modality-keyed
    # one both write "videos" here, and the pair is what identifies a block.
    failure = {
        (r["task"], r["phase"], r["rank"]): r["option"]
        for r in rows
        if r["block"] == "failure_mode"
    }
    assert failure[("", "videos", 1)] == "fails to grasp the milk"
    assert failure[("", "graph", 1)] == "fails to place the milk"
    assert failure[("cooking", "graph", 1)] == "fails to pick up the lid"

    # An unfilled rank slot is None in the response and is not a pick.
    strategy = [r for r in rows if r["block"] == "strategy"]
    assert [r["rank"] for r in strategy] == [1, 2]

    collection = {
        (r["phase"], r["rank"]): r["option"] for r in rows if r["block"] == "data_collection"
    }
    assert collection[("graph", 1)] == "milk_left"
    assert collection[("videos", 1)] == "milk_middle"


def test_the_single_pick_is_read_from_either_key():
    both = analytics.choice_rows([CONTRASTIVE])
    assert {r["phase"]: r["choice"] for r in both} == {
        "videos": "Milk left, cola middle, cupcake right",
        "graph": "Milk left, cola middle, cupcake right",
    }

    # A response saved with only the nested copy, or only the flat one.
    nested_only = dict(CONTRASTIVE)
    nested_only.pop("data_collection_priorities")
    assert len(analytics.choice_rows([nested_only])) == 2

    flat_only = dict(CONTRASTIVE)
    flat_only.pop("understanding_responses")
    assert len(analytics.choice_rows([flat_only])) == 2


def test_ratings_cover_the_likert_blocks_and_the_per_phase_survey():
    rows = analytics.rating_rows(ALL)
    blocks = {(r["participant"], r["block"], r["phase"]) for r in rows}

    assert ("p1", "Strategy selection", "") in blocks
    assert ("p1", "Final assessment", "") in blocks
    assert ("p2", "Graph interaction", "") in blocks
    assert ("p3", "understanding", "graph") in blocks
    assert ("p3", "understanding", "videos") in blocks


def test_the_free_text_box_is_found_under_either_name():
    assert analytics.remarks(PRODUCTION) == "harder than it looked"
    assert analytics.remarks(CONTRASTIVE) == "the graph was easier to skim"
    assert analytics.remarks({"notes": None, "remarks": None}) == ""


def test_a_participant_row_totals_their_time():
    rows = {r["participant"]: r for r in analytics.participant_rows(ALL)}

    assert rows["p1"]["minutes"] == 2.0
    assert rows["p4"]["phases"] == "videos → graph"
    assert rows["p4"]["experience"] == "A good deal"
    # Nothing recorded rather than zero minutes.
    assert rows["p2"]["minutes"] is None


def test_the_change_between_modalities_is_the_headline():
    rows = analytics.phase_disagreements([CONTRASTIVE])
    by_block = {r["block"]: r for r in rows}

    assert by_block["data_collection"]["changed"] is False
    assert by_block["top_failure_mode"]["changed"] is True
    assert by_block["top_failure_mode"]["first_answer"] == "fails to grasp the milk"
    assert by_block["top_failure_mode"]["second_answer"] == "fails to place the milk"


def test_a_ranked_answer_is_compared_on_its_top_pick():
    """The order-effect layout ranks data collection rather than picking one."""
    rows = analytics.phase_disagreements([ORDER_EFFECT])

    assert [(r["block"], r["changed"]) for r in rows] == [("top_data_collection", True)]
    assert rows[0]["first_answer"] == "milk_left"  # graph came first for this group
    assert rows[0]["second_answer"] == "milk_middle"


def test_a_single_condition_response_has_nothing_to_compare():
    assert analytics.phase_disagreements([PRODUCTION]) == []


def test_steps_are_named_from_the_layout_that_recorded_them():
    """The stored durations are keyed by index; an index labels no chart."""
    rows = {r["step_index"]: r["step"] for r in analytics.duration_rows([CONTRASTIVE])}

    assert rows[0] == "Introduction"
    # Group A saw the videos first, so step 1 is the videos for this response.
    assert rows[1] == "Rollout videos + questions"


def test_a_stored_name_list_wins_over_the_reconstructed_one():
    """So a response stays readable after its layout is renamed."""
    renamed = dict(CONTRASTIVE, step_names=["Welcome", "First look"])

    rows = {r["step_index"]: r["step"] for r in analytics.duration_rows([renamed])}

    assert rows[0] == "Welcome"
    assert rows[1] == "First look"


def test_a_step_no_layout_can_name_keeps_its_number():
    unknown = {"study_variant": "future_layout_v2", "step_durations_seconds": {"7": 12.0}}

    assert analytics.duration_rows([unknown])[0]["step"] == "Step 7"


def test_the_group_order_decides_which_phase_step_one_was():
    a_first = analytics.step_labels(CONTRASTIVE)
    b_first = analytics.step_labels(dict(CONTRASTIVE, evidence_order=["graph", "videos"]))

    assert a_first[1] == "Rollout videos + questions"
    assert b_first[1] == "Behavior graph + questions"


def test_the_closing_questions_are_their_own_answer():
    rows = analytics.exit_answer_rows([CONTRASTIVE, PRODUCTION, PRODUCTION_ALLOCATING])
    by_participant = {r["participant"]: r for r in rows}

    assert by_participant["p4"]["experience"] == "A good deal"
    assert by_participant["p4"]["experience"] in analytics.EXPERIENCE_SCALE
    assert by_participant["p4"]["remarks"] == "the graph was easier to skim"
    # The production layout has no experience question, but it does have notes.
    assert by_participant["p1"]["experience"] == ""
    assert by_participant["p1"]["remarks"] == "harder than it looked"
    # Answered neither: no row rather than an empty one.
    assert "p2" not in by_participant


def test_a_layout_that_wrote_the_answers_flat_does_not_take_the_page_down():
    """The contrastive layout's first build (c90fa4a) wrote
    ``understanding_responses`` un-nested, and the store is append-only and
    unvalidated — so one of those responses used to raise `AttributeError` from
    `.get` on a list, which is the whole page rather than one panel."""
    flat = dict(
        CONTRASTIVE,
        understanding_responses={
            "failure_mode_ranking": ["fails to grasp the milk"],
            "data_collection_choice": "Milk left, cola middle, cupcake right",
        },
    )

    assert analytics.ranking_rows([flat]) == []
    # Still found via data_collection_priorities, which is per phase in that build.
    assert {r["phase"] for r in analytics.choice_rows([flat])} == {"videos", "graph"}
    assert analytics.phase_disagreements([flat])  # and the page still compares them


def test_a_task_keyed_block_keeps_the_task_and_resolves_its_display():
    """The three-stage layout keys its answers by task, and the display a task
    was read in varies per participant — so a row carries both.

    Colouring a chart by "phase" off the raw key would have put "cooking" in a
    legend of modalities.
    """
    by_task = {row["task"]: row for row in analytics.choice_rows([THREE_STAGE])}

    assert set(by_task) == {"cooking", "set_table", "prepare_tray"}
    assert by_task["cooking"]["phase"] == "graph"
    assert by_task["prepare_tray"]["phase"] == "videos"
    assert by_task["cooking"]["choice"] == "Pick up the lid and place it on the pot."

    tlx = {(r["task"], r["dimension_key"]): r["score"] for r in analytics.tlx_rows([THREE_STAGE])}
    assert tlx[("set_table", "effort")] == 60.0
    assert {r["phase"] for r in analytics.tlx_rows([THREE_STAGE])} == {"graph", "videos"}

    ranked = analytics.ranking_rows([THREE_STAGE])
    assert {(r["task"], r["phase"]) for r in ranked} >= {("cooking", "graph")}
    assert all(row["block"] == "failure_mode" for row in ranked)


def test_a_modality_keyed_block_still_has_no_task():
    """The contrastive layout's keys are modalities; nothing to resolve there."""
    assert {row["task"] for row in analytics.choice_rows([CONTRASTIVE])} == {""}
    assert {row["phase"] for row in analytics.choice_rows([CONTRASTIVE])} == {"videos", "graph"}
    assert {row["task"] for row in analytics.tlx_rows([PRODUCTION, ORDER_EFFECT])} == {""}


def test_the_stages_are_their_own_table():
    """Which task in which stage, which display, and who picked it."""
    rows = analytics.stage_rows([THREE_STAGE])

    assert [(r["stage"], r["task"], r["phase"], r["assignment"]) for r in rows] == [
        (1, "cooking", "graph", "assigned"),
        (2, "set_table", "videos", "assigned"),
        (3, "prepare_tray", "videos", "chosen"),
    ]
    assert rows[0]["task_label"] == "Cook the meal"
    # Nothing to say about a layout that shows one task.
    assert analytics.stage_rows([CONTRASTIVE, PRODUCTION]) == []


def test_the_free_choice_is_readable_off_the_stage_rows():
    """The result the last stage exists for: with nothing assigned, which
    display does someone reach for?"""
    chosen = [r for r in analytics.stage_rows([THREE_STAGE]) if r["assignment"] == "chosen"]

    assert [r["phase"] for r in chosen] == ["videos"]


def test_a_three_stage_participant_row_lists_the_tasks_in_order():
    row = next(r for r in analytics.participant_rows(ALL) if r["participant"] == "p5")

    assert row["tasks"] == "cooking → set_table → prepare_tray"
    assert row["group_label"] == "randomized (graph → videos → videos)"
    # Three phases, so there is no two-modality comparison to draw.
    assert analytics.phase_disagreements([THREE_STAGE]) == []


def test_three_stage_step_names_survive_into_the_timing_chart():
    """What was stored wins, even after the layout renamed its steps.

    The fixture's stored names say "Stage 1", which is what an attempt saved
    before the steps were renumbered to "Task 1" carries. Reading it back in
    today's wording would relabel history.
    """
    rows = {r["step_index"]: r["step"] for r in analytics.duration_rows([THREE_STAGE])}

    assert rows[1] == "Stage 1 (behavior graph)"
    assert rows[5] == "Stage 3 (rollout videos · your choice)"
    # A response that stored none is still named, off the modalities — in the
    # wording the layout uses now.
    unstored = {k: v for k, v in THREE_STAGE.items() if k != "step_names"}
    assert analytics.step_labels(unstored)[1] == "Task 1 (behavior graph)"


def test_a_malformed_duration_block_does_not_break_the_first_row():
    """`participant_rows` runs before any panel, so it is the worst place to fail."""
    broken = dict(CONTRASTIVE, step_durations_seconds=[151.1, 1001.3])

    rows = analytics.participant_rows([broken])

    assert rows[0]["minutes"] is None
    assert analytics.duration_rows([broken]) == []


def test_an_order_that_names_one_modality_gets_no_step_names():
    """Padding it would label both phase steps with the same modality, and an
    empty order would assert videos-first for a graph-first participant."""
    from behavior_graphs.viz.streamlit.user_studies.step_names import step_names

    assert step_names("contrastive_set_table_v1", ["videos"]) == []
    assert step_names("contrastive_set_table_v1", []) == []
    assert step_names("order_effect_v1", ["graph"]) == []
    # Which leaves the numbers, rather than a wrong word.
    half = dict(CONTRASTIVE, evidence_order=["videos"])
    assert {r["step"] for r in analytics.duration_rows([half])} == {"Step 0", "Step 1"}


# ── Who a response came from ──────────────────────────────────────────────────
# One reading, used by every row builder in the module. It had two, and the
# difference was invisible until a panel counted participants: see
# `analytics.participant` for the four panels that were caught by it.


def test_a_participant_id_is_used_where_the_response_carries_one():
    assert analytics.participant({"participant_id": "p5", "session_id": "abc"}) == "p5"
    # Whitespace is not an id — a padded field is an empty one.
    assert analytics.participant({"participant_id": "  ", "session_id": "abc"}) == "session:abc"


def test_an_anonymous_session_is_its_own_participant_and_not_one_shared_name():
    """The recruitment link carries no participant id, so this is every real
    response the three-stage layout wrote. Collapsing them is what made the
    Participants tile read 1 beside Responses 30."""
    responses = [{"session_id": f"s{index}"} for index in range(30)]

    named = {analytics.participant(response) for response in responses}

    assert len(named) == 30
    assert "anon" not in named


def test_a_response_identifying_itself_no_way_at_all_is_still_a_row():
    assert analytics.participant({}) == "anon"


def test_every_row_builder_names_a_participant_the_same_way():
    """The bug was one function disagreeing with another, so the test is that
    they agree — not that either returns a particular string."""
    anonymous = dict(THREE_STAGE, session_id="s1")
    anonymous.pop("participant_id")
    expected = analytics.participant(anonymous)

    assert expected == "session:s1"
    assert [row["participant"] for row in analytics.participant_rows([anonymous])] == [expected]
    for rows in (
        analytics.choice_rows([anonymous]),
        analytics.ranking_rows([anonymous]),
        analytics.tlx_rows([anonymous]),
        analytics.stage_rows([anonymous]),
    ):
        assert rows, "fixture stopped producing rows for one of these builders"
        assert {row["participant"] for row in rows} == {expected}


def test_a_task_keyed_rating_says_what_the_task_is_called():
    """The page groups the load ratings by task, and the raw key is not what a
    chart legend should say — the label the response stored is."""
    labelled = {row["task"]: row["task_label"] for row in analytics.tlx_rows([THREE_STAGE])}

    assert labelled == {
        "cooking": "Cook the meal",
        "set_table": "Set the table",
        "prepare_tray": "Prepare the tray",
    }
    # A layout that rates a session once has no task, so no label either: a
    # chart grouped by task would otherwise draw a category called "".
    assert {row["task_label"] for row in analytics.tlx_rows([PRODUCTION, CONTRASTIVE])} == {""}


def test_the_time_on_a_task_is_resolved_through_the_stage_it_sat_on():
    """A task arrives at a different step for every participant, because the
    stage order is drawn — so time per step is the wrong key for a question
    about tasks."""
    rows = analytics.task_duration_rows([THREE_STAGE])

    assert [(r["task"], r["stage"], r["seconds"]) for r in rows] == [
        ("cooking", 1, 300.0),
        ("prepare_tray", 3, 240.0),
    ]
    assert rows[0]["task_label"] == "Cook the meal"
    assert rows[0]["phase"] == "graph"
    # Stage 2's step has no recorded duration. No row, rather than a zero: an
    # unfinished session did not spend no time on the stage it never reached.
    assert "set_table" not in {row["task"] for row in rows}


def test_both_spellings_of_a_stage_step_are_read():
    """The layout calls them "Task 1 (…)" now and called them "Stage 1 (…)"
    when the first responses were collected, and a response carries the names it
    was written with."""
    renamed = dict(
        THREE_STAGE,
        step_names=[
            "Introduction",
            "Task 1 (behavior graph)",
            "Intermission",
            "Task 2 (rollout videos)",
            "Intermission",
            "Task 3 (rollout videos · your choice)",
            "Closing questions",
        ],
    )

    rows = analytics.task_duration_rows([renamed])

    assert [(r["task"], r["seconds"]) for r in rows] == [
        ("cooking", 300.0),
        ("prepare_tray", 240.0),
    ]


def test_time_on_task_needs_both_a_task_order_and_named_steps():
    """The layouts that show one task have nothing to say here, and a response
    whose steps are only numbered is not guessed at — attributing a duration to
    the wrong task would answer the question backwards."""
    assert analytics.task_duration_rows([PRODUCTION, CONTRASTIVE, ORDER_EFFECT]) == []
    numbered = {k: v for k, v in THREE_STAGE.items() if k != "step_names"}
    numbered = dict(numbered, evidence_order=[], study_variant="something_later_v1")
    assert analytics.step_labels(numbered) == []
    assert analytics.task_duration_rows([numbered]) == []


def test_a_malformed_task_duration_is_skipped_rather_than_charted():
    assert analytics.task_duration_rows([dict(THREE_STAGE, step_durations_seconds=[300.0])]) == []
    text = dict(THREE_STAGE, step_durations_seconds={"1": "five minutes", "5": 240.0})
    assert [row["task"] for row in analytics.task_duration_rows([text])] == ["prepare_tray"]


def test_every_row_says_which_attempt_it_came_from():
    """The live recruitment link carries no participant id, so every stored
    record's ``participant`` is "anon" and counting distinct ones reports one
    person however many there were. The attempt id is the key that works."""
    saved = dict(THREE_STAGE, _response_id="session_abc.json")

    assert {row["response_id"] for row in analytics.tlx_rows([saved])} == {"session_abc.json"}
    # A record that predates the field, or was never saved, falls back to its
    # timestamp rather than collapsing every attempt into one blank key.
    assert {row["response_id"] for row in analytics.tlx_rows([THREE_STAGE])} == {"20260827_101500"}
    assert {row["response_id"] for row in analytics.task_duration_rows([saved])} == {
        "session_abc.json"
    }
