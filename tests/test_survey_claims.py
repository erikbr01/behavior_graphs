"""The claims the paper may make from the study, and their denominators.

What breaks here is not the arithmetic but the *set* it is over: the store holds
earlier layouts, abandoned sessions and the lab app's previews, and a pilot
session that ran to the end looks complete by every test the store can apply.
So most of what is pinned below is who counts — and, for the ratings, that the
comparison stays within the participant rather than becoming a between-groups
one by accident.
"""

from __future__ import annotations

from behavior_graphs.viz.streamlit.user_study import claims, figure_export, regret

# pytest puts this file's directory on sys.path (tests/ is not a package).
from test_survey_analytics import PRODUCTION, THREE_STAGE


def _decision_row(task: str, phase: str, gain, best: bool, regret_points: float = 0.0) -> dict:
    """One scored choice.

    Both quantities, because the summaries report both and for different
    reasons: regret is the decision measure and is always defined, gain answers
    "did this help at all" and is ``None`` where the task's baseline has no
    mounted rollouts.
    """
    return {
        "task": task,
        "phase": phase,
        "regret": regret_points,
        "gain_over_baseline": gain,
        "chose_best": best,
    }


#: A three-stage response whose ratings are on the slider's own grid. The study
#: asks for these in steps of 20 and the modelled rows drop anything off it, so
#: `THREE_STAGE`'s 30s and 50s are read as no rating at all -- realistic of a
#: hand-written fixture and not of a participant.
RATED = dict(
    THREE_STAGE,
    participant_id="p-rated",
    nasa_tlx={
        # cooking is the graph stage, set_table the assigned videos one.
        "cooking": {"temporal_demand": 20, "effort": 40, "confidence": 80},
        "set_table": {"temporal_demand": 40, "effort": 40, "confidence": 60},
        "prepare_tray": {"temporal_demand": 100, "effort": 100, "confidence": 0},
    },
)


# ── who is in the denominator ───────────────────────────────────────────────


def test_only_the_reported_layouts_finished_participants_count():
    """An earlier layout's session is the trap: it ran to the end, so it is
    complete by any test the store can apply, while having asked different
    questions of a different person. Two of them in the denominator turned
    "16 of 18" into "80% of 20"."""
    kept = claims.study_responses(
        [
            THREE_STAGE,
            PRODUCTION,
            dict(THREE_STAGE, participant_id="p-unfinished", status="in_progress"),
            dict(THREE_STAGE, participant_id="p-preview", origin="lab"),
        ]
    )

    assert kept == [THREE_STAGE]


def test_narrowing_twice_is_the_same_as_narrowing_once():
    """Callers narrow before they hand the responses over, and the claims
    narrow again rather than trusting them to have done it."""
    once = claims.study_responses([THREE_STAGE, PRODUCTION])

    assert claims.study_responses(once) == once


# ── preference ──────────────────────────────────────────────────────────────


def test_preference_counts_only_the_reported_participants():
    claim = claims.preference_claim([THREE_STAGE, PRODUCTION])

    # Only the three-stage layout has a free-choice stage at all, and only its
    # participant is in the denominator.
    assert claim["counts"] == {"graph": 0, "videos": 1}
    assert claim["n_chose"] == 1
    assert claim["n_participants"] == 1
    assert claim["share"]["videos"] == 100.0


def test_a_store_with_no_free_choice_claims_nothing():
    claim = claims.preference_claim([PRODUCTION])

    assert claim["n_chose"] == 0
    assert claim["n_participants"] == 0
    # No division by zero and no 0% that reads as a measured zero.
    assert claim["share"] == {"graph": None, "videos": None}


def test_preference_agrees_with_the_figure_beside_it():
    """The claim reads the chosen stage through the same function the figure's
    preference panel does; a second reading of `chosen_modality` drifted from it
    the first time the study's stage keys changed."""
    from behavior_graphs.viz.streamlit.user_study import figure_export

    responses = [THREE_STAGE, PRODUCTION]

    assert claims.preference_claim(responses)["counts"] == figure_export.preference_counts(
        responses
    )


# ── ratings ─────────────────────────────────────────────────────────────────


def test_ratings_are_compared_within_the_participant():
    rows = {row["outcome"]: row for row in claims.paired_rating_claims([RATED])}

    # cooking was the graph stage (20), set_table the assigned videos one (40);
    # the third stage was the participant's own pick, so it answers a preference
    # question rather than a randomized one and is left out.
    assert rows["temporal_demand"]["n_pairs"] == 1
    assert rows["temporal_demand"]["mean_graph"] == 20.0
    assert rows["temporal_demand"]["mean_videos"] == 40.0
    assert rows["temporal_demand"]["difference"] == -20.0
    assert rows["temporal_demand"]["lower"] == 1
    assert rows["temporal_demand"]["higher"] == 0
    assert rows["temporal_demand"]["tied"] == 0


def test_a_participant_who_rated_both_the_same_is_tied_and_not_dropped():
    """A tie is evidence — it is what "no difference" looks like — so it counts
    in `n_pairs` and in neither direction."""
    effort = next(row for row in claims.paired_rating_claims([RATED]) if row["outcome"] == "effort")

    assert effort["n_pairs"] == 1
    assert effort["difference"] == 0.0
    assert (effort["lower"], effort["higher"], effort["tied"]) == (0, 0, 1)


def test_an_outcome_nobody_rated_is_absent_rather_than_zero():
    """A confidence row of 0.0 in a table of results reads as a measurement,
    and this participant was never asked."""
    unasked = dict(
        RATED,
        nasa_tlx={
            task: {key: value for key, value in scores.items() if key != "confidence"}
            for task, scores in RATED["nasa_tlx"].items()
        },
    )
    outcomes = [row["outcome"] for row in claims.paired_rating_claims([unasked])]

    assert outcomes == ["temporal_demand", "effort"]
    assert "confidence" in claims.CLAIM_OUTCOMES


def test_outcomes_come_back_in_the_order_the_paper_reports_them():
    ordered = [row["outcome"] for row in claims.paired_rating_claims([RATED])]

    assert ordered == list(claims.CLAIM_OUTCOMES)


def test_a_participant_missing_one_side_is_left_out_of_the_pairs():
    """Half a pair is not a within-participant comparison. Comparing it against
    the other display's group mean is the between-groups claim this module
    exists to avoid making by accident."""
    one_sided = dict(
        RATED,
        participant_id="p-one-sided",
        # Only the graph stage keeps its assignment.
        modality_assignment={
            "cooking": "assigned",
            "set_table": "chosen",
            "prepare_tray": "chosen",
        },
    )
    paired = claims.paired_rating_claims([RATED, one_sided])

    assert {row["n_pairs"] for row in paired} == {1}


def test_both_the_points_and_the_relative_figure_come_back():
    rows = {row["outcome"]: row for row in claims.paired_rating_claims([RATED])}

    # 20 against 40 is -20 points of the 0-100 scale the participant picked on,
    # and -50% *of the videos mean*. Both are returned, and the second is only
    # as good as the sentence around it -- a percentage of a rating, said of the
    # rating and not of the thing rated.
    assert rows["temporal_demand"]["difference"] == -20.0
    assert rows["temporal_demand"]["relative"] == -50.0


# ── the fitted estimates ────────────────────────────────────────────────────


def _effect(outcome: str, mean: float, videos: float, graph: float) -> dict:
    """One `mixed_effects.continuous_effects` row, with its marginal means."""
    return {
        "outcome": outcome,
        "mean": mean,
        "lower": mean - 12.0,
        "upper": mean + 12.0,
        "r_hat": 1.0,
        "marginal_means": {
            "videos": {"mean": videos, "lower": videos - 8, "upper": videos + 8},
            "graph": {"mean": graph, "lower": graph - 8, "upper": graph + 8},
        },
    }


def test_the_adjusted_percentage_is_of_the_adjusted_mean():
    """The whole reason this function exists. A sentence carrying the fit's
    interval has to carry the fit's percentage, and the two divisors differ:
    -26.1 of the adjusted videos mean (64.8) is -40.3%, of the raw one (64.4)
    it is -40.5%. Close enough that mixing them is never noticed, which is the
    argument for not leaving it to whoever writes the sentence."""
    row = claims.adjusted_rating_claims([_effect("temporal_demand", -26.1, 64.8, 38.7)])[0]

    assert row["difference"] == -26.1
    assert (round(row["lower"], 1), round(row["upper"], 1)) == (-38.1, -14.1)
    assert round(row["relative"], 1) == -40.3
    assert row["label"] == "Temporal demand"


def test_a_fit_without_marginal_means_reports_no_percentage():
    """`None` rather than a percentage of the raw mean: a relative figure that
    silently changed divisor is worse than one that is absent."""
    row = claims.adjusted_rating_claims(
        [{"outcome": "effort", "mean": -22.9, "lower": -37.7, "upper": -7.8}]
    )[0]

    assert row["difference"] == -22.9
    assert row["relative"] is None
    assert row["mean_graph"] is None and row["mean_videos"] is None


def test_the_fitted_and_paired_claims_have_the_same_shape():
    """They are rendered by the same table code and read against each other, so
    a key present in one and missing from the other is a column of em dashes."""
    fitted = claims.adjusted_rating_claims([_effect("effort", -22.9, 61.0, 38.1)])[0]
    paired = claims.paired_rating_claims([RATED])[0]

    assert {"outcome", "label", "mean_graph", "mean_videos", "difference", "relative"} <= (
        set(fitted) & set(paired)
    )


def test_the_decision_is_a_row_of_the_same_table_as_the_ratings():
    """The outcome the study is about, in the table a sentence is written from,
    rather than a number to be read off a figure. Same estimand as the rows
    above it -- the `graph` coefficient and each display's adjusted mean -- so
    the same keys, or the column comes out as em dashes."""
    row = claims.adjusted_decision_claim(
        {
            "mean": 0.9,
            "lower": -6.1,
            "upper": 8.5,
            "r_hat": 1.0,
            "marginal_means": {
                "graph": {"mean": 7.9, "lower": 3.0, "upper": 12.8},
                "videos": {"mean": 7.0, "lower": 2.2, "upper": 12.1},
            },
        }
    )

    assert row is not None
    # The figures' own name for it, so a table row and the panel it refers to
    # cannot come to call one quantity two things.
    assert row["label"] == figure_export.STUDY_PANEL_TITLES["regret"] == "Success rate regret"
    assert row["outcome"] == regret.DECISION_METRIC
    assert (row["mean_graph"], row["mean_videos"]) == (7.9, 7.0)
    assert row["difference"] == 0.9
    # Of the adjusted videos mean, as the rating rows are -- a ratio of two
    # small numbers here, which is what the interval beside it is for.
    assert round(row["relative"], 1) == 12.9
    assert set(row) == set(claims.adjusted_rating_claims([_effect("effort", -22.9, 61.0, 38.1)])[0])


def test_a_deployment_that_cannot_score_a_choice_reports_no_decision_row():
    """Scoring one needs every option's retrained policy on disk. No row rather
    than a row of em dashes, which reads as a fit that came back empty."""
    assert claims.adjusted_decision_claim(None) is None
    assert claims.adjusted_decision_claim({}) is None
    assert claims.adjusted_decision_claim({"lower": -1.0, "upper": 1.0}) is None


def test_the_decision_row_reports_no_percentage_without_a_marginal_mean():
    """As the rating rows do: a relative figure with no divisor of its own
    would silently be of something else."""
    row = claims.adjusted_decision_claim({"mean": 0.9, "lower": -6.1, "upper": 8.5})

    assert row["difference"] == 0.9
    assert row["relative"] is None
    assert row["mean_graph"] is None and row["mean_videos"] is None


# ── decisions ───────────────────────────────────────────────────────────────


def test_decisions_are_reported_pooled_by_display_and_by_task():
    rows = [
        _decision_row("cooking", "graph", -5.0, False, regret_points=8.0),
        _decision_row("cooking", "videos", -10.0, False, regret_points=13.0),
        _decision_row("prepare_tray", "graph", 20.0, True, regret_points=0.0),
        _decision_row("prepare_tray", "videos", 10.0, False, regret_points=10.0),
    ]
    found = claims.decision_claims(rows)

    assert found["pooled"]["n"] == 4
    # Regret is the decision measure; gain answers the separate "did it help".
    assert found["pooled"]["mean_regret"] == 7.75
    assert found["pooled"]["share_positive"] == 50.0
    assert found["by_display"]["graph"]["mean_regret"] == 4.0
    assert found["by_task"]["cooking"]["share_positive"] == 0.0
    assert found["by_task"]["prepare_tray"]["share_positive"] == 100.0


def test_the_display_comparison_leaves_out_the_stage_they_chose():
    """The three-stage layout's last stage is the participant's own pick, and
    on the live store 16 of 18 picked the graph -- so pooling it in makes the
    display a result of the participant rather than something drawn for them.
    It also reverses the answer, which is what this pins: the graph's mean gain
    is +20 over the randomized choices and +2.5 with the chosen one folded in,
    while the pooled and per-task rows keep every choice, because "was this
    choice worth anything" is a question the chosen stage can answer too."""
    rows = [
        _decision_row("prepare_tray", "graph", 20.0, True, regret_points=0.0),
        _decision_row("prepare_tray", "videos", 10.0, False, regret_points=10.0),
        dict(
            _decision_row("cooking", "graph", -15.0, False, regret_points=18.0),
            assignment="chosen",
        ),
    ]
    found = claims.decision_claims(rows)

    assert found["n_randomized"] == 2
    assert found["by_display"]["graph"]["n"] == 1
    assert found["by_display"]["graph"]["mean_regret"] == 0.0
    assert found["pooled"]["n"] == 3
    assert found["by_task"]["cooking"]["n"] == 1


def test_a_layout_that_assigned_nothing_still_compares_its_displays():
    """No `assignment` means no free choice to contaminate the comparison --
    the layouts before the three-stage one showed both displays to everyone.
    Treating a missing field as "chosen" would empty their display rows."""
    rows = [
        _decision_row("cooking", "graph", 4.0, True, regret_points=0.0),
        _decision_row("cooking", "videos", 2.0, False, regret_points=2.0),
    ]
    found = claims.decision_claims(rows)

    assert found["n_randomized"] == 2
    assert found["by_display"]["graph"]["mean_regret"] == 0.0


def test_tasks_are_reported_in_the_studys_order():
    rows = [
        _decision_row("prepare_tray", "graph", 1.0, True),
        _decision_row("cooking", "graph", 1.0, True),
    ]

    assert list(claims.decision_claims(rows)["by_task"]) == [
        task for task in regret.TASK_ORDER if task in {"cooking", "prepare_tray"}
    ]


def test_a_choice_with_no_scored_baseline_still_counts_as_a_choice():
    """`gain_over_baseline` is None when the task's baseline policy has no
    rollouts mounted. That choice was still made and can still have picked the
    best option, so it belongs in the `chose_best` denominator and not in the
    gain one -- averaging it in as a zero would report a decision as neutral
    when it was unmeasured."""
    rows = [
        _decision_row("cooking", "graph", None, True, regret_points=0.0),
        _decision_row("cooking", "graph", 10.0, True, regret_points=0.0),
    ]
    found = claims.decision_claims(rows)["pooled"]

    assert found["n"] == 2
    # Regret is scored for both; only one has a baseline to have gained over.
    assert found["mean_regret"] == 0.0
    assert found["share_positive"] == 100.0  # of the one scored choice
    assert found["share_best"] == 100.0  # of both choices


def test_nothing_scored_claims_nothing_rather_than_dividing_by_zero():
    found = claims.decision_claims(
        [{"task": "cooking", "phase": "graph", "gain_over_baseline": None, "chose_best": False}]
    )

    assert found["pooled"]["mean_regret"] is None
    assert found["pooled"]["share_positive"] is None
