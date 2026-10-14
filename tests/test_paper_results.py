"""The exported results: the loader's contract, and the shipped file's content.

Two kinds of test, deliberately mixed in one file. The first few are about
:mod:`behavior_graphs.results` — what it does with a document it cannot read. The
rest are about ``paper_results.json`` itself, which is checked in and is what
the public site draws: a number that moves there moves the website, so it is
worth a few assertions that the file is internally consistent and still says
what the paper says.
"""

from __future__ import annotations

import json
import statistics

import pytest

from behavior_graphs import results

# ── The loader ───────────────────────────────────────────────────────────────


def test_a_missing_export_names_the_command_that_writes_it(tmp_path):
    """The one failure a deployment can hit: the file is simply not there."""
    with pytest.raises(results.ResultsUnavailable, match="export_paper_results"):
        results.load(tmp_path / "absent.json")


def test_a_document_from_another_schema_is_refused(tmp_path):
    """A field that has moved draws blank rather than loudly, so refuse first."""
    path = tmp_path / "paper_results.json"
    path.write_text(json.dumps({"schema": results.SCHEMA + 1, "study": {"participants": 18}}))
    with pytest.raises(results.ResultsUnavailable, match="schema"):
        results.load(path)


def test_an_absent_section_says_which_run_produces_it():
    """Sections come off three different machines, so three of four is normal."""
    with pytest.raises(results.ResultsUnavailable, match="--sections progress"):
        results.section({"schema": results.SCHEMA}, "progress")


def test_an_unknown_section_is_a_key_error_not_a_missing_one():
    """A typo in a page is a bug in the page, not a gap in the export."""
    with pytest.raises(KeyError):
        results.section({"schema": results.SCHEMA, "study": {"a": 1}}, "studies")


def test_pick_filters_on_every_field_given():
    table = [
        {"method": "graph_stage", "metric": "voc", "value": 0.9},
        {"method": "graph_stage", "metric": "mae", "value": 0.1},
        {"method": "mlp", "metric": "voc", "value": 0.7},
    ]
    assert results.pick(table, method="graph_stage", metric="voc") == [table[0]]


# ── The shipped export ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def document() -> dict:
    return results.load()


def test_every_section_the_site_draws_is_present(document):
    """Four pages, and three of them render nothing without their section."""
    for name in results.SECTIONS:
        assert results.section(document, name)


def test_each_policy_arm_is_a_tally_a_posterior_can_be_taken_over(document):
    for task in results.section(document, "policies")["tasks"]:
        for arm in task["arms"]:
            assert 0 <= arm["successes"] <= arm["trials"], (task["label"], arm["label"])
            assert arm["trials"] > 0, (task["label"], arm["label"])


def test_exactly_one_arm_per_task_is_the_baseline(document):
    """Every panel is read against its baseline, and `policy_table` and the
    violin colours both find it by this flag rather than by position."""
    for task in results.section(document, "policies")["tasks"]:
        assert sum(1 for arm in task["arms"] if arm["baseline"]) == 1, task["label"]


def test_the_best_arm_carries_the_first_letter(document):
    """The letters are named by success rate, so `a` is the best-performing
    policy in every panel and not the leftmost one. A page that drew them in
    display order would quietly relabel the comparison."""
    for task in results.section(document, "policies")["tasks"]:
        best = max(task["arms"], key=lambda arm: arm["successes"] / arm["trials"])
        assert "a" in best["letters"], task["label"]


def test_the_pvalue_matrix_matches_the_arms_it_is_indexed_by(document):
    for task in results.section(document, "policies")["tasks"]:
        size = len(task["arms"])
        assert len(task["pvalues"]) == size
        for row in task["pvalues"]:
            assert len(row) == size
        # A policy against itself is not a test, and a page that read the
        # diagonal as one would report every arm as separated from itself.
        assert all(task["pvalues"][i][i] is None for i in range(size))


def test_policy_arms_rebuilds_the_tally_the_paper_tested(document):
    task = results.section(document, "policies")["tasks"][0]
    arms = results.policy_arms(task)
    assert [arm.name for arm in arms] == [arm["label"] for arm in task["arms"]]
    assert [arm.trials for arm in arms] == [arm["trials"] for arm in task["arms"]]


def test_the_study_carries_one_pair_of_ratings_per_participant_and_outcome(document):
    """The paired panel needs exactly that, and a participant with one of the
    two would be drawn as a bubble on a pair that does not exist."""
    study = results.section(document, "study")
    seen: dict[tuple[str, str], set[str]] = {}
    for row in study["ratings"]:
        seen.setdefault((row["participant"], row["outcome"]), set()).add(row["display"])
    assert seen, "no ratings exported"
    assert all(displays == {"graph", "videos"} for displays in seen.values())
    assert len(seen) == study["participants"] * len(study["outcomes"])


def test_no_store_identifier_leaves_the_export(document):
    """The ids are re-keyed to P01… on the way out. A `session:` prefix here
    would be the store's own key on a page a paper's abstract links to."""
    study = results.section(document, "study")
    names = {row["participant"] for row in study["ratings"]}
    names |= {row["participant"] for row in study["regret"]}
    assert names, "no participants exported"
    assert all(name.startswith("P") and name[1:].isdigit() for name in names), sorted(names)


def test_the_preference_counts_add_up_to_the_participants_who_chose(document):
    preference = results.section(document, "study")["preference"]
    assert sum(preference["counts"].values()) == preference["n_chose"]


def test_a_choice_share_is_a_share_of_the_answers_its_display_drew(document):
    for row in results.section(document, "study")["choices"]:
        assert 0 <= row["picks"] <= row["answers"], row


def test_every_option_of_a_task_gets_a_row_per_display_that_answered_it(document):
    """Including the options nobody picked: an option at zero and an option
    that was never offered are different statements, and the bars have to keep
    them apart."""
    choices = results.section(document, "study")["choices"]
    per_display: dict[tuple[str, str], set[str]] = {}
    for row in choices:
        per_display.setdefault((row["task"], row["display"]), set()).add(row["dataset"])
    by_task: dict[str, set[str]] = {}
    for (task, _), datasets in per_display.items():
        by_task.setdefault(task, set()).update(datasets)
    for (task, display), datasets in per_display.items():
        assert datasets == by_task[task], (task, display)


def test_the_adjusted_rows_carry_an_interval_around_their_estimate(document):
    rows = results.section(document, "adjusted")["rows"]
    assert {row["outcome"] for row in rows} >= {"temporal_demand", "effort", "confidence"}
    for row in rows:
        assert row["lower"] <= row["difference"] <= row["upper"], row["outcome"]
        # The difference is the two marginal means apart, which is what lets a
        # table print all three and a reader check one against the others.
        assert row["difference"] == pytest.approx(row["mean_graph"] - row["mean_videos"], abs=0.05)


def test_the_chains_agreed_on_one_posterior(document):
    """Carried on every row so a re-export that did not converge is visible
    here rather than only in the log of whoever ran it."""
    for row in results.section(document, "adjusted")["rows"]:
        assert row["r_hat"] <= 1.01, row["outcome"]


def test_the_graph_reduces_perceived_time_and_effort_and_changes_neither_decision(document):
    """The paper's claim, as the three signs that make it.

    Not the digits: those move when the store does. The signs and whether each
    interval excludes zero are the result, and an export that reversed one of
    them is an export nobody should publish without noticing.
    """
    rows = {row["outcome"]: row for row in results.section(document, "adjusted")["rows"]}
    for outcome in ("temporal_demand", "effort"):
        assert rows[outcome]["upper"] < 0, outcome  # lower off the graph, and separated
    for outcome in ("confidence", "regret"):
        assert rows[outcome]["lower"] < 0 < rows[outcome]["upper"], outcome  # unchanged


def test_the_progress_benchmark_counts_only_datasets_every_method_scored(document):
    """A mean over whatever each method happened to be run on is not a
    comparison, and the VLM baselines were run on a subset of the zoo."""
    progress = results.section(document, "progress")
    methods = [method["key"] for method in progress["methods"]]
    per_dataset: dict[str, set[str]] = {}
    for row in progress["rows"]:
        per_dataset.setdefault(row["dataset"], set()).add(row["method"])
    assert len(per_dataset) == progress["datasets"]
    assert all(found == set(methods) for found in per_dataset.values())


def test_each_domains_dataset_count_is_the_weight_its_mean_was_taken_with(document):
    progress = results.section(document, "progress")
    counted = {domain["label"]: domain["datasets"] for domain in progress["domains"]}
    for row in progress["means"]:
        if row["domain"] == "Average":
            assert row["datasets"] == progress["datasets"]
        else:
            assert row["datasets"] == counted[row["domain"]], row["domain"]


def test_a_domain_mean_is_the_mean_of_its_rows(document):
    """The means are exported rather than computed by the page, so this is the
    one place the two can be held to each other."""
    progress = results.section(document, "progress")
    for mean in progress["means"]:
        for metric in (m["key"] for m in progress["metrics"]):
            values = [
                row[metric]
                for row in progress["rows"]
                if row["method"] == mean["method"]
                and metric in row
                and (mean["domain"] == "Average" or row["domain"] == mean["domain"])
            ]
            assert mean[f"{metric}_datasets"] == len(values), (mean["domain"], metric)
            if values:
                assert mean[metric] == pytest.approx(statistics.fmean(values)), (
                    mean["domain"],
                    mean["method"],
                    metric,
                )


def test_the_graph_leads_the_average_on_the_three_fidelity_metrics(document):
    """The paper's second claim. AUROC is left out on purpose: a baseline leads
    it, which the site says, and a test asserting otherwise would be asserting
    the wrong thing."""
    progress = results.section(document, "progress")
    average = {row["method"]: row for row in progress["means"] if row["domain"] == "Average"}
    ours = next(method["key"] for method in progress["methods"] if method["ours"])
    others = [key for key in average if key != ours]
    for metric in ("voc", "kendall"):
        assert average[ours][metric] > max(average[key][metric] for key in others), metric
    assert average[ours]["mae"] < min(average[key]["mae"] for key in others)


# ── The curves file ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def curves() -> dict:
    return results.load_curves()


def test_a_missing_curves_file_names_the_section_that_writes_it(tmp_path):
    with pytest.raises(results.ResultsUnavailable, match="--sections progress"):
        results.load_curves(tmp_path / "absent.json")


def test_the_curves_cover_exactly_the_collections_the_table_counts(document, curves):
    """Both come off one ``--sections progress`` run, so a checkout holding a
    document and a curves file that disagree is one where somebody wrote one of
    them by hand."""
    progress = results.section(document, "progress")
    counted = {row["dataset"] for row in progress["rows"]}
    drawn = {entry["dataset"] for entry in curves["datasets"]}
    assert drawn == counted
    assert len(drawn) == progress["datasets"]


def test_every_curve_is_the_length_of_the_grid(curves):
    """A chart plots a curve against the grid with no length check of its own."""
    width = len(curves["grid"])
    assert width > 1
    for entry in curves["datasets"]:
        for rollout in entry["rollouts"]:
            for method, values in rollout["curves"].items():
                assert len(values) == width, (entry["dataset"], rollout["idx"], method)


def test_a_rollout_carries_its_index_and_outcome(curves):
    """The index is how the page finds the rollout's video and the outcome is
    what the reference diagonal means on it."""
    for entry in curves["datasets"]:
        for rollout in entry["rollouts"]:
            assert isinstance(rollout["idx"], int)
            assert isinstance(rollout["success"], bool)


def test_no_curve_is_an_aggregate_across_rollouts(curves):
    """An earlier export carried a median across rollouts with a 10-to-90 band
    around it. Both are quantities the method never produces: the estimator is
    deterministic and gives exactly one curve per rollout, so the band was
    spread between *rollouts* in the costume of a confidence interval.
    Guarded in the data, so it cannot quietly come back and be drawn."""
    for entry in curves["datasets"]:
        assert "series" not in entry, entry["dataset"]
        for rollout in entry["rollouts"]:
            assert set(rollout) == {"idx", "success", "curves"}, entry["dataset"]
            assert not {"mid", "lo", "hi"} & set(rollout["curves"]), entry["dataset"]


def test_progress_stays_within_the_unit_interval(curves):
    for entry in curves["datasets"]:
        for rollout in entry["rollouts"]:
            for values in rollout["curves"].values():
                assert min(values) >= 0.0, entry["dataset"]
                assert max(values) <= 1.0, entry["dataset"]


def test_both_outcomes_are_offered_wherever_the_collection_has_both(curves):
    """The diagonal means different things on the two, so a browser that only
    ever showed successes would show half the reading."""
    for entry in curves["datasets"]:
        outcomes = {rollout["success"] for rollout in entry["rollouts"]}
        if 0 < entry["successes"] < entry["episodes"]:
            assert outcomes == {True, False}, entry["dataset"]


def test_every_rollout_was_scored_by_every_method(curves):
    """A chart missing one of its four lines invites reading the gap as a
    result."""
    for entry in curves["datasets"]:
        per_rollout = {frozenset(r["curves"]) for r in entry["rollouts"]}
        assert len(per_rollout) == 1, entry["dataset"]
        assert len(next(iter(per_rollout))) == 4, entry["dataset"]


def test_the_graph_estimator_has_a_curve_for_every_rollout(curves):
    """It is the line the chart is built around; a rollout missing it would
    draw three baselines against nothing."""
    for entry in curves["datasets"]:
        for rollout in entry["rollouts"]:
            assert "graph_stage" in rollout["curves"], (entry["dataset"], rollout["idx"])
