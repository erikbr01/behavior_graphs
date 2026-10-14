"""Which policies the paper compares, and what they are called.

Configuration, so what is worth testing is that it is self-consistent. A typo in
a dataset name here is silent at runtime: the group quietly loses a policy, or a
policy quietly keeps its raw name, and both show up as a figure that went into a
paper with something missing from it.
"""

from __future__ import annotations

import pytest

from behavior_graphs.viz.streamlit import policy_groups as pg


def _group_of(dataset: str) -> str | None:
    """Which default group claims *dataset*, if any."""
    for label, patterns in pg.DEFAULT_GROUPS.items():
        if dataset in pg.resolve(patterns, [dataset]):
            return label
    return None


def test_every_named_policy_belongs_to_a_group():
    """A label for a dataset no group asks for is a label nothing will ever
    use — usually a typo, and one that leaves the real dataset showing its raw
    name in the figure."""
    orphans = [dataset for dataset in pg.PAPER_LABELS if _group_of(dataset) is None]

    assert orphans == []


def test_every_group_has_a_baseline_and_says_so_in_one_word():
    """A group is its own comparison, so "baseline" only has to be unambiguous
    within one panel — and three panels each with a "baseline" is easier to read
    across than three baselines called three different things."""
    by_group: dict[str, list[str]] = {}
    for dataset, label in pg.PAPER_LABELS.items():
        by_group.setdefault(_group_of(dataset), []).append(label)

    assert set(by_group) == set(pg.DEFAULT_GROUPS)
    for label, names in by_group.items():
        assert names.count(pg.BASELINE_LABEL) == 1, f"{label} has no single baseline"


def test_no_two_policies_in_a_group_share_a_label():
    """Two arms with one name are numbered apart downstream, which puts
    "baseline" and "baseline (2)" on a figure and tells the reader nothing about
    which was which."""
    by_group: dict[str, list[str]] = {}
    for dataset, label in pg.PAPER_LABELS.items():
        by_group.setdefault(_group_of(dataset), []).append(label)

    for label, names in by_group.items():
        assert len(set(names)) == len(names), f"{label} repeats a label"


def test_the_additional_policies_say_what_was_added():
    """The dataset name says where the extra demonstrations came from; the label
    says it in words a caption can use. Every non-baseline is a "plus"."""
    for dataset, label in pg.PAPER_LABELS.items():
        assert label == pg.BASELINE_LABEL or label.startswith("plus "), f"{dataset} -> {label}"


def test_a_pattern_takes_the_datasets_it_names_in_the_order_it_names_them():
    available = ["b_two", "a_one", "c_three"]

    assert pg.resolve(("a_*", "c_*"), available) == ["a_one", "c_three"]
    # De-duplicated: an arm drawn twice is a policy compared against itself.
    assert pg.resolve(("a_*", "a_one"), available) == ["a_one"]


def test_a_plain_name_matches_only_itself():
    """Which is how prepare_tray picks its three and leaves `_dp` and `_rgrl`
    out of the comparison."""
    available = ["prepare_tray_baseline", "prepare_tray_baseline_dp", "prepare_tray_baseline_rgrl"]

    assert pg.resolve(("prepare_tray_baseline",), available) == ["prepare_tray_baseline"]


def test_a_group_that_matches_nothing_here_is_left_out():
    """The page runs on machines with none of these datasets, and an empty
    group there would be three quarters of the screen explaining itself."""
    resolved = pg.default_groups(["set_table_baseline_pretrained_dit", "unrelated_dataset"])

    assert list(resolved) == ["Set table"]
    assert pg.default_groups(["nothing_matching"]) == {}


def test_the_groups_are_in_the_order_the_paper_shows_them():
    """The order is the figure's panel order, left to right, so it is a
    decision rather than a dict's accident."""
    assert list(pg.DEFAULT_GROUPS) == ["Set table", "Prepare tray", "Cooking"]


@pytest.mark.parametrize(
    "dataset", ["set_table_baseline_pretrained_dit", "cooking_baseline_lid_dit"]
)
def test_a_known_dataset_gets_its_paper_name_and_an_unknown_one_gets_none(dataset: str):
    assert pg.paper_label(dataset) == pg.PAPER_LABELS[dataset]
    assert pg.paper_label("some_run_collected_yesterday") is None


def test_the_baseline_label_is_what_the_baselines_are_called():
    """The Figures page pins the baseline to the left of every panel by looking
    for this label among a group's arms, having thrown the dataset names away by
    then. If the constant and the labels drift apart the pin silently stops
    finding anything and the panels go back to ranking the baseline with the
    rest."""
    assert pg.BASELINE_LABEL in set(pg.PAPER_LABELS.values())
