"""How the tree divides its width between labels, and why wrapping helps.

The graph never draws wider than its frame, so a level's width is a fixed pot
split between siblings in proportion to what their labels need
(:func:`_label_demand`), and wrapping is what lowers what a long name needs.
These tests pin both halves.
"""

from __future__ import annotations

import pytest

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (  # noqa: E402
    _LABEL_MAX_LINES,
    _LABEL_MAX_PX,
    TERMINAL_NODE_IDS,
    _label_demand,
    _wrap_label,
)


def test_a_short_label_stays_on_one_line():
    assert _wrap_label("Grasp") == ["Grasp"]


def test_a_long_label_wraps_at_a_space():
    lines = _wrap_label("Grasp the milk carton firmly")

    assert len(lines) == 2
    # Nothing dropped, and the break is at a space.
    assert " ".join(lines) == "Grasp the milk carton firmly"


def test_a_label_too_long_for_two_lines_keeps_its_words_rather_than_losing_them():
    text = " ".join(["reallyquitelongword"] * 6)

    lines = _wrap_label(text)

    assert len(lines) == _LABEL_MAX_LINES
    assert " ".join(lines) == text


def test_an_unbreakable_label_cannot_wrap():
    text = "Grasp_the_milk_carton_firmly"

    assert _wrap_label(text) == [text]


def _nodes(*names: str) -> tuple[list[dict], dict]:
    """A root with one child per name, each child a leaf."""
    nodes = [{"path": (0,), "cluster_id": 0, "label": "root"}]
    for i, name in enumerate(names):
        nodes.append({"path": (0, i + 1), "cluster_id": i + 1, "label": name})
    children_of = {(0,): [(0, i + 1) for i in range(len(names))]}
    return nodes, children_of


def test_a_parent_needs_the_sum_of_what_its_children_need():
    nodes, children_of = _nodes("Grasp", "Rotate", "Nudge")

    demand = _label_demand(nodes, children_of)

    assert demand[(0,)] == sum(demand[(0, i)] for i in (1, 2, 3))


def test_wrapping_a_long_name_lowers_what_its_branch_asks_for():
    spaced, children_of = _nodes("Grasp the milk carton firmly")
    unbreakable, u_children = _nodes("Grasp_the_milk_carton_firmly")

    wrapped = _label_demand(spaced, children_of)[(0, 1)]
    one_line = _label_demand(unbreakable, u_children)[(0, 1)]

    # Same characters; the one that can wrap needs roughly half the width.
    assert wrapped < one_line * 0.65
    assert wrapped <= _LABEL_MAX_PX + 12.0  # the wrap target, plus the sibling gap


def _terminal_demand(name: str) -> float:
    nodes, children_of = _nodes(name)
    nodes[1]["cluster_id"] = next(iter(TERMINAL_NODE_IDS))  # what marks it terminal
    return _label_demand(nodes, children_of)[(0, 1)]


def test_a_terminal_leaves_room_for_the_percentage_appended_in_the_browser():
    """Budgeting for text the layout never sees — the browser appends " 20%"."""
    # Stacked below a short outcome word, the percentage is the wider of the two
    # lines, so it is what sets the width.
    assert _terminal_demand("END") > _label_demand(*_nodes("END"))[(0, 1)]


def test_stacking_a_terminal_makes_its_percentage_horizontally_free():
    """Outcome above share, so a leaf asks for the wider line, not their sum.

    Terminals are the most numerous labels at a tree's deepest levels, and
    charging each of them for "FAILURE" *plus* " 12%" is what made those levels
    collide. Mirrors LABEL_WRAP_ALWAYS in the component.
    """
    assert _terminal_demand("SUCCESS") == _label_demand(*_nodes("SUCCESS"))[(0, 1)]


def test_a_cycle_does_not_hang_the_layout():
    nodes, children_of = _nodes("Grasp")
    children_of[(0, 1)] = [(0,)]  # malformed: child points back at the root

    demand = _label_demand(nodes, children_of)

    assert demand[(0,)] > 0
