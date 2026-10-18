"""What a node's name shortens to when it is drawn on the graph.

The rule these pin down is which halves of a name the picture already says. Get
it wrong in the generous direction and the tree is a field of colliding chips
again; get it wrong in the aggressive direction and a name someone wrote to be
read gets eaten.
"""

from __future__ import annotations

import pytest

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
    BehaviorNode,
    node_display_name,
)
from behavior_graphs.viz.labels import number_repeats, short_node_label
from behavior_graphs.viz.streamlit.user_study.graph_plot import _build_graph_json


class TestShortNodeLabel:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Behavior 0", "B0"),
            ("Behavior 12", "B12"),
            # A state-merging context copy: the suffix is the only thing telling
            # two copies apart, so it survives.
            ("Behavior 2 #2", "B2 #2"),
            ("+3 rare variants (7 eps)", "+3"),
            ("+1 rare variant (1 eps)", "+1"),
        ],
    )
    def test_the_redundant_half_goes(self, name, expected):
        assert short_node_label(name) == expected

    @pytest.mark.parametrize(
        "name",
        ["START", "SUCCESS", "FAILURE", "END", "FAILURE #2", "SUCCESS (unclustered)"],
    )
    def test_a_named_node_keeps_its_word(self, name):
        """Where a run began and whether it ended well are what the graph is asked.

        The ▶/★/✕/■ glyph says it too, and it is still worth the ink: a reader
        should not have to learn a glyph to find out where the failures went.
        """
        assert short_node_label(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "Grasp the milk carton firmly",
            "reach_for_handle",
            # "Behaviour 3" lives in test_only_generated_behavior_ids_are_shortened,
            # same input and same assertion; one copy is enough.
        ],
    )
    def test_a_name_someone_wrote_is_left_alone(self, name):
        """VLM annotations and experimenter-supplied names pass through whole."""
        assert short_node_label(name) == name

    def test_every_generated_name_is_handled(self):
        """The names :func:`node_display_name` produces are all accounted for."""
        for node_id in (START_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID):
            name = node_display_name(node_id)
            assert short_node_label(name) == name
        assert short_node_label(node_display_name(7)) == "B7"


def _graph() -> BehaviorGraph:
    nodes = {
        START_NODE_ID: BehaviorNode(
            cluster_id=START_NODE_ID, name="START", num_windows=0, num_episodes=10
        ),
        3: BehaviorNode(cluster_id=3, name="Behavior 3", num_windows=20, num_episodes=10),
        FAILURE_NODE_ID: BehaviorNode(
            cluster_id=FAILURE_NODE_ID, name="FAILURE", num_windows=0, num_episodes=10
        ),
    }
    return BehaviorGraph(
        nodes=nodes,
        transition_counts={START_NODE_ID: {3: 10}, 3: {FAILURE_NODE_ID: 10}},
        transition_probs={START_NODE_ID: {3: 1.0}, 3: {FAILURE_NODE_ID: 1.0}},
        num_episodes=10,
    )


class TestDrawnPayload:
    """The component is handed the short label; the tooltip keeps the full name."""

    def _nodes(self) -> dict[int, dict]:
        import json

        graph = _graph()
        pos = {nid: (float(i), 0.0) for i, nid in enumerate(graph.nodes)}
        payload = json.loads(_build_graph_json(graph, pos))
        return {n["id"]: n for n in payload["nodes"]}

    def test_the_drawn_label_is_short(self):
        assert self._nodes()[3]["label"] == "B3"

    def test_a_terminal_keeps_its_outcome(self):
        """The component appends the reach percentage after it — "FAILURE 93%"."""
        assert self._nodes()[FAILURE_NODE_ID]["label"] == "FAILURE"

    def test_start_keeps_its_name(self):
        assert self._nodes()[START_NODE_ID]["label"] == "START"

    def test_the_tooltip_still_names_the_node_in_full(self):
        """Shortening is for the drawing only — anything read keeps the name."""
        assert self._nodes()[3]["tooltip"].startswith("Behavior 3")
        assert self._nodes()[START_NODE_ID]["tooltip"] == "START"

    @pytest.mark.parametrize(
        "name",
        [
            "Behavior of the gripper",  # a name, not a generated id
            "Behavior changes at contact",
            "Behaviour 3",  # not the prefix we generate
        ],
    )
    def test_only_generated_behavior_ids_are_shortened(self, name):
        """`Behavior <digits>` is ours; `Behavior <word>` is someone's sentence."""
        assert short_node_label(name) == name


def test_a_figure_can_ask_for_the_unshortened_names():
    """ "B2" is for a real graph with no room for words; a small figure has it.

    The flag lives on the payload builder rather than on the component, so the
    shortening happens in exactly one place either way.
    """
    import json

    from behavior_graphs.viz.streamlit.user_study.graph_intro import _example_graph
    from behavior_graphs.viz.streamlit.user_study.graph_plot import _build_graph_json

    graph, pos = _example_graph()
    short = json.loads(_build_graph_json(graph, pos, prob_floor=0.0))
    verbatim = json.loads(_build_graph_json(graph, pos, prob_floor=0.0, shorten_labels=False))

    assert [n["label"] for n in short["nodes"] if n["id"] in (0, 1, 2)] == ["B1", "B2", "B3"]
    assert [n["label"] for n in verbatim["nodes"] if n["id"] in (0, 1, 2)] == [
        "Behavior 1",
        "Behavior 2",
        "Behavior 3",
    ]


# ── Numbering the names that repeat ──────────────────────────────────────────


def test_two_clusters_described_in_the_same_words_are_numbered():
    """A labeling model does this often enough to matter — a run of set_table
    has two clusters both called "reaching for the red can". Drawn that way the
    graph asserts something false: that one behavior appears twice, where in
    fact two distinct clusters do, with their own episodes and outcomes."""
    numbered = number_repeats({5: "reaching for the red can", 7: "reaching for the red can"})

    assert numbered == {
        5: "reaching for the red can (1)",
        7: "reaching for the red can (2)",
    }


def test_every_member_of_a_repeated_group_is_numbered():
    """Leaving the first one bare would read as the behavior with the rest as
    variants of it, and a reader could not tell an unnumbered name from one
    that never repeated at all."""
    numbered = number_repeats(dict.fromkeys((3, 1, 2), "stalling near the shelf"))

    assert sorted(numbered.values()) == [
        "stalling near the shelf (1)",
        "stalling near the shelf (2)",
        "stalling near the shelf (3)",
    ]
    # In cluster-id order, so the numbering does not move between reruns — and
    # a figure exported twice from one graph is the same figure.
    assert numbered[1].endswith("(1)")
    assert numbered[3].endswith("(3)")


def test_a_name_that_stands_alone_is_left_alone():
    names = {0: "grasping the yellow pumpkin", 1: "placing the chocolate milk"}

    assert number_repeats(names) == names


def test_numbering_does_not_touch_the_names_it_was_not_asked_about():
    """Only the repeats. A graph where one name in eleven repeats should differ
    from the un-numbered one in exactly two labels."""
    names = {i: f"behavior {i}" for i in range(9)}
    names[9] = names[10] = "reaching for the red can"

    numbered = number_repeats(names)

    assert [k for k in numbered if numbered[k] != names[k]] == [9, 10]
