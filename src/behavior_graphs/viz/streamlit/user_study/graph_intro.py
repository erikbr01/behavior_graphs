"""The graph's key, drawn as a small example of the graph itself.

This started as a one-time dialog: a participant read it once, dismissed it, and
had no way back to it while working. It is now part of the page, and it is a
figure rather than prose — a four-behavior tree drawn by the same component the
real graph is drawn by, with callout lines naming each channel where it happens.
"""

from __future__ import annotations

import streamlit as st

from behavior_graphs.core.graph import START_NODE_ID, BehaviorGraph, BehaviorNode
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import outcome_color
from behavior_graphs.viz.streamlit.user_study.graph_plot import render_graph_component

# Ids for the example's own terminals. The study draws a *tree*: each leaf ends
# in its own ✓ or ✕, rather than every branch running into one shared pair of
# outcome nodes. Reusing SUCCESS_NODE_ID/FAILURE_NODE_ID would draw the shared
# kind and teach the wrong shape, so the figure uses synthetic ids with a symbol
# override, exactly as the trajectory tree does.
#
# Both outcomes hang off *both* branches, because that is what the real tree
# draws: a behavior is a step on the way, not a verdict, and one that usually
# works still fails sometimes. With a single ✓ under the common branch and a
# single ✕ under the rare one, the figure read as "this branch is the good one",
# and a participant who learned that would misread every mixed node in the graph
# below it.
_LEAF_OK_COMMON = 100
_LEAF_FAIL_COMMON = 101
_LEAF_OK_RARE = 102
_LEAF_FAIL_RARE = 103

_SUCCESS_LEAVES = frozenset({_LEAF_OK_COMMON, _LEAF_OK_RARE})

# Rollouts through each node — counts, not drawing instructions. The component
# derives every radius and band width from them exactly as it does for the real
# graph, so the figure cannot encode something the graph does not.
#
# 30 of them rather than the 20 this figure used to run on: the rare branch has
# to hold a ✓ and a ✕ now, and it can only stay small *and* stay red if its one
# success is a small enough share to fall in the bottom color bin.
_COUNTS = {
    START_NODE_ID: 30,
    0: 30,
    1: 24,
    2: 6,
    _LEAF_OK_COMMON: 20,
    _LEAF_FAIL_COMMON: 4,
    _LEAF_OK_RARE: 1,
    _LEAF_FAIL_RARE: 5,
}

_TRANSITIONS = {
    START_NODE_ID: {0: 30},
    0: {1: 24, 2: 6},
    1: {_LEAF_OK_COMMON: 20, _LEAF_FAIL_COMMON: 4},
    2: {_LEAF_OK_RARE: 1, _LEAF_FAIL_RARE: 5},
}

_NAMES = {
    START_NODE_ID: "START",
    0: "Behavior 1",
    1: "Behavior 2",
    2: "Behavior 3",
    _LEAF_OK_COMMON: "SUCCESS",
    _LEAF_FAIL_COMMON: "FAILURE",
    _LEAF_OK_RARE: "SUCCESS",
    _LEAF_FAIL_RARE: "FAILURE",
}

# Two leaves per branch, success above failure in both — the trajectory tree
# lines its ✓ terminals up the same way, so the figure's shape is the shape a
# participant then meets.
_POSITIONS = {
    START_NODE_ID: (0.0, 0.5),
    0: (0.26, 0.5),
    1: (0.6, 0.15),
    2: (0.6, 0.85),
    _LEAF_OK_COMMON: (1.0, 0.0),
    _LEAF_FAIL_COMMON: (1.0, 0.3),
    _LEAF_OK_RARE: (1.0, 0.7),
    _LEAF_FAIL_RARE: (1.0, 1.0),
}


def _successes_below(node_id: int) -> int:
    """Rollouts that reached a ✓ from *node_id*, itself included if it is one."""
    if node_id in _SUCCESS_LEAVES:
        return _COUNTS[node_id]
    return sum(_successes_below(target) for target in _TRANSITIONS.get(node_id, {}))


def _share_of_all(node_id: int) -> float:
    """*node_id*'s rollouts as a share of every rollout in the figure."""
    return _COUNTS[node_id] / _COUNTS[START_NODE_ID]


def _example_graph() -> tuple[BehaviorGraph, dict[int, tuple[float, float]]]:
    """A tree: one behavior branching into a common and a rare one, each ending
    in both outcomes.
    """
    nodes = {
        nid: BehaviorNode(
            cluster_id=nid,
            name=_NAMES[nid],
            num_windows=n,
            num_episodes=n,
            episode_indices=list(range(n)),
        )
        for nid, n in _COUNTS.items()
    }
    transition_probs = {
        src: {tgt: n / sum(outs.values()) for tgt, n in outs.items()}
        for src, outs in _TRANSITIONS.items()
    }
    graph = BehaviorGraph(
        nodes=nodes,
        transition_counts=_TRANSITIONS,
        transition_probs=transition_probs,
        num_episodes=_COUNTS[START_NODE_ID],
    )
    return graph, dict(_POSITIONS)


def _example_colors() -> dict[int, str]:
    """Each node in the color its own success rate earns it.

    Read off the counts through :func:`outcome_color`, the same function the
    real graph colors its nodes with, rather than picked by hand: a figure whose
    Behavior 2 is greener than its share of ✓ warrants is teaching a scale the
    graph below it does not use.
    """
    return {
        nid: outcome_color(_successes_below(nid) / _COUNTS[nid])
        for nid in (0, 1, 2, *_SUCCESS_LEAVES, _LEAF_FAIL_COMMON, _LEAF_FAIL_RARE)
    }


def _example_annotations() -> list[dict]:
    """Four callouts covering every channel the drawing uses.

    Two channels per branch node rather than a callout each: a node is large
    *and* green, and separating the two into labels on opposite sides of one
    circle crowded the figure without making either clearer. The two bands get
    one each: the contrast between them is the encoding, so a single callout
    would leave the reader to infer the other half.

    One leaf is called out and the other three are not. The component already
    writes SUCCESS / FAILURE and a percentage beside every terminal, but a bare
    "67%" does not say what it is a percentage *of* — and a reader who takes it
    for "67% of the rollouts that got this far" has misread the one number the
    graph is for. Said once, on the largest terminal, it reads as the key to all
    four rather than as four repetitions of the same sentence.
    """
    return [
        # Offset sideways as well as up, so the leader line is a diagonal:
        # straight above a node, a label reads as that node's name rather than
        # as a note about it.
        #
        # To the *left*, though the node is the figure's right-hand column: the
        # room to its right belongs to the four terminals and the SUCCESS /
        # FAILURE captions under them. Anchored right of the node, these two ran
        # straight into those captions on any window narrower than about
        # 1300px — which is most of them, since the figure is drawn at
        # three-quarters of the page.
        {
            "node": 1,
            "dx": -66,
            "dy": -58,
            "wrap": 30,
            "text": "large and green: many rollouts came here, and most succeeded",
        },
        {
            # Clears the node's own name chip, which sits between the two.
            "node": 2,
            "dx": -66,
            "dy": 40,
            "wrap": 30,
            "text": "small and red: few rollouts came here, and nearly all failed",
        },
        {
            # Up and to the left of the band: directly above it, this label and
            # the one over the common node abutted into one four-line block;
            # directly below, it ran across the branch node and the thin edge.
            "edge": [0, 1],
            "dx": -70,
            "dy": -32,
            "wrap": 24,
            "text": "wide band: most rollouts took this step",
        },
        {
            # Mirrored below the other branch, so the two band callouts read as
            # a pair: the contrast between them is the encoding.
            "edge": [0, 2],
            "dx": -84,
            "dy": 48,
            "wrap": 24,
            "text": "slim band: few rollouts took this one",
        },
        {
            # Up and to the *left*, into the top padding. The room to the right
            # is the canvas edge: anchored there, the label ran off it and the
            # figure showed "success:" over a truncated second line. Far enough
            # left to clear the terminal's own SUCCESS caption, not so far as to
            # reach the common branch's callout, which stops around x=0.55.
            "node": _LEAF_OK_COMMON,
            "dx": -74,
            "dy": -50,
            "wrap": 26,
            # Off the counts, like the colors: a figure whose caption and
            # drawing disagree teaches the disagreement.
            "text": (f"success: {_share_of_all(_LEAF_OK_COMMON):.0%} of all rollouts end here"),
        },
    ]


def render_graph_intro() -> None:
    """Under its own subheader: the example graph, drawn by the component that
    draws the real one.

    The callouts name each channel where it happens, and a list under the
    figure says the same things in words.

    That list was argued against here on the grounds that prose beside a figure
    reads as instructions to get past rather than as a key to come back to. It
    is here anyway, for a reason from outside this page: the rollout grid has a
    key of five lines, and a study that asks which of two displays a participant
    prefers cannot hand one of them the explaining and the other a diagram. The
    figure is still the primary; the list repeats it for a reader who would
    rather have it in sentences.

    Still no separate colour scale (the example is already green and red) and no
    worked example path (the graph on the page below is the example).
    """
    st.subheader("How to Read Behavior Graphs")
    graph, pos = _example_graph()
    # Centred: the figure is half the content width, and hard against the left
    # edge it read as a stray panel rather than as the page's illustration.
    _, figure, _ = st.columns([1, 6, 1])
    with figure:
        render_graph_component(
            graph,
            # Four leaves, not two, and each one writes SUCCESS / FAILURE and a
            # percentage under itself: at 330 the row pitch was 51px against a
            # 25px label plus a marker that is 34px across at 67% of the
            # rollouts, so the ✕ below sat in the ✓'s caption.
            height=430,
            key="graph_intro_example",
            pos=pos,
            symbol_override={
                _LEAF_OK_COMMON: "star",
                _LEAF_OK_RARE: "star",
                _LEAF_FAIL_COMMON: "x",
                _LEAF_FAIL_RARE: "x",
            },
            # "Behavior 2", not "B2": the shortening is for a real graph with no
            # room for the words, and this figure's whole subject is the nodes.
            shorten_labels=False,
            color_override=_example_colors(),
            prob_floor=0.0,
            annotations=_example_annotations(),
            static_figure=True,
            # Room above and below for the callouts -- a two-line one needs the
            # node's radius plus its own height -- and on the right for the
            # SUCCESS / FAILURE labels the component writes itself.
            padding={"left": 26, "right": 76, "top": 86, "bottom": 74},
        )
    st.markdown(
        "- Each circle is one behavior, labelled with what the robot was doing.\n"
        "- A circle is bigger the more rollouts reached it, and greener the more "
        "of those went on to succeed.\n"
        "- The band between two circles is wider the more rollouts took that step.\n"
        "- The **✓** and **✗** at the ends are where rollouts finished, with the "
        "share of all rollouts that ended there.\n"
        "- Click any node to see the rollouts that passed through it."
    )
