"""Expanding a "+N rare variants" summary node reveals what it stands for.

A bucket holds its parent's rare children *and their subtrees*. Everything in
there is rarer than the child that was folded, so revealing one layer just hands
the user another summary node — as many times as the rare subtree is deep. These
tests drive the click the way the graph component does (by writing the click into
session state) and read the visible node set back out of it.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from behavior_graphs.viz.simplification import RARE_COLLAPSE_CLUSTER_ID

#: Twelve episodes down one common branch, and three rare ones that split again
#: a level deeper — so the bucket stands for a subtree, not just leaves.
SEQS = [[0, 1, 2]] * 12 + [[0, 3, 4]] + [[0, 3, 5]] + [[0, 6, 7]]
OUTCOMES = [True] * 12 + [False, True, False]
#: 1/15 ≈ 6.7% per rare episode, so 20% folds every rare branch (the (0, 3)
#: subtree is 2/15 ≈ 13%) and keeps the 12-episode one (80%).
MIN_VISIT_FRACTION = 0.2

PREFIX = "tf"
#: Every episode starts in behavior 0, so the branch point is (0,), not the root.
BRANCH = (0,)


def _tree_script() -> None:
    from pathlib import Path

    import numpy as np
    import streamlit as st

    from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (
        render_trajectory_tree,
    )

    seqs = st.session_state["seqs"]
    outcomes = st.session_state["outcomes"]
    labels = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
    metadata = [
        {
            "episode_idx": ep,
            "rollout_idx": ep,
            "window_start": t * 10,
            "window_end": t * 10 + 10,
            "success": outcomes[ep],
        }
        for ep, seq in enumerate(seqs)
        for t in range(len(seq))
    ]

    # In a fragment, as every real caller does it: the tree's own Collapse
    # buttons issue st.rerun(scope="fragment"), which raises outside one.
    @st.fragment
    def _view() -> None:
        render_trajectory_tree(
            labels=labels,
            metadata=metadata,
            min_branch=1,
            mp4_dir=Path("/nonexistent"),
            mp4_index={"fps": 10, "episodes": [], "by_index": {}},
            level="rollout",
            key_prefix="tf",
            show_stats=False,
            min_visit_fraction=st.session_state["min_visit_fraction"],
        )

    _view()


def _run(*, seqs: list[list[int]] = SEQS, outcomes: list[bool] = OUTCOMES) -> AppTest:
    at = AppTest.from_function(_tree_script, default_timeout=60)
    at.session_state["seqs"] = seqs
    at.session_state["outcomes"] = outcomes
    at.session_state["min_visit_fraction"] = MIN_VISIT_FRACTION
    at.run()
    assert not at.exception
    return at


def _visible_paths(at: AppTest) -> set[tuple]:
    """The paths the tree actually drew, from the mapping it publishes."""
    return set(at.session_state[f"{PREFIX}_path_to_id"])


def _node_id(at: AppTest, path: tuple) -> int:
    return at.session_state[f"{PREFIX}_path_to_id"][path]


def _click_node(at: AppTest, path: tuple) -> AppTest:
    """Click a node the way the SVG component reports it, then rerun."""
    at.session_state[f"{PREFIX}_graph_selected"] = _node_id(at, path)
    at.run()
    assert not at.exception
    return at


def _click_fold_badge(at: AppTest, path: tuple) -> AppTest:
    """Click a node's +/- badge, which the component reports separately."""
    at.session_state[f"{PREFIX}_graph_fold_toggle"] = (_node_id(at, path), 1)
    at.run()
    assert not at.exception
    return at


def test_rare_branches_start_folded_into_one_summary_node():
    at = _run()

    paths = _visible_paths(at)

    assert BRANCH + (RARE_COLLAPSE_CLUSTER_ID,) in paths
    # Nothing of the rare subtree is drawn while its bucket is closed.
    assert not [p for p in paths if len(p) > 1 and p[1] in (3, 6)]


def test_clicking_a_summary_node_reveals_its_whole_subtree():
    at = _run()

    at = _click_node(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    paths = _visible_paths(at)
    # Both rare children, and every level below them — including the two-way
    # split under (0, 3), which is two clicks deep in the folded subtree.
    for revealed in [(0, 3), (0, 6), (0, 3, 4), (0, 3, 5), (0, 6, 7)]:
        assert revealed in paths, f"{revealed} should be revealed; got {sorted(paths)}"
    # And nothing left folded inside what was opened.
    assert not [p for p in paths if RARE_COLLAPSE_CLUSTER_ID in p]


def test_the_reveal_does_not_list_terminals_as_expanded_branch_points():
    """A ✓/✗ leaf in `expanded` reads as "this node's subtree is showing".

    That is what draws the node's fold badge, so a revealed terminal would get a
    "−" for children it hasn't got — and, being tested before the sub-cluster
    branch, would swallow the first click on the badge it should have had.
    """
    at = _run()

    at = _click_node(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    terminals = [p for p in at.session_state[f"{PREFIX}_expanded_buckets"] if p[-1] < 0]
    assert not terminals, f"terminals should stay out of `expanded`: {terminals}"
    # Still drawn, though — they are revealed, just not as branch points.
    assert [p for p in _visible_paths(at) if len(p) > 2 and p[1] == 3 and p[-1] < 0]


def test_the_fold_badge_reveals_as_much_as_the_node_click():
    """Same gesture, same reveal — the summary node has both affordances."""
    at = _run()

    at = _click_fold_badge(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    assert {(0, 3), (0, 3, 4), (0, 3, 5), (0, 6, 7)} <= _visible_paths(at)


def test_expanding_a_bucket_leaves_other_buckets_alone():
    """The bucket's contents, not everything rare on the canvas.

    The common branch has its own rare child; opening the rare bucket beside it
    must not open that too, or one click would flatten the whole tree.
    """
    at = _run(seqs=[*SEQS, [0, 1, 8]], outcomes=[*OUTCOMES, False])
    assert (0, 1, RARE_COLLAPSE_CLUSTER_ID) in _visible_paths(at), "the other bucket"

    at = _click_node(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    paths = _visible_paths(at)
    assert (0, 3) in paths, "the clicked bucket opened"
    assert (0, 1, RARE_COLLAPSE_CLUSTER_ID) in paths, "the other bucket stayed shut"
    assert (0, 1, 8) not in paths


def test_collapsing_the_branch_point_folds_the_whole_reveal_back():
    at = _run()
    at = _click_node(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    # "-" on the branch point whose bucket was opened.
    at = _click_fold_badge(at, BRANCH)

    paths = _visible_paths(at)
    assert BRANCH + (RARE_COLLAPSE_CLUSTER_ID,) in paths, "back to one summary node"
    assert not [p for p in paths if len(p) > 1 and p[1] in (3, 6)]


def test_collapsing_a_root_level_bucket_leaves_other_expansions_alone():
    """The root is a prefix of every path, so it needs care twice over.

    A tree whose *initial* behaviors include a rare one has a bucket at the root
    itself. Listing expansions by prefix would hide every other expansion behind
    that one, and collapsing it by prefix would drop them all.
    """
    # Rare initial behavior 9 (1 of 17) as well as the rare branches under 0.
    seqs = [*SEQS, [0, 1, 8], [9, 1, 2]]
    outcomes = [*OUTCOMES, False, True]
    at = _run(seqs=seqs, outcomes=outcomes)
    assert (RARE_COLLAPSE_CLUSTER_ID,) in _visible_paths(at), "a bucket at the root"

    # Open the root's bucket, then a second, unrelated one under the common branch.
    at = _click_node(at, (RARE_COLLAPSE_CLUSTER_ID,))
    at = _click_node(at, (0, 1, RARE_COLLAPSE_CLUSTER_ID))
    assert (9,) in _visible_paths(at), "the root's rare initial behavior"
    assert (0, 1, 8) in _visible_paths(at), "the other bucket's contents"

    # Both are offered for collapsing, rather than the deeper one hiding behind
    # the root — the buttons the operator actually gets.
    offered = {
        b.key.removeprefix(f"{PREFIX}_collapse_") for b in at.button if "_collapse_" in b.key
    }
    assert "()" in offered and "(0, 1)" in offered, offered

    # And collapsing the root's bucket keeps the unrelated expansion.
    next(b for b in at.button if b.key == f"{PREFIX}_collapse_()").click().run()
    assert not at.exception

    paths = _visible_paths(at)
    assert (9,) not in paths, "the root's bucket closed again"
    assert (0, 1, 8) in paths, "the other expansion should have survived"


def test_the_expanded_list_names_the_click_not_every_node_it_revealed():
    at = _run()

    at = _click_node(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    # One entry per expansion: the reveal puts five paths in `expanded`, and
    # collapsing any but the top of them is meaningless, so only the top is
    # offered.
    labels = [str(e.label) for e in at.expander]
    assert any("Expanded branches (1)" in label for label in labels), labels


def test_a_summary_node_click_does_not_also_select_a_node():
    """The clicked node no longer exists after the expansion.

    Left in place, the click would point the video panel at a node id that the
    next render numbers as something else entirely.
    """
    at = _run()

    at = _click_node(at, BRANCH + (RARE_COLLAPSE_CLUSTER_ID,))

    assert f"{PREFIX}_graph_selected" not in at.session_state
