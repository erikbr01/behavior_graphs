"""Headless tests for the ported visualization stack.

Builds a synthetic run dir through RunArtifacts, reloads it, and checks that
the main ported figure functions return plotly Figures.
"""

from __future__ import annotations

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from behavior_graphs.core.graph import BehaviorGraph  # noqa: E402
from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths  # noqa: E402
from behavior_graphs.pipeline.artifacts import RunArtifacts  # noqa: E402
from behavior_graphs.stages.graph.smoothing import in_clustering_space  # noqa: E402
from behavior_graphs.viz import simplification as gs  # noqa: E402


@pytest.fixture()
def run_dir(tmp_path):
    """Synthetic 4-episode run dir written through RunArtifacts."""
    rng = np.random.default_rng(0)
    lengths = [12, 12, 10, 10]
    successes = [True, True, False, None]
    episodes = episodes_from_lengths(lengths, successes)

    windows = []
    labels = []
    for ep_idx, n in enumerate(lengths):
        for start in range(0, n - 4, 2):
            windows.append(WindowInfo(episode_idx=ep_idx, start=start, end=start + 4))
            # Phase-like labels: early windows → 0, mid → 1, late → 2 (or 3 on failure).
            frac = start / n
            if frac < 0.3:
                labels.append(0)
            elif frac < 0.6:
                labels.append(1)
            else:
                labels.append(2 if successes[ep_idx] is not False else 3)
    labels = np.asarray(labels, dtype=np.int64)
    features = rng.normal(size=(len(windows), 5)).astype(np.float32)
    window_set = WindowSet(features=features, windows=windows, episodes=episodes)

    graph = BehaviorGraph.from_window_labels(labels, window_set)

    artifacts = RunArtifacts(tmp_path / "run")
    artifacts.save_windows(window_set)
    artifacts.save_array("labels", labels)
    artifacts.save_array("coords_2d", rng.normal(size=(len(windows), 2)))
    artifacts.save_graph(graph)
    artifacts.save_node_values(graph.compute_values())
    artifacts.save_metrics({"rollout_dir": str(tmp_path / "rollouts")})
    return artifacts.run_dir


def _load(run_dir):
    artifacts = RunArtifacts(run_dir)
    windows = artifacts.load_windows()
    labels = artifacts.load_array("labels")
    graph = artifacts.load_graph()
    coords = artifacts.load_array("coords_2d")
    return artifacts, windows, labels, graph, coords


def test_behavior_graph_figure(run_dir):
    from behavior_graphs.viz.plotly.behavior_graph import create_behavior_graph_plot

    _, _, _, graph, _ = _load(run_dir)
    fig = create_behavior_graph_plot(graph)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_cluster_scatter_figure(run_dir):
    from behavior_graphs.viz.compat import meta_from_windows
    from behavior_graphs.viz.plotly.clusters import create_cluster_scatter_2d

    _, windows, labels, _, coords = _load(run_dir)
    meta = meta_from_windows(windows)
    fig = create_cluster_scatter_2d(coords, labels, meta)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    fig2 = create_cluster_scatter_2d(coords, labels, meta, color_by="success")
    assert isinstance(fig2, go.Figure)


def test_paths_and_value_figures(run_dir):
    from behavior_graphs.viz.plotly.behavior_graph import (
        create_behavior_paths_plot,
        create_node_value_bar_chart,
        create_transition_matrix_heatmap,
    )

    artifacts, _, _, graph, _ = _load(run_dir)
    paths = graph.enumerate_paths(max_paths=10)
    assert paths, "synthetic graph should have at least one path"
    fig = create_behavior_paths_plot(graph, paths)
    assert isinstance(fig, go.Figure)
    values = artifacts.load_node_values()
    assert values
    fig2 = create_node_value_bar_chart(graph, values)
    assert isinstance(fig2, go.Figure)
    fig3 = create_transition_matrix_heatmap(graph)
    assert isinstance(fig3, go.Figure)


def test_trajectory_tree_figures(run_dir):
    from behavior_graphs.viz import simplification as gs
    from behavior_graphs.viz.compat import meta_from_windows
    from behavior_graphs.viz.plotly.trajectory_tree import (
        create_trajectory_icicle,
        create_trajectory_sunburst,
    )

    _, windows, labels, _, _ = _load(run_dir)
    meta = meta_from_windows(windows)
    nodes = gs.build_trajectory_tree(labels, meta)
    assert nodes and nodes[0]["n_episodes"] == len(windows.episodes)
    for fn in (create_trajectory_sunburst, create_trajectory_icicle):
        fig = fn(nodes)
        assert isinstance(fig, go.Figure)


def _with_visit_fractions(nodes):
    """Test helper mirroring build_trajectory_tree's visit_fraction field:
    each node's share of the root's total n_episodes (1.0 for the root)."""
    root_n_episodes = next(nd["n_episodes"] for nd in nodes if nd["parent_path"] is None)
    out = []
    for nd in nodes:
        nd = dict(nd)
        nd["visit_fraction"] = (
            1.0
            if nd["parent_path"] is None
            else nd["n_episodes"] / root_n_episodes
            if root_n_episodes
            else 0.0
        )
        out.append(nd)
    return out


def test_collapse_rare_children_bounds_branching_without_hiding():
    from behavior_graphs.viz.simplification import RARE_COLLAPSE_CLUSTER_ID, collapse_rare_children

    def node(path, n_episodes, n_success=0, n_failure=0):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_success,
            "n_failure": n_failure,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    # Root with 5 children of decreasing count, one of which (child 4, the
    # rarest) has its own child — its whole subtree must fold together.
    nodes = _with_visit_fractions(
        [
            node((), 100),
            node((0,), 50),
            node((1,), 30),
            node((2,), 10),
            node((3,), 6),
            node((4,), 4, n_failure=4),
            node((4, 9), 4, n_failure=4),
        ]
    )

    out = collapse_rare_children(nodes, min_visit_fraction=0.08)
    by_path = {tuple(nd["path"]): nd for nd in out}

    # Children at/above the threshold keep their own branch, regardless of
    # how many of them there are (unlike a fixed top-K rank cutoff).
    assert (0,) in by_path
    assert (1,) in by_path
    assert (2,) in by_path
    # Below-threshold children (3, 4) fold, including 4's subtree.
    assert (3,) not in by_path
    assert (4,) not in by_path
    assert (4, 9) not in by_path

    bucket = by_path[(RARE_COLLAPSE_CLUSTER_ID,)]
    assert bucket["n_episodes"] == 6 + 4  # children 3, 4 — not double-counting 4's child
    assert bucket["n_failure"] == 4
    assert len(bucket["episode_indices"]) == 10  # 6 + 4 raw episode-slots preserved
    # 2 branches folded (children 3, 4), even though 4's own subtree adds a
    # 3rd hidden node (4, 9) — n_folded_branches counts branches, not nodes.
    assert bucket["n_folded_branches"] == 2

    # Nothing below the threshold: untouched.
    assert collapse_rare_children(nodes, min_visit_fraction=0.02) == nodes
    # Disabled: untouched.
    assert collapse_rare_children(nodes, min_visit_fraction=None) == nodes
    assert collapse_rare_children(nodes, min_visit_fraction=0.0) == nodes


def test_collapse_rare_children_never_folds_a_terminal_child():
    """Rare terminal leaves stay on the canvas however many there are.

    A terminal has no subtree to hide, so bucketing one buys no
    simplification while hiding the outcome — expanding a "+2 rare variants"
    node to find nothing but ✗ FAILURE behind it is exactly the click this
    avoids. Rare children with their own subtrees still fold."""
    from behavior_graphs.core.graph import FAILURE_NODE_ID, SUCCESS_NODE_ID
    from behavior_graphs.viz.simplification import RARE_COLLAPSE_CLUSTER_ID, collapse_rare_children

    def node(path, n_episodes, n_success=0, n_failure=0):
        cid = path[-1] if path else -2
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": cid,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_success,
            "n_failure": n_failure,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    # (0,)'s only rare child is a lone FAILURE leaf: nothing to gain by
    # folding a single node into a single bucket node.
    # 0.15 threshold: every rare child below is 2/20 = 10% of the root
    # (comfortably below threshold, i.e. a fold candidate), while every
    # kept sibling is 16-18/20 = 80-90% (comfortably above) — the exact
    # value isn't load-bearing, only that it separates the two groups.
    solo_terminal = _with_visit_fractions(
        [
            node((), 20),
            node((0,), 18),
            node((0, FAILURE_NODE_ID), 2, n_failure=2),
        ]
    )
    out = collapse_rare_children(solo_terminal, min_visit_fraction=0.15)
    by_path = {tuple(nd["path"]): nd for nd in out}
    assert (0, FAILURE_NODE_ID) in by_path
    assert out == solo_terminal

    # Two rare terminal children at the same branch point: still unfolded.
    # Bucketing them would hide both outcomes behind a click that reveals two
    # bare leaves — no structure saved, an outcome lost.
    two_terminals = _with_visit_fractions(
        [
            node((), 20),
            node((0,), 16),
            node((0, SUCCESS_NODE_ID), 2, n_success=2),
            node((0, FAILURE_NODE_ID), 2, n_failure=2),
        ]
    )
    out2 = collapse_rare_children(two_terminals, min_visit_fraction=0.15)
    by_path2 = {tuple(nd["path"]): nd for nd in out2}
    assert (0, SUCCESS_NODE_ID) in by_path2
    assert (0, FAILURE_NODE_ID) in by_path2
    assert (0, RARE_COLLAPSE_CLUSTER_ID) not in by_path2
    assert out2 == two_terminals

    # A rare behavior child sharing a branch point with a rare terminal: the
    # behavior's subtree folds, the terminal stays put.
    mixed = _with_visit_fractions(
        [
            node((), 20),
            node((0,), 14),
            node((0, FAILURE_NODE_ID), 2, n_failure=2),
            node((0, 7), 2, n_success=2),
            node((0, 7, SUCCESS_NODE_ID), 2, n_success=2),
        ]
    )
    out3 = collapse_rare_children(mixed, min_visit_fraction=0.15)
    by_path3 = {tuple(nd["path"]): nd for nd in out3}
    assert (0, FAILURE_NODE_ID) in by_path3
    assert (0, 7) not in by_path3
    assert (0, 7, SUCCESS_NODE_ID) not in by_path3
    assert by_path3[(0, RARE_COLLAPSE_CLUSTER_ID)]["n_folded_branches"] == 1

    # A single rare child with its own subtree: still worth folding, since
    # it's hiding more than just itself.
    solo_with_subtree = _with_visit_fractions(
        [
            node((), 20),
            node((0,), 16),
            node((0, 9), 2),
            node((0, 9, SUCCESS_NODE_ID), 2, n_success=2),
        ]
    )
    out3 = collapse_rare_children(solo_with_subtree, min_visit_fraction=0.15)
    by_path3 = {tuple(nd["path"]): nd for nd in out3}
    assert (0, 9) not in by_path3
    assert (0, 9, SUCCESS_NODE_ID) not in by_path3
    assert (0, RARE_COLLAPSE_CLUSTER_ID) in by_path3


def test_collapse_rare_children_does_not_mislabel_uniform_branching_as_rare():
    """Regression: a wide, roughly-uniform branch point (e.g. the root
    splitting into K similarly-sized initial behaviors) must not have most
    of its data folded into a "rare" bucket just for not being in some fixed
    top-N — only children that are actually small in absolute terms fold."""
    from behavior_graphs.viz.simplification import collapse_rare_children

    def node(path, n_episodes):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_episodes,
            "n_failure": 0,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    # 20 roughly-equal initial behaviors (~50 episodes each, 1000 total) —
    # none of them are individually rare. 0.05 threshold intentionally
    # equals each child's own share (50/1000) exactly: folding uses a
    # strict "<", so this checks the boundary isn't folded either.
    nodes = _with_visit_fractions([node((), 1000)] + [node((i,), 50) for i in range(20)])

    out = collapse_rare_children(nodes, min_visit_fraction=0.05)
    by_path = {tuple(nd["path"]): nd for nd in out}
    # Every one of the 20 keeps its own branch; nothing gets folded.
    for i in range(20):
        assert (i,) in by_path
    assert out == nodes


def test_collapse_rare_children_no_orphaned_bucket_when_its_own_parent_folds():
    """Regression: a node that is itself folded into an ancestor's bucket
    may have had its own rare children waiting to fold into a bucket of its
    own — that bucket must not survive as an orphan (no node in the result
    should reference a parent_path that isn't itself in the result)."""
    from behavior_graphs.viz.simplification import (
        RARE_COLLAPSE_CLUSTER_ID,
        collapse_rare_children,
    )

    def node(path, n_episodes):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_episodes,
            "n_failure": 0,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    # (1,) itself has only 2 episodes (rare -> folds under root), but it has
    # its own rare children (1,5) and (1,6) that would independently fold
    # into a bucket under (1,) — a bucket whose parent (1,) is gone.
    nodes = _with_visit_fractions(
        [
            node((), 30),
            node((0,), 20),  # kept
            node((1,), 2),  # rare -> folds under root
            node((1, 5), 1),
            node((1, 6), 1),
        ]
    )

    out = collapse_rare_children(nodes, min_visit_fraction=0.1)
    by_path = {tuple(nd["path"]): nd for nd in out}

    # No orphans: every node's parent_path is present in the result.
    for nd in out:
        if nd["parent_path"] is not None:
            assert tuple(nd["parent_path"]) in by_path, f"orphaned node: {nd['path']}"

    assert (1,) not in by_path
    assert (1, 5) not in by_path
    assert (1, 6) not in by_path
    bucket = by_path[(RARE_COLLAPSE_CLUSTER_ID,)]
    assert bucket["n_episodes"] == 2  # (1,)'s own count already subsumes (1,5)/(1,6)


def test_collapse_rare_children_expanded_reveals_recursively():
    """Expanding a branch point reveals its immediate children (undoing the
    fold there only); any of those children with their own rare grandchildren
    get a fresh, independently-collapsed nested summary node — recursion
    without special-casing, since each newly-visible node is just another
    branch point subject to the same rule."""
    from behavior_graphs.viz.simplification import RARE_COLLAPSE_CLUSTER_ID, collapse_rare_children

    def node(path, n_episodes):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_episodes,
            "n_failure": 0,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    nodes = _with_visit_fractions(
        [
            node((), 10),
            node((0,), 8),  # kept
            node((1,), 2),  # rare at root -> folds
            node((1, 5), 1),  # (1,)'s own rare child
            node((1, 6), 1),  # (1,)'s own rare child
        ]
    )

    # visit_fraction is global (share of the root's total), so (1,5)/(1,6)
    # are 1/10 = 10% each even though they're 1/2 = 50% of their own parent
    # (1,)'s 2 episodes. 0.6 threshold: folds (1,)=2/10=20% at the root and,
    # once revealed, (1,5)/(1,6)=10% each under it — while keeping
    # (0,)=8/10=80% at every level. Any threshold in (20%, 80%] would do;
    # 0.6 was picked with margin on both sides, not derived from a formula.
    collapsed = {
        tuple(nd["path"]): nd for nd in collapse_rare_children(nodes, min_visit_fraction=0.6)
    }
    assert (1,) not in collapsed
    assert (RARE_COLLAPSE_CLUSTER_ID,) in collapsed

    # Expand the root's bucket: (1,) becomes a real node again, but its own
    # rare children (5, 6) — never separately expanded — form a fresh nested
    # bucket under (1,), not the root.
    expanded_root = {
        tuple(nd["path"]): nd
        for nd in collapse_rare_children(nodes, min_visit_fraction=0.6, expanded=frozenset({()}))
    }
    assert (1,) in expanded_root
    assert (1, 5) not in expanded_root
    assert (1, 6) not in expanded_root
    assert (1, RARE_COLLAPSE_CLUSTER_ID) in expanded_root
    assert expanded_root[(1, RARE_COLLAPSE_CLUSTER_ID)]["n_episodes"] == 2

    # Expanding both levels reveals everything.
    fully_expanded = {
        tuple(nd["path"]): nd
        for nd in collapse_rare_children(
            nodes, min_visit_fraction=0.6, expanded=frozenset({(), (1,)})
        )
    }
    assert (1, 5) in fully_expanded
    assert (1, 6) in fully_expanded


def test_bucket_reveal_paths_opens_a_bucket_all_the_way_down():
    """What a caller passes as ``expanded`` to open one summary node.

    Everything inside a bucket is rarer than the child that was folded, so
    revealing a layer at a time only produces another summary node one level
    down. The reveal set covers the rare children's whole subtrees — and
    nothing else on the canvas.
    """
    from behavior_graphs.viz.simplification import (
        RARE_COLLAPSE_CLUSTER_ID,
        bucket_reveal_paths,
        collapse_rare_children,
    )

    def node(path, n_episodes):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_episodes,
            "n_failure": 0,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    from behavior_graphs.core.graph import SUCCESS_NODE_ID, TERMINAL_NODE_IDS

    nodes = _with_visit_fractions(
        [
            node((), 10),
            node((0,), 8),  # frequent, kept
            node((0, 4), 1),  # ...with a rare child of its own
            node((1,), 2),  # rare at the root -> folds
            node((1, 5), 1),  # inside the bucket
            node((1, 5, 6), 1),  # deeper inside it
            node((1, 5, 6, SUCCESS_NODE_ID), 1),  # how that branch ends
        ]
    )

    reveal = bucket_reveal_paths(nodes, 0.6, ())

    # The branch point itself, plus the folded child's entire subtree.
    assert reveal == {(), (1,), (1, 5), (1, 5, 6)}
    # Terminals stay out: only branch points mean anything in `expanded`, and a
    # ✓/✗ leaf listed there gets a collapse badge for children it hasn't got.
    assert not [p for p in reveal if p and p[-1] in TERMINAL_NODE_IDS]
    # Not the frequent sibling's own rare child: that is a different bucket, and
    # one click should not flatten the tree.
    assert (0, 4) not in reveal

    opened = {tuple(nd["path"]) for nd in collapse_rare_children(nodes, 0.6, frozenset(reveal))}
    assert {(1,), (1, 5), (1, 5, 6)} <= opened
    assert RARE_COLLAPSE_CLUSTER_ID not in {p[-1] for p in opened if p and p[0] == 1}
    # The frequent branch keeps its own summary node.
    assert (0, RARE_COLLAPSE_CLUSTER_ID) in opened


def test_bucket_reveal_paths_is_a_no_op_without_a_threshold():
    """No threshold, no folding, nothing to reveal but the node itself."""
    from behavior_graphs.viz.simplification import bucket_reveal_paths

    nodes = _with_visit_fractions(
        [
            {
                "path": (),
                "label": "root",
                "cluster_id": -2,
                "depth": 0,
                "n_episodes": 4,
                "n_success": 4,
                "n_failure": 0,
                "parent_path": None,
                "episode_indices": [0, 1, 2, 3],
            }
        ]
    )

    assert bucket_reveal_paths(nodes, None, ()) == {()}
    assert bucket_reveal_paths(nodes, 0.0, ()) == {()}


def test_collapse_manual_hides_only_descendants_and_keeps_the_node():
    """Manually collapsing a branch point hides its subtree but keeps the
    node itself (unlike rare-folding, which replaces siblings with a new
    synthetic bucket) so it can carry its own "+" badge to re-expand."""
    from behavior_graphs.viz.simplification import collapse_manual

    def node(path, n_episodes):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_episodes,
            "n_failure": 0,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    nodes = [
        node((), 10),
        node((0,), 6),
        node((0, 1), 4),
        node((0, 1, 2), 4),
        node((3,), 4),
    ]

    out = collapse_manual(nodes, frozenset({(0, 1)}))
    by_path = {tuple(nd["path"]): nd for nd in out}
    # The collapsed node itself survives...
    assert (0, 1) in by_path
    # ...but its descendants are gone.
    assert (0, 1, 2) not in by_path
    # Everything outside the collapsed subtree is untouched.
    assert (0,) in by_path
    assert (3,) in by_path

    # Nothing collapsed: untouched.
    assert collapse_manual(nodes, frozenset()) == nodes


def test_collapse_manual_nested_collapse_survives_ancestor_reexpand():
    """A node collapsed underneath an already-collapsed ancestor stays
    recorded; once the ancestor is un-collapsed, the nested collapse still
    applies on its own (no special-casing needed for the ordering)."""
    from behavior_graphs.viz.simplification import collapse_manual

    def node(path, n_episodes):
        return {
            "path": path,
            "label": f"n{path}",
            "cluster_id": path[-1] if path else -2,
            "depth": len(path),
            "n_episodes": n_episodes,
            "n_success": n_episodes,
            "n_failure": 0,
            "parent_path": path[:-1] if path else None,
            "episode_indices": list(range(n_episodes)),
        }

    nodes = [
        node((), 10),
        node((0,), 8),
        node((0, 1), 6),
        node((0, 1, 2), 4),
    ]

    # Both (0,) and (0, 1) marked collapsed: (0, 1) is already hidden by its
    # ancestor, so its own collapse entry is a no-op — but it's still valid
    # to pass, and re-expanding (0,) alone must still hide (0, 1, 2).
    both_collapsed = {
        tuple(nd["path"]): nd for nd in collapse_manual(nodes, frozenset({(0,), (0, 1)}))
    }
    assert (0,) in both_collapsed
    assert (0, 1) not in both_collapsed
    assert (0, 1, 2) not in both_collapsed

    only_child_collapsed = {
        tuple(nd["path"]): nd for nd in collapse_manual(nodes, frozenset({(0, 1)}))
    }
    assert (0,) in only_child_collapsed
    assert (0, 1) in only_child_collapsed
    assert (0, 1, 2) not in only_child_collapsed


def test_compat_graph_roundtrip(run_dir):
    """graph_from_labels_meta over the legacy metadata matches the saved graph."""
    from behavior_graphs.viz.compat import graph_from_labels_meta, meta_from_windows

    _, windows, labels, graph, _ = _load(run_dir)
    meta = meta_from_windows(windows)
    rebuilt = graph_from_labels_meta(labels, meta)
    assert set(rebuilt.nodes) == set(graph.nodes)
    assert rebuilt.transition_counts == graph.transition_counts
    assert rebuilt.num_episodes == graph.num_episodes


def test_simplification_layouts(run_dir):
    from behavior_graphs.viz import simplification as gs
    from behavior_graphs.viz.compat import meta_from_windows

    _, windows, labels, graph, _ = _load(run_dir)
    meta = meta_from_windows(windows)
    pos = gs.temporal_layout(graph, labels, meta)
    assert set(pos) >= set(graph.nodes)
    pruned = gs.prune_edges_by_count(graph, min_count=1)
    assert isinstance(pruned, BehaviorGraph)


class TestSmoothingEffect:
    """Pricing what flicker smoothing would do to one run's tree.

    The point of measuring is that the answer is not general: over sweep_all's
    176 best-per-K runs, sticky smoothing at λ=25 shrinks the trajectory tree on
    60% of runs and *grows* it on 26%. These pin the two directions on data built
    to oscillate and data built not to.
    """

    @staticmethod
    def _run(seqs, successes, *, jitter: float):
        """Episodes whose features sit on their label's centroid, plus jitter.

        ``jitter`` is how far a window strays from its own cluster's centre, so a
        large value is what makes sticky decode prefer staying put over following
        the noisy per-window label.
        """
        centroids = {0: [0.0, 0.0], 1: [10.0, 0.0], 2: [0.0, 10.0], 3: [10.0, 10.0]}
        rng = np.random.default_rng(0)
        labels, windows, meta = [], [], []
        for ep, seq in enumerate(seqs):
            for t, c in enumerate(seq):
                labels.append(c)
                windows.append(WindowInfo(episode_idx=ep, start=t, end=t + 1))
                meta.append(
                    {
                        "rollout_idx": ep,
                        "window_start": t,
                        "window_end": t + 1,
                        "success": successes[ep],
                    }
                )
        feats = np.asarray([centroids[c] for c in labels], dtype=np.float32)
        feats += rng.normal(scale=jitter, size=feats.shape).astype(np.float32)
        return (
            np.asarray(labels, dtype=np.int64),
            WindowSet(
                features=feats,
                windows=windows,
                episodes=episodes_from_lengths([len(s) for s in seqs], successes),
            ),
            meta,
        )

    def test_oscillating_labels_shrink_the_tree(self):
        """A→B→A→B flicker: run-length collapse cannot remove it, smoothing can."""
        seqs = [[0, 1, 0, 1, 0, 1, 2]] * 8
        labels, windows, meta = self._run(seqs, [True] * 8, jitter=6.0)

        raw, smoothed = gs.smoothing_effect(labels, windows, meta, sticky_lambda=5.0)

        assert smoothed[0] < raw[0], (raw, smoothed)
        assert smoothed[1] < raw[1], "and the flicker was costing depth"

    def test_too_little_smoothing_makes_the_tree_bigger(self):
        """The failure mode that makes a default dangerous.

        Below the usable band the decoder still re-assigns every window, but from
        each episode's *own* features — so paths come out shorter and less alike,
        and a prefix tree charges for branching. Measured on real runs too: at
        λ=10, 69 of sweep_all's 176 best-per-K runs gained nodes.
        """
        seqs = [[0, 1, 0, 1, 0, 1, 2]] * 8
        labels, windows, meta = self._run(seqs, [True] * 8, jitter=6.0)

        raw, smoothed = gs.smoothing_effect(labels, windows, meta, sticky_lambda=0.1)

        assert smoothed[0] > raw[0], (raw, smoothed)

    def test_too_much_smoothing_collapses_real_behaviors(self):
        """The other end: clean, well-separated behaviors merged into one run.

        Together with the test above this is why the page prices smoothing per run
        rather than shipping a λ. λ is a flat penalty against *per-episode
        normalized* unary costs (see ``sticky_decode``), whose contrast depends on
        K and on how far apart the centroids are — so the band that helps moves
        from run to run, and the repo's λ=25 is deep into saturation at this
        feature scale. Over sweep_all at λ=25 the median run drops from 3.4
        behaviors per episode to 1.9.
        """
        seqs = [[0, 1, 2]] * 8  # three distinct behaviors, no flicker to remove
        labels, windows, meta = self._run(seqs, [True] * 8, jitter=0.01)

        raw, smoothed = gs.smoothing_effect(labels, windows, meta, sticky_lambda=25.0)

        assert raw[1] == 4, "START -> 0 -> 1 -> 2 -> SUCCESS"
        assert smoothed[1] < raw[1], (raw, smoothed)

    def test_tree_size_reports_nodes_and_depth(self):
        seqs = [[0, 1, 2]] * 4
        labels, _windows, meta = self._run(seqs, [True] * 4, jitter=0.01)
        nodes = gs.build_trajectory_tree(labels, meta, level="rollout")

        count, depth = gs.tree_size(nodes)

        assert count == len(nodes)
        assert depth == max(nd["depth"] for nd in nodes)


class TestClusteringSpace:
    """Smoothing has to measure distance where the clusters are actually compact.

    The pipeline fits the clustering on the *reduced* array but a ``WindowSet``
    carries raw features, and UMAP does not preserve distances — so scoring
    "distance to centroid" on raw features scores it in a space the clusters were
    never defined in. Over sweep_all's 176 best-per-K runs that one substitution
    is the difference between the smoother shrinking the median tree by 11% and
    growing it by 56%.
    """

    @staticmethod
    def _windows(n: int, raw_dim: int = 8):
        import numpy as np

        from behavior_graphs.core.types import WindowInfo, WindowSet, episodes_from_lengths

        rng = np.random.default_rng(0)
        return WindowSet(
            features=rng.normal(size=(n, raw_dim)).astype(np.float32),
            windows=[WindowInfo(episode_idx=0, start=t, end=t + 1) for t in range(n)],
            episodes=episodes_from_lengths([n], [True]),
        )

    def test_it_swaps_in_the_reduced_features(self):
        import numpy as np

        windows = self._windows(6)
        reduced = np.arange(12, dtype=np.float32).reshape(6, 2)

        out = in_clustering_space(windows, reduced)

        assert out.features.shape == (6, 2)
        assert np.array_equal(out.features, reduced)
        # Structure is untouched: graph building reads only these.
        assert out.windows == windows.windows
        assert out.episodes == windows.episodes
        assert windows.features.shape == (6, 8), "the input is not mutated"

    def test_no_reduction_stage_means_raw_is_the_space(self):
        windows = self._windows(6)

        assert in_clustering_space(windows, None) is windows

    def test_a_mismatched_reduced_array_is_refused(self):
        """Rather than silently smoothing against the wrong rows."""
        import numpy as np

        windows = self._windows(6)

        with pytest.raises(ValueError, match="5 rows but there are 6 windows"):
            in_clustering_space(windows, np.zeros((5, 2), dtype=np.float32))
