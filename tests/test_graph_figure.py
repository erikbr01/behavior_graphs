"""The behavior graph as a paper figure.

What is worth pinning here is not how it looks — that is a judgement, and it is
made by looking at it — but the promises the figure makes to the paper it goes
into. It is exported at the column width, so its 8 pt type is 8 pt on the page.
It embeds TrueType outlines, because a Type 3 figure is what a submission gets
bounced over. It carries no timestamp, so an unchanged graph re-exports
unchanged. And it draws *the graph on screen*: the same nodes after pruning, the
same glyph per role, and the same layout it was handed.
"""

from __future__ import annotations

import itertools

import pytest

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
    BehaviorNode,
)
from behavior_graphs.viz import graph_figure, paper


def _graph(
    *,
    behaviors: int = 3,
    episodes: int = 20,
    rare_edge: bool = False,
) -> BehaviorGraph:
    """START → a chain of behaviors → SUCCESS and FAILURE.

    Plus, on request, a one-episode edge straight from the first behavior to
    FAILURE: the shape the state-merging mode preserves and dashes.
    """
    nodes = {
        START_NODE_ID: BehaviorNode(START_NODE_ID, "START", 0, episodes),
        SUCCESS_NODE_ID: BehaviorNode(SUCCESS_NODE_ID, "SUCCESS", 0, episodes // 2),
        FAILURE_NODE_ID: BehaviorNode(FAILURE_NODE_ID, "FAILURE", 0, episodes // 2),
    }
    for i in range(behaviors):
        nodes[i] = BehaviorNode(i, f"Behavior {i}", 100 * (i + 1), episodes - i)
    counts: dict[int, dict[int, int]] = {START_NODE_ID: {0: episodes}}
    for i in range(behaviors - 1):
        counts[i] = {i + 1: episodes - i - 1}
    counts[behaviors - 1] = {SUCCESS_NODE_ID: episodes // 2, FAILURE_NODE_ID: episodes // 2}
    if rare_edge:
        counts.setdefault(0, {})[FAILURE_NODE_ID] = 1
    probs = {
        src: {tgt: c / (sum(tgts.values()) or 1) for tgt, c in tgts.items()}
        for src, tgts in counts.items()
    }
    return BehaviorGraph(
        nodes=nodes, transition_counts=counts, transition_probs=probs, num_episodes=episodes
    )


def _layout(graph: BehaviorGraph) -> dict[int, tuple[float, float]]:
    """A level-based layout, like the ones the app hands the component."""
    from behavior_graphs.viz.graph_plot import layered_layout

    return layered_layout(graph)


def _one_label(fig) -> str:
    """The only label in a figure drawn with exactly one."""
    (text,) = fig.axes[0].texts
    return text.get_text()


def _label_width_pt(fig) -> float:
    (text,) = fig.axes[0].texts
    return text.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi


def _rgba(color: str) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba

    return tuple(round(v, 3) for v in to_rgba(color))


def _circles(fig) -> list:
    """The node bodies. Every node is one, and a role's glyph is a second
    collection on top of it — see ``gid``."""
    return [c for c in fig.axes[0].collections if (c.get_gid() or "").startswith("node-")]


def _glyphs(fig) -> dict[str, object]:
    """The role glyphs drawn inside the nodes, by node id."""
    return {
        (c.get_gid() or "").removeprefix("glyph-"): c
        for c in fig.axes[0].collections
        if (c.get_gid() or "").startswith("glyph-")
    }


def _label_boxes(fig) -> dict[str, tuple[float, float, float, float]]:
    """Where each label ended up, in display pixels, keyed by its text."""
    renderer = fig.canvas.get_renderer()
    boxes = {}
    for text in fig.axes[0].texts:
        box = text.get_window_extent(renderer)
        boxes[text.get_text()] = (box.x0, box.y0, box.x1, box.y1)
    return boxes


# ── What the figure promises the paper ───────────────────────────────────────


def test_it_is_exported_at_the_column_width():
    """Not scaled to it: type at 8 pt in a figure that the LaTeX then shrinks is
    not 8 pt on the page, which is the whole reason for sizing in inches."""
    graph = _graph()

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph))

    assert fig.get_figwidth() == pytest.approx(paper.DOUBLE_COLUMN_WIDTH_IN)

    single = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), width_in=paper.SINGLE_COLUMN_WIDTH_IN
    )
    assert single.get_figwidth() == pytest.approx(paper.SINGLE_COLUMN_WIDTH_IN)


def test_the_same_graph_exports_the_same_bytes():
    """No timestamp in it, so a figure that has not changed does not turn up as
    a diff in the paper repo."""
    graph = _graph()
    pos = _layout(graph)

    first = graph_figure.behavior_graph_figure_files(graph=graph, pos=pos)
    second = graph_figure.behavior_graph_figure_files(graph=graph, pos=pos)

    assert first is not None and second is not None
    assert first["pdf"] == second["pdf"]
    assert b"CreationDate" not in first["pdf"]


def test_the_labels_are_text_in_an_embedded_truetype_font():
    """Three things at once, because they fail together.

    ``/FontFile2`` is a TrueType font *program* in the file, which is what
    ``pdf.fonttype: 42`` produces and what IEEE PDF eXpress wants; ``/Type3`` is
    the fallback it rejects, and matplotlib takes it silently. And a font object
    at all is the evidence that the labels went in as text rather than as paths:
    any path effect on a ``Text`` — a white halo behind it, say — renders the
    glyphs as outlines, leaving a PDF with no fonts, nothing searchable and
    nothing for the ``fonttype`` setting to apply to.
    """
    graph = _graph()

    files = graph_figure.behavior_graph_figure_files(graph=graph, pos=_layout(graph))

    assert files is not None
    assert b"/FontFile2" in files["pdf"]
    assert b"/Type3" not in files["pdf"]
    # The face itself is whatever the machine has of the paper's list — Times on
    # a licensed one, Tinos or Liberation Serif (both metric-compatible) in the
    # container. Which one is not this test's business; that it is one of them is.
    assert paper.BASE_RC["font.serif"][0] == "Times New Roman"


def test_nothing_it_draws_falls_outside_the_figure():
    """The margins are taken out of the data limits after the fact, because a
    node's radius and its label are in points and only the finished figure knows
    how much of it they cover. A label over the edge is a clipped label."""
    graph = _graph(behaviors=4)
    pos = _layout(graph)

    fig = graph_figure.behavior_graph_figure(graph, pos=pos)

    ax = fig.axes[0]
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # The drawing is laid out in the unit square, so limits wider than it are
    # exactly the room the markers and labels needed.
    assert x0 < 0.0 < 1.0 < x1
    assert y0 < 0.0 < 1.0 < y1
    renderer = fig.canvas.get_renderer()
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -0.5, f"{text.get_text()} runs off the left edge"
        assert box.x1 <= fig.bbox.x1 + 0.5, f"{text.get_text()} runs off the right edge"


# ── It draws the graph on screen ─────────────────────────────────────────────


def test_a_pruned_node_is_not_in_the_figure():
    graph = _graph(behaviors=3)
    pos = _layout(graph)

    assert graph_figure.drawn_nodes(graph, pos, excluded=frozenset({1})) == sorted(
        set(graph.nodes) - {1}
    )


def test_the_edge_filters_are_the_components():
    """A probability floor, a probability minimum and a count minimum — so the
    figure and the screen agree on which transitions exist."""
    graph = _graph(behaviors=2, rare_edge=True)
    kept = graph_figure.drawn_nodes(graph, _layout(graph))

    everything = graph_figure.drawn_edges(graph, kept, prob_floor=0.0)
    assert (0, FAILURE_NODE_ID, pytest.approx(1 / 20), 1) in everything

    # The one-episode edge is 5% of Behavior 0's outgoing mass: a floor above
    # that drops it, and so does asking for two observations.
    assert not [e for e in graph_figure.drawn_edges(graph, kept, prob_floor=0.1) if e[3] == 1]
    assert not [
        e
        for e in graph_figure.drawn_edges(graph, kept, min_edge_count=2, prob_floor=0.0)
        if e[3] == 1
    ]


def test_a_graph_with_every_node_pruned_draws_nothing():
    """``None``, not an empty axes: a blank page in a paper reads as a plotting
    bug rather than as an empty result."""
    graph = _graph()

    assert (
        graph_figure.behavior_graph_figure(
            graph, pos=_layout(graph), excluded=frozenset(graph.nodes)
        )
        is None
    )
    assert (
        graph_figure.behavior_graph_figure_files(
            graph=graph, pos=_layout(graph), excluded=frozenset(graph.nodes)
        )
        is None
    )


def test_the_layout_it_is_handed_is_the_layout_it_draws():
    """Two layouts of one graph are two different pictures, so this is the one
    input the figure must not recompute for itself."""
    graph = _graph(behaviors=2)
    pos = {
        START_NODE_ID: (0.0, 0.0),
        0: (1.0, 1.0),
        1: (2.0, 1.0),
        SUCCESS_NODE_ID: (3.0, 0.0),
        FAILURE_NODE_ID: (3.0, 2.0),
    }

    fig = graph_figure.behavior_graph_figure(graph, pos=pos)

    drawn = {tuple(c.get_offsets()[0]) for c in fig.axes[0].collections}
    # Normalised into the unit square with y flipped: START is bottom-left of
    # its column, FAILURE top-right.
    assert (0.0, 1.0) in drawn
    assert (1.0, 0.0) in drawn


def test_a_synthetic_outcome_leaf_is_drawn_as_an_outcome():
    """The trajectory tree's nodes are synthetic, so its outcome leaves are not
    special nodes: only the component's ``symbol_override`` says which ended
    well, and the figure has to read the same thing."""
    graph = _graph(behaviors=1)
    leaf = 90_001
    graph.nodes[leaf] = BehaviorNode(leaf, "SUCCESS", 0, 5)
    graph.transition_counts[0] = {**graph.transition_counts.get(0, {}), leaf: 5}
    graph.transition_probs[0] = {leaf: 1.0}
    symbols = {leaf: "star"}

    assert graph_figure._role_of(graph, leaf, symbols) == SUCCESS_NODE_ID
    # And with no override it is what it looks like: an ordinary behavior.
    assert graph_figure._role_of(graph, leaf, None) is None

    fig = graph_figure.behavior_graph_figure(
        graph, pos={**_layout(graph), leaf: (3.0, 1.0)}, symbols=symbols
    )

    # Drawn as one: the outcome colour, the tick inside it, and the share of
    # episodes that got there — none of which its own node data would give.
    assert str(leaf) in _glyphs(fig)
    assert "SUCCESS\n25%" in _label_boxes(fig)


def test_every_role_has_its_own_glyph():
    """Shape, not colour alone: the figure has to survive a greyscale print and
    a reader who cannot tell the green from the red."""
    glyphs = {
        role: graph_figure.ROLE_GLYPHS[role]
        for role in (START_NODE_ID, SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID)
    }

    assert len(set(glyphs.values())) == len(glyphs)


def test_a_role_is_drawn_inside_its_node_rather_than_instead_of_it():
    """As on screen, and for the same reasons: one radius law for every node
    means a size comparison between two of them is always about episode counts,
    and a glyph inside a circle cannot shrink into a smudge the way a bare star
    or cross does at four points across."""
    graph = _graph(behaviors=2)

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph))

    # A body per node, and a glyph on top of the three that have a role.
    assert len(_circles(fig)) == len(graph.nodes)
    assert set(_glyphs(fig)) == {str(START_NODE_ID), str(SUCCESS_NODE_ID), str(FAILURE_NODE_ID)}
    circle = _circles(fig)[0].get_paths()[0].vertices
    assert all(c.get_paths()[0].vertices.shape == circle.shape for c in _circles(fig))


def test_a_rare_outcome_is_still_big_enough_to_be_identified():
    """One episode in a hundred ended here is a thing the figure has to be able
    to say. Below the floor the glyph inside the node is a smudge, and nothing
    is lost by it — the exact share is written in the label."""
    graph = _graph(behaviors=1, episodes=100)
    graph.nodes[SUCCESS_NODE_ID] = BehaviorNode(SUCCESS_NODE_ID, "SUCCESS", 0, 1)
    kept = graph_figure.drawn_nodes(graph, _layout(graph))

    radii = graph_figure._radii_pt(graph, kept, node_scale=1.0)

    assert radii[SUCCESS_NODE_ID] == pytest.approx(graph_figure.NODE_ROLE_MIN_PT)
    # A behavior that rare is left alone: it is one cluster among many, and its
    # size is the only thing saying how rare.
    graph.nodes[1] = BehaviorNode(1, "Behavior 1", 10, 1)
    kept = graph_figure.drawn_nodes(graph, {**_layout(graph), 1: (1.0, 1.0)})
    assert graph_figure._radii_pt(graph, kept, node_scale=1.0)[1] < graph_figure.NODE_ROLE_MIN_PT


def test_the_glyph_takes_the_colour_its_node_can_carry():
    """The component always draws white, because everything on its canvas sits on
    a dark background. Here the fill is whatever the page's colour encoding
    produced, and a white tick on pale yellow is not a tick."""
    assert graph_figure._glyph_color("#d55e00") == "white"
    assert graph_figure._glyph_color(graph_figure.GREYSCALE_COLOR) == paper.INK


# ── Labels ───────────────────────────────────────────────────────────────────


def test_behaviors_only_leaves_the_roles_to_their_glyphs():
    graph = _graph(behaviors=2)
    pos = _layout(graph)

    def _names(label_mode):
        fig = graph_figure.behavior_graph_figure(
            graph, pos=pos, label_mode=label_mode, outcome_shares=False
        )
        return {t.get_text() for t in fig.axes[0].texts}

    assert _names("all") == {"START", "SUCCESS", "FAILURE", "B0", "B1"}
    assert _names("behaviors") == {"B0", "B1"}
    assert _names("none") == set()


def test_an_outcome_says_what_share_of_episodes_reached_it():
    """The number a reader cannot get from the picture: marker size does not make
    small differences between two outcomes readable, which is exactly why the
    component writes it too."""
    graph = _graph(behaviors=1, episodes=20)

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph))

    texts = {t.get_text() for t in fig.axes[0].texts}
    # Half the episodes each, on a second line under the name.
    assert "SUCCESS\n50%" in texts
    assert "FAILURE\n50%" in texts
    # Not START: every episode begins there, so "100%" under it says nothing.
    assert "START" in texts


def test_the_share_outlives_the_name_it_was_written_under():
    """ "Behaviors only" drops a role's *name* because the glyph says which
    outcome it is. How many episodes ended there is the one thing the glyph
    cannot say, so the share stays."""
    graph = _graph(behaviors=1, episodes=20)

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph), label_mode="behaviors")

    assert {t.get_text() for t in fig.axes[0].texts} == {"B0", "50%"}


def test_a_share_under_one_percent_is_not_rounded_away():
    """ "One episode in three hundred ended here" is a different claim from "none
    did", and a graph of failures is often about the difference."""
    graph = _graph(behaviors=1, episodes=300)
    graph.nodes[SUCCESS_NODE_ID] = BehaviorNode(SUCCESS_NODE_ID, "SUCCESS", 0, 1)

    assert graph_figure._outcome_share(graph, SUCCESS_NODE_ID, 300) == "0.3%"
    assert graph_figure._outcome_share(graph, FAILURE_NODE_ID, 300) == "50%"


def test_the_shares_can_be_left_off():
    graph = _graph(behaviors=1)

    fig = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), outcome_shares=False, label_mode="behaviors"
    )

    assert {t.get_text() for t in fig.axes[0].texts} == {"B0"}


def test_a_dense_tree_starts_without_its_outcome_names():
    """The rule is the count of role nodes, not the kind of graph — so a folded
    tree that is down to three outcomes gets its names back."""
    graph = _graph(behaviors=2)
    kept = graph_figure.drawn_nodes(graph, _layout(graph))
    assert graph_figure.suggested_label_mode(graph, kept) == "all"

    leaves = {}
    for i in range(graph_figure.ROLE_LABEL_LIMIT + 1):
        nid = 90_000 + i
        graph.nodes[nid] = BehaviorNode(nid, "FAILURE", 0, 1)
        leaves[nid] = "x"
    kept = graph_figure.drawn_nodes(
        graph, {**_layout(graph), **{n: (5.0, i) for i, n in enumerate(leaves)}}
    )

    assert graph_figure.suggested_label_mode(graph, kept, leaves) == "behaviors"


def test_full_names_are_available_for_a_figure_with_room():
    graph = _graph(behaviors=1)
    pos = _layout(graph)

    short = graph_figure.behavior_graph_figure(
        graph, pos=pos, label_mode="behaviors", outcome_shares=False
    )
    full = graph_figure.behavior_graph_figure(
        graph, pos=pos, label_mode="behaviors", shorten_labels=False, outcome_shares=False
    )

    assert [t.get_text() for t in short.axes[0].texts] == ["B0"]
    assert [t.get_text() for t in full.axes[0].texts] == ["Behavior 0"]


def test_a_label_the_caller_wrote_is_drawn_as_written():
    """A VLM annotation or an experimenter's own name for a cluster was written
    to be read; nothing here gets to shorten it or drop it."""
    graph = _graph(behaviors=1, episodes=20)

    fig = graph_figure.behavior_graph_figure(
        graph,
        pos=_layout(graph),
        labels={0: "Reaches for the fork", SUCCESS_NODE_ID: "Placed"},
        label_mode="behaviors",
        shorten_labels=True,
    )

    # Every word of it, wherever the lines fell: wrapping is the only thing
    # allowed to touch a name the caller wrote, and it changes no word.
    texts = {" ".join(t.get_text().split()) for t in fig.axes[0].texts}
    assert "Reaches for the fork" in texts
    # Even in "behaviors only", since it is not the role's own name — and the
    # share still rides under it, as it does under a name the figure chose.
    assert "Placed 50%" in texts


def _row_of_behaviors(count: int, *, episodes: int = 10) -> BehaviorGraph:
    """*count* behaviors in a chain, laid out on one row — the shape that runs
    out of horizontal room, which is the room labels need."""
    nodes = {START_NODE_ID: BehaviorNode(START_NODE_ID, "START", 0, episodes)}
    counts: dict[int, dict[int, int]] = {START_NODE_ID: {0: episodes}}
    for i in range(count):
        nodes[i] = BehaviorNode(i, f"Behavior {i}", 100, episodes - i)
        if i < count - 1:
            counts[i] = {i + 1: episodes // 2}
    probs = {
        src: {tgt: c / (sum(tgts.values()) or 1) for tgt, c in tgts.items()}
        for src, tgts in counts.items()
    }
    return BehaviorGraph(
        nodes=nodes, transition_counts=counts, transition_probs=probs, num_episodes=episodes
    )


def test_no_two_labels_land_on_each_other():
    """A name is several times wider than the node it names, so a figure of any
    density has labels colliding long before it looks crowded. Each one goes
    below its node where it can and above, right or left of it where it cannot,
    which is what makes a real graph readable at all."""
    graph = _row_of_behaviors(5)
    pos = {START_NODE_ID: (-1.0, 0.0), **{i: (float(i), 0.0) for i in range(5)}}
    # Long enough that "below" cannot hold them all — the interesting case.
    labels = dict.fromkeys(range(5), "Slides the tray out")

    # Unwrapped on purpose: wrapping is the other half of the answer and would
    # make these fit below their nodes, leaving the fallback untested.
    fig = graph_figure.behavior_graph_figure(
        graph, pos=pos, labels=labels, width_in=paper.SINGLE_COLUMN_WIDTH_IN, label_wrap_pt=0.0
    )

    renderer = fig.canvas.get_renderer()
    placed = [text.get_window_extent(renderer) for text in fig.axes[0].texts]
    assert len(placed) == 6
    for first, second in itertools.combinations(placed, 2):
        overlap_x = min(first.x1, second.x1) - max(first.x0, second.x0)
        overlap_y = min(first.y1, second.y1) - max(first.y0, second.y0)
        assert overlap_x <= 0 or overlap_y <= 0, "two labels on the same patch of figure"
    # And it got there by using the other side, not by shrinking the type.
    assert {text.get_va() for text in fig.axes[0].texts} == {"top", "bottom"}


def test_a_label_never_lands_on_a_node():
    """Including its own: a name over the node it names is worse than no name."""
    graph = _row_of_behaviors(4)
    pos = {START_NODE_ID: (-1.0, 0.0), **{i: (float(i), 0.0) for i in range(4)}}

    fig = graph_figure.behavior_graph_figure(
        graph, pos=pos, labels=dict.fromkeys(range(4), "Slides the tray out"), node_scale=2.0
    )

    ax = fig.axes[0]
    renderer = fig.canvas.get_renderer()
    radii = graph_figure._radii_pt(graph, sorted(graph.nodes), node_scale=2.0)
    for text in ax.texts:
        box = text.get_window_extent(renderer)
        for nid, (x, y) in graph_figure._unit_positions(pos, sorted(graph.nodes)).items():
            px, py = ax.transData.transform((x, y))
            half = radii[nid] * fig.dpi / 72.0
            clear_x = box.x1 < px - half or box.x0 > px + half
            clear_y = box.y1 < py - half or box.y0 > py + half
            assert clear_x or clear_y, f"{text.get_text()!r} sits on node {nid}"


def test_a_label_sits_on_a_patch_and_not_under_a_halo():
    """Placement can keep a label off the nodes but not off the edges — a graph's
    edges converge exactly where its outcome labels have to go. The white
    background is a patch rather than a path effect for a reason that is not
    cosmetic: a path effect renders the glyphs as outlines, and a PDF whose
    labels are paths has no fonts in it (see the fonttype test)."""
    graph = _graph(behaviors=2)

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph))

    for text in fig.axes[0].texts:
        assert text.get_bbox_patch() is not None, "nothing behind this label"
        assert not text.get_path_effects(), "a path effect would cost the figure its fonts"


def test_a_long_name_breaks_onto_a_second_line_rather_than_across_the_graph():
    """Which is what makes a name worth drawing at all: "Slides the tray out
    from under the shelf" says something "B0" does not, and on one line it is
    four nodes wide and centred over none of them."""
    graph = _graph(behaviors=2)
    name = "Slides the tray out from under the shelf"
    # An explicit width, not the tuned default: this is about what wrapping does
    # to a name too wide for the room it has, and it should keep saying so when
    # somebody settles on a different default by looking at the figures.
    wrap_pt = 54.0

    def _drawn(**kwargs):
        return graph_figure.behavior_graph_figure(
            graph,
            pos=_layout(graph),
            # An empty label draws none, which leaves exactly one to measure.
            labels={0: name, 1: ""},
            label_mode="behaviors",
            outcome_shares=False,
            **kwargs,
        )

    wrapped, flat = _drawn(label_wrap_pt=wrap_pt), _drawn(label_wrap_pt=0.0)

    assert "\n" in _one_label(wrapped)
    assert _one_label(flat) == name
    # Narrower, which is the room these layouts are short of: levels divide the
    # horizontal space between siblings and have vertical space to spare. Held
    # against the wrap width rather than a ratio between the two, so the numbers
    # here follow the setting instead of the label size of the day. Twice it,
    # not once, because the last line allowed takes the overflow.
    assert _label_width_pt(wrapped) <= 2 * wrap_pt
    assert _label_width_pt(flat) > 2 * wrap_pt
    # And every word of it survives — a name cut short cannot be recovered.
    assert _one_label(wrapped).split() == name.split()


def test_the_break_is_measured_in_the_font_it_will_be_printed_in():
    """Not counted in characters. The component has to estimate, because it lays
    a tree out before the browser has the text; this has the renderer and the
    label's own font in hand, so "too wide" means the width the PDF gets."""
    graph = _graph(behaviors=1)
    name = "Reaches for the metal bowl"

    wrap_pt = 54.0
    fig = graph_figure.behavior_graph_figure(
        graph,
        pos=_layout(graph),
        labels={0: name},
        label_mode="behaviors",
        outcome_shares=False,
        label_wrap_pt=wrap_pt,
    )

    renderer = fig.canvas.get_renderer()
    (text,) = fig.axes[0].texts
    font = text.get_fontproperties()
    limit = wrap_pt * fig.dpi / 72.0
    for line in text.get_text().split("\n"):
        assert renderer.get_text_width_height_descent(line, font, False)[0] <= limit


def test_a_name_with_no_space_in_it_is_left_wide():
    """There is no break in "Behavior12" a reader would recognise, and
    hyphenating one in is worse than a wide label."""
    assert graph_figure._wrapped_lines("Behavior12#2", fits=lambda _: False, max_lines=2) == [
        "Behavior12#2"
    ]


def test_the_last_line_allowed_takes_the_overflow():
    """A third line is the room the level below needs, so the remainder runs
    long instead — readable, where a truncated name is not."""
    lines = graph_figure._wrapped_lines(
        "one two three four five", fits=lambda _: False, max_lines=2
    )

    assert lines == ["one", "two three four five"]
    assert " ".join(lines).split() == "one two three four five".split()


def test_the_line_cap_is_a_cap_and_not_a_target():
    """Most names are two words and take one line. The cap is there so a label
    cannot grow down into the level below it."""
    assert graph_figure._wrapped_lines(
        "Picks up the fork", fits=lambda _: True, max_lines=graph_figure.LABEL_MAX_LINES
    ) == ["Picks up the fork"]


def test_an_outcomes_share_keeps_its_own_line():
    """The share is not counted against the name's line limit: it is a second
    thing written about the node, not the name running on."""
    graph = _graph(behaviors=1, episodes=20)

    fig = graph_figure.behavior_graph_figure(
        graph,
        pos=_layout(graph),
        labels={SUCCESS_NODE_ID: "Placed both items on the table"},
        # Narrow enough that the name uses every line it is allowed, so the
        # share has to be a line the limit did not count.
        label_wrap_pt=40.0,
    )

    placed = next(t.get_text() for t in fig.axes[0].texts if "Placed" in t.get_text())
    assert placed.endswith("\n50%")
    assert len(placed.split("\n")) == graph_figure.LABEL_MAX_LINES + 1


def test_the_label_size_is_a_lever():
    """The last resort for a figure whose names have nowhere left to go. The
    default is the paper's own body size, so a name on the graph reads at the
    size the sentence about it does; room comes from the wrap width and the
    placement rather than from shrinking the type."""
    graph = _graph(behaviors=1)

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph), label_font_pt=5.0)

    assert graph_figure.LABEL_FONT_PT == paper.BASE_RC["font.size"]
    assert all(text.get_fontsize() == 5.0 for text in fig.axes[0].texts)


def test_an_empty_name_silences_one_node_and_not_the_rest():
    """The lever for the one node whose name has nowhere to go, short of
    dropping every name in the figure."""
    graph = _graph(behaviors=2)

    fig = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), labels={0: ""}, label_mode="behaviors", outcome_shares=False
    )

    assert {t.get_text() for t in fig.axes[0].texts} == {"B1"}


def test_an_unknown_label_mode_is_refused():
    graph = _graph()

    with pytest.raises(ValueError, match="label_mode"):
        graph_figure.behavior_graph_figure(graph, pos=_layout(graph), label_mode="outcomes")


# ── Size and encoding ────────────────────────────────────────────────────────


def test_a_deeper_layout_gets_a_taller_figure():
    """Every level needs its nodes and a line of label under them; a flat graph
    gets the wide, short band a paper figure wants."""
    graph = _graph(behaviors=3)
    flat = {n: (float(i), 0.0) for i, n in enumerate(graph.nodes)}
    deep = {n: (0.0, float(i)) for i, n in enumerate(graph.nodes)}
    kept = sorted(graph.nodes)

    short = graph_figure.default_height_in(flat, kept, width_in=7.16)
    tall = graph_figure.default_height_in(deep, kept, width_in=7.16)

    assert short == pytest.approx(graph_figure.MIN_HEIGHT_IN)
    assert tall > short
    # Never taller than it is wide: a figure the LaTeX has to scale down is one
    # whose type is no longer the size it was exported at.
    assert tall <= 7.16


def test_the_height_budgets_rows_and_not_coordinates():
    """The layered layout centres each layer, so a four-node layer sits at ±0.5
    and ±1.5 while a three-node one sits at −1, 0, 1 — seven distinct y values
    for four rows of nodes. The four are what a reader sees: two nodes half a
    row apart are in different columns and never meet. Budgeting for the
    coordinates made every figure of an uneven graph far taller than its own
    drawing."""
    graph = _graph(behaviors=5)
    four = {
        n: (0.0, offset) for n, offset in zip(graph.nodes, (-1.5, -0.5, 0.5, 1.5), strict=False)
    }
    three = {
        n: (1.0, offset) for n, offset in zip(list(graph.nodes)[4:], (-1.0, 0.0, 1.0), strict=False)
    }
    pos = {**four, **three}
    kept = sorted(pos)

    assert len({round(pos[n][1], 6) for n in kept}) == 7
    assert graph_figure._row_pitch(pos, kept) == pytest.approx(1.0)
    assert graph_figure.default_height_in(pos, kept, width_in=7.16) == pytest.approx(
        graph_figure.ROW_PITCH_IN * 4 + 0.2
    )


def test_a_layout_that_steps_by_something_else_is_measured_not_assumed():
    """Both layouts here step by 1, but the pitch is read off the layout rather
    than hardcoded, so one that does not is budgeted correctly instead of being
    given four times the height it needs."""
    graph = _graph(behaviors=3)
    pos = {n: (0.0, 0.25 * i) for i, n in enumerate(graph.nodes)}
    kept = sorted(pos)

    assert graph_figure._row_pitch(pos, kept) == pytest.approx(0.25)
    # Five nodes in one column is five rows, whatever the units say.
    assert graph_figure.default_height_in(pos, kept, width_in=7.16) == pytest.approx(
        graph_figure.ROW_PITCH_IN * len(kept) + 0.2
    )


def test_a_more_visited_node_is_drawn_bigger():
    graph = _graph(behaviors=3)
    kept = graph_figure.drawn_nodes(graph, _layout(graph))

    radii = graph_figure._radii_pt(graph, kept, node_scale=1.0)

    assert radii[0] > radii[2]  # Behavior 0 is in every episode, Behavior 2 in fewer.
    assert radii[0] == pytest.approx(graph_figure.NODE_BASE_PT + graph_figure.NODE_SLOPE_PT)
    # START is a fixed anchor: scaling it by "episodes that visited it" says
    # nothing, since every episode does.
    assert radii[START_NODE_ID] == pytest.approx(graph_figure.NODE_START_PT)


def test_node_scale_moves_every_node_together():
    graph = _graph(behaviors=2)
    kept = graph_figure.drawn_nodes(graph, _layout(graph))

    plain = graph_figure._radii_pt(graph, kept, node_scale=1.0)
    bigger = graph_figure._radii_pt(graph, kept, node_scale=2.0)

    assert all(bigger[n] == pytest.approx(2 * plain[n]) for n in kept)


def test_probability_is_encoded_once_and_every_edge_is_solid():
    """One channel, no ramp, no alpha. A figure is read at a glance: a second
    encoding of the same number is a decode step rather than reinforcement, and
    the faint end of a grey ramp is an edge the reader does not see at all —
    often the rare transition the graph is about."""
    graph = _graph(behaviors=2, rare_edge=True)

    fig = graph_figure.behavior_graph_figure(graph, pos=_layout(graph), prob_floor=0.0)

    edges = fig.axes[0].patches
    assert len({tuple(round(v, 4) for v in p.get_edgecolor()) for p in edges}) == 1
    assert all(p.get_edgecolor()[3] == 1.0 for p in edges), "no transparency"
    # Width is the encoding, and it spreads far enough to read: the widest line
    # is several times the thinnest, and the thinnest is still a visible line.
    widths = sorted(p.get_linewidth() for p in edges)
    assert widths[0] >= graph_figure.EDGE_BASE_PT
    assert widths[-1] > 3 * widths[0]


def test_a_rare_edge_is_dashed_rather_than_faded():
    """The state-merging mode keeps outcome outliers. Dashed, so a reader sees
    "individual episodes" rather than a probability estimate — a distinct
    channel, unlike a fade, which is both invisible at a glance and the thing a
    print pipeline is least predictable about."""
    graph = _graph(behaviors=2, rare_edge=True)

    fig = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), prob_floor=0.0, rare_edge_max_count=1
    )

    dashed = [p for p in fig.axes[0].patches if p.get_linestyle() != "solid"]
    assert len(dashed) == 1


def test_greyscale_drops_the_colour_encoding_and_keeps_the_shapes():
    graph = _graph(behaviors=2)
    colors = {0: "#ff0000", 1: "#00ff00"}

    fig = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), colors=colors, greyscale=True
    )

    faces = {tuple(c.get_facecolor()[0]) for c in _circles(fig)}
    assert len(faces) == 1
    # The roles are still told apart, by the glyph inside the node — and in ink
    # rather than white, which on a pale grey fill would be no glyph at all.
    glyphs = _glyphs(fig)
    assert len({c.get_paths()[0].vertices.shape for c in glyphs.values()}) == len(glyphs)
    assert all(
        tuple(round(v, 3) for v in c.get_facecolor()[0])
        == pytest.approx(_rgba(paper.INK), abs=1e-3)
        for c in glyphs.values()
    )


def test_the_pages_colour_encoding_carries_over():
    graph = _graph(behaviors=2)

    fig = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), colors={0: "#ff0000"}, greyscale=False
    )

    reds = [
        c
        for c in _circles(fig)
        if tuple(round(v, 3) for v in c.get_facecolor()[0]) == (1.0, 0.0, 0.0, 1.0)
    ]
    assert len(reds) == 1


def test_probabilities_can_be_written_on_the_edges():
    graph = _graph(behaviors=1)

    fig = graph_figure.behavior_graph_figure(
        graph, pos=_layout(graph), edge_probabilities=True, label_mode="none"
    )

    # Two edges out of the chain's one behavior, plus START's — every one
    # labelled, and nothing else is drawing text in this figure.
    assert len(fig.axes[0].texts) == len(
        graph_figure.drawn_edges(graph, graph_figure.drawn_nodes(graph, _layout(graph)))
    )


def test_the_layout_is_computed_when_none_is_given():
    """For a caller outside the app — a script writing a figure straight out of
    a run — rather than as the app's path, which passes the layout it drew."""
    graph = _graph(behaviors=2)

    fig = graph_figure.behavior_graph_figure(graph)

    assert fig is not None
    assert len(_circles(fig)) == len(graph.nodes)


# ── The shared paper style ───────────────────────────────────────────────────


def test_the_style_is_applied_when_the_file_is_written_not_only_when_it_is_drawn():
    """A font family and ``pdf.fonttype`` are resolved at save time, so a figure
    built inside the style and saved outside it comes out in the default
    sans-serif with Type 3 fonts — which is why :func:`paper.rendered` is meant
    to be called inside the context."""
    import matplotlib.pyplot as plt

    with paper.paper_rc():
        assert plt.rcParams["pdf.fonttype"] == 42
        assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["pdf.fonttype"] != 42 or paper.BASE_RC["pdf.fonttype"] == 42


def test_a_figure_with_axes_asks_for_the_chart_extras():
    """The typography is shared; the axis furniture is not. A node-link diagram
    has no axes, and switching the grid on for one would draw it."""
    import matplotlib.pyplot as plt

    assert "axes.grid" not in paper.BASE_RC
    with paper.paper_rc(paper.CHART_RC):
        assert plt.rcParams["axes.grid"] is True
        # And the shared half is still in force.
        assert plt.rcParams["pdf.fonttype"] == 42


def test_nothing_it_writes_carries_a_timestamp():
    import matplotlib.pyplot as plt

    with paper.paper_rc():
        fig = plt.figure(figsize=(1, 1))
        try:
            assert b"CreationDate" not in paper.rendered(fig, "pdf", 100)
        finally:
            plt.close(fig)
