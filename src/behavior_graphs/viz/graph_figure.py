"""A behavior graph as a figure that can go into the paper.

The app already exports the graph: the SVG the browser component has on screen,
serialized. That is the right export for "what I am looking at" — it carries any
node somebody dragged — and the wrong one for a paper. It comes out in the
browser's sans-serif at whatever size the canvas happened to be, in screen
colours chosen for a dark background, with the type scaled by however the LaTeX
has to squeeze it into a column.

This draws the same graph in matplotlib instead, at the column width it will be
printed at, in the template's serif, with TrueType outlines and no timestamp —
see :mod:`behavior_graphs.viz.paper` for why each of those is not cosmetic.

**It draws what the screen draws.** Same layout (the caller hands over the
positions the component was given), same marker per role, same pruning, and the
same two quantities encoded: line width for transition probability, node radius
for how many episodes visited a node. What changes is the *scale* of them — a
stroke measured in screen pixels is a slab at 8 pt — so they are re-expressed in
points here rather than reused as numbers.

**One channel per quantity, and no transparency at all.** The screen encodes
probability twice, as width *and* as grey level, which is right for a display
someone is going to sit and explore. A figure is read at a glance, and a second
encoding of the same number is not reinforcement there — it is a decode step
("how dark is dark?") for something the width has already said. Worse, the faint
end of any such ramp is an edge the reader simply does not see: the rare
transition that a graph of failures is often *about*. So every edge is drawn in
one solid colour, dark enough to be seen, and thickness alone carries p. The
lightest line is a line.

Alpha is out for the same reason and one more: it compounds where edges cross,
so two overlapping faint edges read as one darker edge that is not there — and
what a print pipeline does with it is its own business.

**A node is a circle whatever it is, with its role drawn inside it** — a tick
for an episode that ended well, a cross for one that did not, a triangle for
START, a square for END. That is what the component does, and for the same two
reasons: one radius law for every node means a size comparison between two of
them is always about episode counts, and a role drawn *inside* a node cannot
shrink into a smudge the way a bare star or cross does at four points across.
Role nodes also get a radius floor, because a marker too small to identify says
nothing at all — and the floor costs no information, since their exact share is
written in the label.

**An outcome's label carries the share of episodes that reached it.** As on
screen, and for the reason the component gives there: ``P(reach this terminal)``
is one of the most useful numbers in the whole figure, and marker size alone
does not make small differences between two outcomes readable. It goes on a
second line under the name, which is also the narrower shape — outcomes are the
most numerous labels at a tree's most crowded level.

Labels are *placed*, not pinned: each goes below its node if it can, and
otherwise above, right or left of it — whichever side is free of the other
labels and of every node. Deterministic (nodes in a fixed order, the sides in a
fixed order), so the same graph still exports the same bytes. Without this a
figure is unreadable well before it is crowded, because a name is several times
wider than the node it names.

A label sits on a white patch, because placement can keep it off the nodes but
not off the edges — a graph's edges converge exactly where its outcome labels
have to go, and a name with four 2 pt lines through it is not a name. The patch
is a *background artist*, which matters: a path effect (``withStroke``, the usual
way to halo text) makes the backend render the glyphs as outlines, and a PDF
whose labels are paths has no fonts in it at all — nothing to search, nothing to
copy, and nothing for the ``fonttype`` guarantee this module is built around to
apply to. A patch behind the text leaves the text as text. If a label is still
hard to read once placed, the figure is too small for the graph: raise the
height, drop the label size, or hide the names the glyphs already account for.

Nothing that is an affordance is drawn: no fold badges, no selection ring, no
"click a node" line. They are controls, and a figure has nothing to click.

Pure: data in, ``matplotlib.figure.Figure`` out, no Streamlit. The panel that
offers it lives in
:mod:`behavior_graphs.viz.streamlit.components.graph_figure_export`.
"""

from __future__ import annotations

import functools
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np

from behavior_graphs.core.graph import (
    END_NODE_ID,
    FAILURE_NODE_ID,
    START_NODE_ID,
    SUCCESS_NODE_ID,
    BehaviorGraph,
)
from behavior_graphs.viz import paper
from behavior_graphs.viz.labels import short_node_label

#: The glyph drawn *inside* a node for each role, in the component's own
#: vocabulary: a tick for an episode that ended well, a cross for one that did
#: not, a right-pointing triangle for START, a square for END. A behavior has
#: none — the circle is the whole node. Shape rather than colour alone, so the
#: roles survive a greyscale print and a reader with deuteranopia.
#:
#: ``"tick"`` is resolved by :func:`_glyph_marker`; the rest are matplotlib
#: marker codes.
ROLE_GLYPHS = {
    START_NODE_ID: ">",
    SUCCESS_NODE_ID: "tick",
    FAILURE_NODE_ID: "X",
    END_NODE_ID: "s",
}

#: Which role each of the component's symbol names stands for. A caller that
#: passes ``symbol_override`` to the component passes it here too, and it is not
#: a nicety: the trajectory tree's nodes are synthetic, so its outcome leaves are
#: not special nodes and only the override says which of them ended well.
SYMBOL_ROLES = {
    "star": SUCCESS_NODE_ID,
    "x": FAILURE_NODE_ID,
    "diamond": START_NODE_ID,
    "square": END_NODE_ID,
    "circle": None,
}

#: The roles whose label carries the share of episodes that reached them: every
#: terminal, which is every role but START. Every episode starts at START, so
#: "100%" written under it is not information.
TERMINAL_ROLES = frozenset({SUCCESS_NODE_ID, FAILURE_NODE_ID, END_NODE_ID})

#: How far across a node its glyph is drawn, as a fraction of the diameter.
#: Big enough to identify at a glance, small enough to leave a ring of the
#: node's own fill around it — which is what still reads as a node.
GLYPH_FRACTION = 0.58


@functools.cache
def _tick_marker():
    """A filled check mark as a marker path, in the unit marker square.

    Built rather than picked out of matplotlib's marker table because there is
    no check in it, and a check is what the component draws for an episode that
    ended well (its symbol is *named* ``star``; the glyph it draws is ✓). Not a
    font glyph either: ✓ is missing from the Times-metric faces this runs
    against, and a missing glyph in a paper figure is a hollow box.

    Cached, and matplotlib imported inside, so the module stays importable
    without a backend and every figure shares one path object.
    """
    from matplotlib.path import Path

    elbow_arm, elbow, long_arm = (
        np.array([-0.42, 0.04]),
        np.array([-0.13, -0.30]),
        np.array([0.44, 0.34]),
    )
    half = 0.115

    def _normal(vector):
        unit_vector = vector / np.linalg.norm(vector)
        return np.array([-unit_vector[1], unit_vector[0]])

    short_n, long_n = _normal(elbow - elbow_arm), _normal(long_arm - elbow)
    # The elbow is offset further than `half`, along the bisector: the miter of
    # a stroke this thick, or the two arms of the tick would not meet.
    bisector = (short_n + long_n) / np.linalg.norm(short_n + long_n)
    miter = bisector * (half / float(np.dot(bisector, short_n)))
    verts = np.array(
        [
            elbow_arm + short_n * half,
            elbow + miter,
            long_arm + long_n * half,
            long_arm - long_n * half,
            elbow - miter,
            elbow_arm - short_n * half,
            elbow_arm + short_n * half,
        ]
    )
    return Path(verts, [Path.MOVETO, *[Path.LINETO] * 5, Path.CLOSEPOLY])


def _glyph_marker(role: int | None):
    """The marker for a role's inner glyph, or ``None`` for a plain node."""
    glyph = ROLE_GLYPHS.get(role)
    return _tick_marker() if glyph == "tick" else glyph


#: Print colours for the roles. Muted against the app's screen palette — a
#: saturated green that reads well on a dark canvas is loud on white paper — and
#: from the Okabe-Ito set, so the pair a reader must not confuse (an episode that
#: ended well and one that did not) stays separable under every common form of
#: colour blindness.
ROLE_COLORS = {
    START_NODE_ID: "#4c72b0",
    SUCCESS_NODE_ID: "#009e73",
    FAILURE_NODE_ID: "#d55e00",
    END_NODE_ID: "#9a9a9a",
}

#: A behavior with nothing else said about it. Light, because the node's own
#: label is drawn beside it and a dark fill competes with it for the eye.
BEHAVIOR_COLOR = "#dfe3e8"

#: Every node in one grey when the figure has to survive a greyscale print with
#: its colour information intact — the roles are still told apart by marker.
GREYSCALE_COLOR = "#d4d4d4"

#: Above this many role nodes (outcomes, start, end), naming them all is noise
#: rather than information — thirty crosses under thirty "FAILURE"s — so
#: :func:`suggested_label_mode` starts a figure on "behaviors" instead.
ROLE_LABEL_LIMIT = 8

#: Which nodes get their name drawn beside them.
#:
#: ``behaviors`` is the one worth knowing about: on a trajectory tree every path
#: ends in an outcome, so "all" writes FAILURE thirty times under thirty
#: crosses that already say it. The marker is the label for a role; the text is
#: only needed for a behavior, whose number is the one thing a reader cannot see.
LABEL_MODES = ("all", "behaviors", "none")

#: Node radius in points: ``NODE_BASE_PT + slope × (episodes / max episodes)``.
#: The screen formula is ``6 + 16 × frac`` in *pixels* on a canvas around a
#: thousand wide, which lands close to this at the double-column width — the
#: encoding is re-expressed at print scale rather than carried over as numbers.
NODE_BASE_PT = 2.4
NODE_SLOPE_PT = 6.2
#: START is a fixed anchor rather than a visited cluster — it is where every
#: episode begins, so scaling it by "how many episodes visited it" says nothing.
#: Same reasoning as the component's fixed diamond.
NODE_START_PT = 4.6
#: No role node is drawn smaller than this, however few episodes reached it. A
#: rare outcome still has to be identifiable *as* an outcome, and below about
#: this radius the glyph inside it is a smudge. Nothing is lost to the floor:
#: the share of episodes that reached it is written in its label.
NODE_ROLE_MIN_PT = 4.2

#: Edge width in points: ``EDGE_BASE_PT + slope × p``, the figure's only
#: encoding of probability. The base is what a p→0 transition is drawn at, and
#: it is deliberately a visible line rather than a hairline — an edge worth
#: keeping in the graph is an edge worth seeing. The slope spreads the rest over
#: a range where a glance separates a common transition from a rare one:
#: roughly 0.4 pt at p→0 against 2 pt at p=1, a fivefold difference.
EDGE_BASE_PT = 0.4
EDGE_SLOPE_PT = 1.6

#: Arrowhead size in points, on the same shape as the width: it has to grow with
#: the line so a thick edge does not out-weigh its own head, and it has to start
#: big enough to be seen, since an arrowhead nobody can make out is a direction
#: the figure did not convey.
ARROW_BASE_PT = 5.0
ARROW_SLOPE_PT = 3.5

#: How far edges bow. The component curves by chord length; one constant here,
#: because a figure is small enough that the long edges are the ones that need
#: the bow and they all need about the same.
EDGE_CURVE = 0.13

#: A rare-but-preserved edge (an outcome outlier the state-merging graph mode
#: keeps) is drawn dashed at a floored weight, as on screen: the reader should
#: see "individual episodes", not a probability estimate.
RARE_PROB_FLOOR = 0.3
RARE_DASHES = (1.6, 1.1)

#: The one colour every edge is drawn in. Dark enough that the thinnest line is
#: unambiguously there, lighter than the node outlines so the nodes stay the
#: subject. One value, not a ramp — see the module docstring.
EDGE_COLOR = "#555555"

#: Room around the drawing, in inches, on top of what the nodes and their labels
#: need. Enough that a bowed edge leaving the outermost node does not clip.
FIGURE_PAD_IN = 0.04

#: Vertical room one level of the layout needs: a node, up to two lines of
#: label under it (an outcome's name over its share), and a gap. What
#: :func:`default_height_in` multiplies up.
ROW_PITCH_IN = 0.52
#: No shorter than this however flat the graph is — two levels in a 0.9-inch
#: figure is a diagram of the arrows, with the nodes as punctuation.
MIN_HEIGHT_IN = 1.5

#: Node names and outcome shares, in points. The paper's body size, so a name
#: on the graph reads at the size the sentence about it does — which is what a
#: reader comparing figure to text wants, and what a smaller size costs even
#: though it would buy room. Room comes from :data:`LABEL_WRAP_PT` and the
#: placement instead, and the panel's slider is there for a figure that needs
#: to go below this.
LABEL_FONT_PT = 8.0

#: Clear air between a node's rim and its label, in points.
LABEL_GAP_PT = 1.6

#: How wide a label may run, in points, before it breaks onto another line.
#: About an inch and a half — twenty-odd characters at 8 pt, a third of a
#: single-column figure. Settled by looking at the figures rather than derived:
#: tighter reads as a stack of two-word fragments, and much looser puts a name
#: back across its neighbours. A fixed width rather than a share of the figure,
#: since it is a property of reading and a narrow figure has the same reader.
#:
#: Wrapping trades width for height, which is the trade these layouts want —
#: levels divide the horizontal room between siblings and run out of it long
#: before they run out of vertical room between depths. It is also what makes
#: :func:`_place_labels` succeed: a narrow label has somewhere to go.
LABEL_WRAP_PT = 102.0

#: Lines a name may take. A cap so a label cannot grow without bound down into
#: the level below it, not a target — most names are two words and take one.
#: Three rather than the component's two because the last line allowed takes
#: whatever is left over: at two lines a name of a dozen words comes out as a
#: short line over a very long one, which is the shape wrapping is meant to
#: avoid. The overflow runs long rather than being cut, since a reader can read
#: a wide name and cannot recover a truncated one. An outcome's share is not
#: counted against this — it is its own line under whatever the name wrapped to.
LABEL_MAX_LINES = 3

#: Where a label may sit, in preference order. Below first, because a row of
#: names under a row of nodes is the shape a reader expects; the rest are what a
#: crowded figure falls back to. See :func:`_place_labels`.
LABEL_POSITIONS = ("below", "above", "right", "left")

#: How many rounds of "place the labels, then fit the margins" a figure gets.
#: They feed each other — placement reads how far apart the nodes are in points,
#: and fitting the margins moves them — and two rounds settle it. A fixed count
#: rather than a convergence test, so two exports of one graph cannot differ.
LAYOUT_PASSES = 3


def _role(graph: BehaviorGraph, node_id: int) -> int | None:
    """Which special role a node plays, or ``None`` for a behavior.

    By ``cluster_id``, not by node id: the trajectory tree builds a synthetic
    graph whose ids are fresh per render, and its terminals would otherwise all
    read as behaviors.
    """
    node = graph.nodes.get(node_id)
    if node is None or not node.is_special:
        return None
    return node.cluster_id


def drawn_nodes(
    graph: BehaviorGraph,
    pos: Mapping[int, tuple[float, float]],
    excluded: frozenset[int] = frozenset(),
) -> list[int]:
    """The nodes the figure will draw, in a stable order.

    Sorted so two exports of one graph put the artists down in the same order —
    which is what makes the bytes comparable.
    """
    return sorted(nid for nid in graph.nodes if nid not in excluded and nid in pos)


def drawn_edges(
    graph: BehaviorGraph,
    kept: Sequence[int],
    *,
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    prob_floor: float = 0.01,
) -> list[tuple[int, int, float, int]]:
    """``(src, tgt, probability, count)`` for every edge the figure will draw.

    The same three filters the component applies, in the same order, so the
    figure and the screen agree on which transitions exist: a probability floor,
    a caller-supplied probability minimum, and a count minimum.
    """
    inside = set(kept)
    edges = []
    for src, targets in graph.transition_probs.items():
        if src not in inside:
            continue
        for tgt, prob in targets.items():
            if tgt not in inside or prob < max(prob_floor, min_edge_prob):
                continue
            count = graph.transition_counts.get(src, {}).get(tgt, 0)
            if min_edge_count > 0 and count < min_edge_count:
                continue
            edges.append((src, tgt, float(prob), int(count)))
    return sorted(edges)


def _unit_positions(
    pos: Mapping[int, tuple[float, float]], kept: Sequence[int]
) -> dict[int, tuple[float, float]]:
    """The layout normalised into the unit square, y up.

    Each axis is normalised independently, which is what the component does to
    fill its canvas — so the figure's proportions are set by its width and
    height rather than by how far apart the layout happened to put things. The
    layouts feeding this are level-based (BFS depth, tree depth): the distance
    between two levels carries no meaning that stretching would destroy.

    y is flipped because the layouts count depth downward while a figure's
    y-axis points up.
    """
    xs = np.array([pos[n][0] for n in kept], dtype=float)
    ys = np.array([pos[n][1] for n in kept], dtype=float)
    span_x = float(xs.max() - xs.min()) or 1.0
    span_y = float(ys.max() - ys.min()) or 1.0
    return {
        n: (
            (float(pos[n][0]) - float(xs.min())) / span_x,
            1.0 - (float(pos[n][1]) - float(ys.min())) / span_y,
        )
        for n in kept
    }


def _radii_pt(
    graph: BehaviorGraph,
    kept: Sequence[int],
    *,
    node_scale: float,
    symbols: Mapping[int, str] | None = None,
) -> dict[int, float]:
    """Node radius in points, keyed by node id."""
    max_episodes = max((graph.nodes[n].num_episodes for n in kept), default=1) or 1
    radii = {}
    for nid in kept:
        role = _role_of(graph, nid, symbols)
        if role == START_NODE_ID:
            radii[nid] = NODE_START_PT * node_scale
            continue
        frac = graph.nodes[nid].num_episodes / max_episodes
        radius = NODE_BASE_PT + NODE_SLOPE_PT * float(np.clip(frac, 0.0, 1.0))
        if role is not None:
            radius = max(radius, NODE_ROLE_MIN_PT)
        radii[nid] = radius * node_scale
    return radii


def _total_episodes(graph: BehaviorGraph, kept: Sequence[int]) -> int:
    """How many episodes the graph is *of* — the denominator of every share.

    START's own count, which is what the component uses: every episode passes
    through it. A trajectory tree's root can be missing after pruning, so the
    graph's recorded count and then the busiest node stand in for it.
    """
    start = graph.nodes.get(START_NODE_ID)
    if start is not None and start.num_episodes:
        return int(start.num_episodes)
    if graph.num_episodes:
        return int(graph.num_episodes)
    return int(max((graph.nodes[n].num_episodes for n in kept), default=0))


def _outcome_share(graph: BehaviorGraph, node_id: int, total_episodes: int) -> str:
    """The share of episodes that reached *node_id*, formatted as on screen.

    A terminal under 1% gets a decimal place rather than reading as ``0%`` —
    "one episode in three hundred ended here" is a different claim from "none
    did". Everything above rounds, which is what a reader compares at a glance.
    """
    if total_episodes <= 0:
        return ""
    frac = graph.nodes[node_id].num_episodes / total_episodes
    return f"{round(frac * 100)}%" if frac >= 0.01 else f"{frac * 100:.1f}%"


def _glyph_color(face: str) -> str:
    """White or ink for the glyph inside a node, whichever its fill can carry.

    The component always draws white, because everything on its canvas is a
    screen colour on a dark background. Here the fill is whatever the page's
    "colour by value" encoding produced, or the greyscale option's pale grey,
    and a white tick on either is not a tick — so the choice is made from the
    fill's luminance rather than fixed.
    """
    from matplotlib.colors import to_rgb

    red, green, blue = to_rgb(face)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.55 else paper.INK


def _node_label(graph: BehaviorGraph, node_id: int, *, shorten: bool) -> str:
    name = graph.nodes[node_id].name
    return short_node_label(name) if shorten else name


def _role_of(graph: BehaviorGraph, node_id: int, symbols: Mapping[int, str] | None) -> int | None:
    """The role a node is *drawn* as, or ``None`` for a behavior.

    The caller's glyph wins over the node's own kind, as it does in the
    component, and that is not a nicety: the trajectory tree's outcome leaves
    are ordinary synthetic nodes carrying a ``"star"`` or an ``"x"``, and
    without the override they would be drawn as behaviors — the wrong glyph, the
    wrong colour, and no share in the label.
    """
    if symbols is not None and node_id in symbols:
        return SYMBOL_ROLES.get(symbols[node_id])
    return _role(graph, node_id)


def _label_for(
    graph: BehaviorGraph,
    node_id: int,
    *,
    given: str | None,
    shorten: bool,
    label_mode: str,
    symbols: Mapping[int, str] | None,
    share: str = "",
) -> str:
    """The text to draw beside a node, or ``""`` for none.

    A caller's explicit name always wins over the node's own: it was written to
    be read. Otherwise *label_mode* decides, and a node counts as a role because
    the caller gave it a role's glyph or because it is a special node.

    *share* — an outcome's reach probability — is appended on its own line, and
    it survives ``"behaviors"`` mode alone: that mode drops the *name* of a role
    because its glyph already says which outcome it is, which is exactly what
    the glyph cannot say about how many episodes ended there.
    """
    role = _role_of(graph, node_id, symbols)
    # `is not None`, so an empty string means "no name for this one" rather than
    # "no name given" — which is how a caller silences one crowded node without
    # silencing every node, and what the component does with an empty label too.
    if given is not None:
        name = given
    elif label_mode == "none":
        name = ""
    elif label_mode == "behaviors" and role is not None:
        name = ""
    else:
        name = _node_label(graph, node_id, shorten=shorten)
    if share and label_mode != "none" and role in TERMINAL_ROLES:
        return f"{name}\n{share}" if name else share
    return name


def behavior_graph_figure(
    graph: BehaviorGraph,
    *,
    pos: Mapping[int, tuple[float, float]] | None = None,
    excluded: frozenset[int] = frozenset(),
    labels: Mapping[int, str] | None = None,
    colors: Mapping[int, str] | None = None,
    symbols: Mapping[int, str] | None = None,
    label_mode: str = "all",
    min_edge_prob: float = 0.0,
    min_edge_count: int = 0,
    prob_floor: float = 0.01,
    rare_edge_max_count: int = 0,
    width_in: float = paper.DOUBLE_COLUMN_WIDTH_IN,
    height_in: float | None = None,
    arrows: bool = True,
    edge_probabilities: bool = False,
    shorten_labels: bool = True,
    label_font_pt: float = LABEL_FONT_PT,
    label_wrap_pt: float = LABEL_WRAP_PT,
    outcome_shares: bool = True,
    node_scale: float = 1.0,
    greyscale: bool = False,
):
    """The graph as a figure, or ``None`` when there is nothing to draw.

    ``None`` rather than an empty axes: pruning can leave a graph with no nodes,
    and a blank page in a paper looks like a plotting bug rather than an empty
    result.

    Args:
        pos: The layout the component was given, ``{node id: (x, y)}``. Passing
            the same positions is what makes the figure the graph on screen
            rather than a second opinion about it; without them a level-based
            layout is computed here (:func:`~behavior_graphs.viz.graph_plot.layered_layout`).
            Nodes a reader dragged are *not* reflected — those live in the
            iframe, and the SVG export is what captures them.
        excluded: Nodes to leave out, as computed by the page's pruning.
        labels: Text to draw per node; defaults to the node's own name, shortened
            unless *shorten_labels* is off. An empty string draws no label, which
            is how a caller silences one crowded node without silencing all.
        colors: Fill per node, overriding the role colours — the page's "colour
            by value / timesteps / outcome" encoding. Ignored under *greyscale*.
        symbols: Glyph per node by the component's own names (see
            :data:`SYMBOL_ROLES`), as passed to it. Required to draw a
            trajectory tree correctly: its outcome leaves are synthetic nodes,
            and this is what says which is which.
        label_mode: Which names to draw — see :data:`LABEL_MODES`. ``"behaviors"``
            leaves the roles to their markers, which is what a dense tree wants.
        edge_probabilities: Write each transition probability on its edge. Off by
            default: on a graph of any size the numbers collide, and the width
            already encodes them.
        label_font_pt: Size of the node labels. Smaller is the lever for a
            crowded figure whose names have nowhere left to go.
        label_wrap_pt: Width at which a label breaks onto another line, in
            points; ``0`` to leave every name on one line. This is what makes
            *shorten_labels* off a usable setting — "Grasp the milk carton" is
            worth more to a reader than "B4", and unwrapped it is five nodes
            wide.
        outcome_shares: Write ``P(reach)`` under each outcome, as the component
            does. Off leaves the outcomes named but unquantified.
        node_scale: Multiplies every radius. For a figure whose nodes are the
            subject rather than the structure between them.

    Save it through :func:`behavior_graph_figure_files`, or from inside
    :func:`behavior_graphs.viz.paper.paper_rc` — a bare ``savefig`` gets the
    process's own style and Type 3 fonts with it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    if pos is None:
        from behavior_graphs.viz.graph_plot import layered_layout

        pos = layered_layout(graph)
    kept = drawn_nodes(graph, pos, excluded)
    if not kept:
        return None

    unit = _unit_positions(pos, kept)
    radii = _radii_pt(graph, kept, node_scale=node_scale, symbols=symbols)
    edges = drawn_edges(
        graph,
        kept,
        min_edge_prob=min_edge_prob,
        min_edge_count=min_edge_count,
        prob_floor=prob_floor,
    )
    extents_pt: dict[int, float] = {}
    if label_mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}, not {label_mode!r}")  # noqa: TRY003
    total_episodes = _total_episodes(graph, kept)
    label_text = {
        nid: _label_for(
            graph,
            nid,
            given=labels.get(nid) if labels is not None else None,
            shorten=shorten_labels,
            label_mode=label_mode,
            symbols=symbols,
            share=_outcome_share(graph, nid, total_episodes) if outcome_shares else "",
        )
        for nid in kept
    }

    if height_in is None:
        height_in = default_height_in(pos, kept, width_in=width_in)
    fig = plt.figure(figsize=(width_in, height_in))
    # The axes fills the figure, and the room the drawing needs is taken out of
    # the *data* limits at the end (see _fit_limits) rather than reserved as a
    # margin here. A node's radius and its label are both measured in points, so
    # how much of the unit square they cover is not known until the figure has a
    # size — and a margin guessed before that either clips a label or leaves a
    # white band down the side of the figure.
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()

    for src, tgt, prob, count in edges:
        rare = 0 < count <= rare_edge_max_count
        weight = max(prob, RARE_PROB_FLOOR) if rare else prob
        # Thickness only. One colour for every edge, so the reader compares
        # lines against each other rather than against a remembered grey scale.
        style = {
            "linewidth": EDGE_BASE_PT + EDGE_SLOPE_PT * weight,
            "color": EDGE_COLOR,
        }
        if rare:
            style["linestyle"] = (0, RARE_DASHES)
        patch = FancyArrowPatch(
            unit[src],
            unit[tgt],
            connectionstyle=f"arc3,rad={EDGE_CURVE}",
            arrowstyle="-|>" if arrows else "-",
            # Points, so the arrow tip stops at the rim of a node whose radius
            # is also in points however the figure is scaled.
            shrinkA=radii[src] + 0.6,
            shrinkB=radii[tgt] + 0.6,
            mutation_scale=(ARROW_BASE_PT + ARROW_SLOPE_PT * weight) if arrows else 0.0,
            joinstyle="round",
            capstyle="round",
            # The axes is the whole figure and the outermost nodes sit on its
            # edge; clipping to it would slice them in half.
            clip_on=False,
            zorder=1,
            **style,
        )
        ax.add_patch(patch)
        if edge_probabilities:
            _draw_edge_label(ax, unit[src], unit[tgt], prob)

    label_artists: dict[int, object] = {}
    for nid in kept:
        role = _role_of(graph, nid, symbols)
        if greyscale:
            face = GREYSCALE_COLOR
        elif colors is not None and nid in colors:
            face = colors[nid]
        elif role is not None:
            face = ROLE_COLORS.get(role, BEHAVIOR_COLOR)
        else:
            face = BEHAVIOR_COLOR
        radius = radii[nid]
        # Scatter sizes are area in points squared, and a circle marker's path
        # is the unit circle — so the diameter is the square root of `s`, and a
        # radius in points stays a radius in points at any figure size.
        extents_pt[nid] = radius + 0.5
        ax.scatter(
            [unit[nid][0]],
            [unit[nid][1]],
            s=(2 * radius) ** 2,
            marker="o",
            facecolor=face,
            edgecolor=paper.INK,
            linewidths=0.4,
            clip_on=False,
            zorder=2,
            gid=f"node-{nid}",
        )
        if (glyph := _glyph_marker(role)) is not None:
            ax.scatter(
                [unit[nid][0]],
                [unit[nid][1]],
                s=(2 * radius * GLYPH_FRACTION) ** 2,
                marker=glyph,
                facecolor=_glyph_color(face),
                edgecolor="none",
                clip_on=False,
                zorder=3,
                gid=f"glyph-{nid}",
            )
        if label_text[nid]:
            label_artists[nid] = ax.annotate(
                label_text[nid],
                unit[nid],
                textcoords="offset points",
                xytext=(0.0, -(radius + LABEL_GAP_PT)),
                ha="center",
                va="top",
                fontsize=label_font_pt,
                # Tight: a label's lines are one label — a name wrapped, or a
                # name over the share that belongs to it — not separate things
                # to read. Loose enough that two lines of a name do not touch.
                linespacing=1.15,
                color=paper.INK,
                # Over the edges it lands on, and as a patch rather than a halo
                # — see the module docstring on what a path effect costs.
                bbox={"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none"},
                clip_on=False,
                zorder=5,
            )
    if label_wrap_pt > 0:
        _wrap_labels(fig, label_artists, label_wrap_pt)
    _place_and_fit(fig, ax, unit, extents_pt, label_artists)
    return fig


def _row_pitch(pos: Mapping[int, tuple[float, float]], kept: Sequence[int]) -> float:
    """The vertical step between two nodes that sit above each other, in layout
    units — the layout's own idea of one row.

    Taken from nodes sharing a column, because those are the ones that actually
    have to clear each other. Both layouts here step by 1 (the Markov layout's
    position within a BFS layer, the tree's depth), but reading it off the
    layout rather than assuming it means a layout that steps by something else
    is measured rather than mis-budgeted.
    """
    columns: dict[float, set[float]] = defaultdict(set)
    for nid in kept:
        x, y = pos[nid]
        columns[round(float(x), 6)].add(round(float(y), 6))
    gaps = [
        second - first
        for ys in columns.values()
        for first, second in zip(sorted(ys), sorted(ys)[1:], strict=False)
    ]
    if not gaps:
        # No column holds two nodes — a tree drawn one node per slot, say. Then
        # the levels themselves are the rows.
        ordered = sorted({round(float(pos[nid][1]), 6) for nid in kept})
        gaps = [second - first for first, second in zip(ordered, ordered[1:], strict=False)]
    return min(gaps) if gaps else 1.0


def default_height_in(
    pos: Mapping[int, tuple[float, float]], kept: Sequence[int], *, width_in: float
) -> float:
    """A height for this layout, in inches.

    From how many rows of nodes could stack on top of each other, because that
    is what has to fit: a row needs its nodes and a line of label, and nothing
    else about the drawing sets a height. The layouts feeding this both put
    meaning on that axis — the Markov layout's rows are the nodes sharing a BFS
    depth, the tree's are its depth — so a flat graph gets the wide, short band
    a paper figure wants and a deep tree gets room instead of a squashed one.

    Rows, *not* the count of distinct y coordinates, which is what this used to
    ask for and is a much larger number: the layered layout centres each layer,
    so a four-node layer sits at ±0.5 and ±1.5 while a three-node one sits at
    −1, 0, 1. Seven coordinates, four rows — and the four are what a reader
    sees, since two nodes half a row apart are in different columns and never
    meet. Budgeting for the coordinates made every figure of an uneven graph
    around 40% taller than the drawing in it.

    Capped at the width, since a figure taller than the column it goes in gets
    scaled down by the LaTeX, which is exactly what the fixed width is for.
    """
    span = max(float(pos[n][1]) for n in kept) - min(float(pos[n][1]) for n in kept)
    rows = round(span / _row_pitch(pos, kept)) + 1
    return float(min(max(ROW_PITCH_IN * rows + 0.2, MIN_HEIGHT_IN), width_in))


def suggested_label_mode(
    graph: BehaviorGraph,
    kept: Sequence[int],
    symbols: Mapping[int, str] | None = None,
) -> str:
    """Which of :data:`LABEL_MODES` this graph should start on.

    A Markov graph has one START and two outcomes, and naming them is worth the
    ink. A trajectory tree ends every path in an outcome, so the same setting
    writes the word thirty times under thirty markers that already say it. The
    rule is the count, not the kind of graph, so a folded tree that is down to
    three outcomes gets its names back.
    """
    roles = sum(1 for nid in kept if _role_of(graph, nid, symbols) is not None)
    return "behaviors" if roles > ROLE_LABEL_LIMIT else "all"


def _wrapped_lines(paragraph: str, *, fits, max_lines: int) -> list[str]:
    """*paragraph* broken into lines, greedily, on spaces only.

    On spaces because the alternative is hyphenating "Behavior" into "Behav-"
    in a figure caption's worth of space, and because a name with no spaces in
    it — a cluster id, a merged state's ``B12 #2`` — has no break a reader would
    recognise. *fits* answers whether a candidate line is narrow enough; the
    last line allowed takes whatever is left over, however wide.
    """
    words = paragraph.split()
    if not words:
        return [paragraph]
    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if fits(candidate) or len(lines) >= max_lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def _wrap_labels(fig, label_artists: Mapping[int, object], width_pt: float) -> None:
    """Break every label that is wider than *width_pt* onto more lines.

    Measured, not estimated: the component has to guess a label's width from a
    character count because it is laying out in a browser before the text
    exists, and this has the renderer and the label's own font in hand — so
    "too wide" here means the width the PDF will actually be written at.

    Runs before :func:`_place_and_fit`, which is the point: placement asks how
    much room each label needs, and this is what decides that.
    """
    renderer = fig.canvas.get_renderer()
    limit_px = width_pt * fig.dpi / 72.0
    for text in label_artists.values():
        font = text.get_fontproperties()

        def fits(candidate: str, font=font) -> bool:
            return renderer.get_text_width_height_descent(candidate, font, False)[0] <= limit_px

        text.set_text(
            "\n".join(
                line
                for paragraph in text.get_text().split("\n")
                for line in _wrapped_lines(paragraph, fits=fits, max_lines=LABEL_MAX_LINES)
            )
        )


def _place_and_fit(
    fig,
    ax,
    unit: Mapping[int, tuple[float, float]],
    extents_pt: Mapping[int, float],
    label_artists: Mapping[int, object],
) -> None:
    """Settle where the labels go and how much room the drawing gets.

    The two are one problem. Placement asks how far apart the nodes are *in
    points*, which depends on the data limits; fitting the limits asks how much
    of the axes the labels cover, which depends on where they were placed. So
    they run against each other for a fixed :data:`LAYOUT_PASSES` rounds and end
    on a fit, which is what guarantees nothing is clipped.
    """
    for _ in range(LAYOUT_PASSES):
        if label_artists:
            _place_labels(fig, ax, unit, extents_pt, label_artists)
        _fit_limits(fig, ax, unit, extents_pt)


def _label_box(
    where: str, center: tuple[float, float], gap: float, size: tuple[float, float]
) -> tuple[tuple[float, float, float, float], tuple[float, float], tuple[str, str]]:
    """One candidate placement: the box it occupies, its offset and alignment.

    All in points from the node's centre, so the caller can compare candidates
    against each other and against the nodes without touching the axes.
    """
    (cx, cy), (width, height) = center, size
    if where == "below":
        return (
            (cx - width / 2, cy - gap - height, cx + width / 2, cy - gap),
            (0.0, -gap),
            ("center", "top"),
        )
    if where == "above":
        return (
            (cx - width / 2, cy + gap, cx + width / 2, cy + gap + height),
            (0.0, gap),
            ("center", "bottom"),
        )
    if where == "right":
        return (
            (cx + gap, cy - height / 2, cx + gap + width, cy + height / 2),
            (gap, 0.0),
            ("left", "center"),
        )
    return (
        (cx - gap - width, cy - height / 2, cx - gap, cy + height / 2),
        (-gap, 0.0),
        ("right", "center"),
    )


def _overlap(
    one: tuple[float, float, float, float], other: tuple[float, float, float, float]
) -> float:
    """Area two boxes share, in points squared — 0 when they are clear."""
    width = min(one[2], other[2]) - max(one[0], other[0])
    height = min(one[3], other[3]) - max(one[1], other[1])
    return width * height if width > 0 and height > 0 else 0.0


def _place_labels(
    fig,
    ax,
    unit: Mapping[int, tuple[float, float]],
    extents_pt: Mapping[int, float],
    label_artists: Mapping[int, object],
) -> None:
    """Put each label on a side of its node that is free.

    Below if it can be, and otherwise the first of :data:`LABEL_POSITIONS` whose
    box touches neither a node nor a label already placed. If every side is
    taken, the least-overlapped one: a name slightly over a neighbour is still
    read, a name centred on one is not.

    Biggest node first, ties by id — so the nodes a reader looks at first get
    the position they expect, and the outcome does not depend on the order a
    dictionary happened to be built in. That fixed order is also what keeps the
    export byte-identical between two runs.
    """
    renderer = fig.canvas.get_renderer()
    to_points = 72.0 / fig.dpi
    centers = {
        nid: tuple(float(v) * to_points for v in ax.transData.transform(unit[nid])) for nid in unit
    }
    # Every node is an obstacle from the start; each label becomes one as it is
    # placed. Edges are not: a name crossing a thin line still reads, and a
    # figure that tried to dodge the edges too would have nowhere to put
    # anything on the graphs that need this most.
    taken = [
        (cx - extents_pt[nid], cy - extents_pt[nid], cx + extents_pt[nid], cy + extents_pt[nid])
        for nid, (cx, cy) in centers.items()
    ]
    for nid in sorted(label_artists, key=lambda n: (-extents_pt[n], n)):
        text = label_artists[nid]
        window = text.get_window_extent(renderer)
        size = (window.width * to_points, window.height * to_points)
        gap = extents_pt[nid] + LABEL_GAP_PT
        best = None
        for rank, where in enumerate(LABEL_POSITIONS):
            box, offset, align = _label_box(where, centers[nid], gap, size)
            cost = sum(_overlap(box, other) for other in taken)
            if best is None or (cost, rank) < best[:2]:
                best = (cost, rank, box, offset, align)
            if cost == 0.0:
                break
        _cost, _rank, box, offset, (ha, va) = best
        text.set_position(offset)
        text.set_ha(ha)
        text.set_va(va)
        taken.append(box)


def _fit_limits(
    fig,
    ax,
    unit: Mapping[int, tuple[float, float]],
    extents_pt: Mapping[int, float],
) -> None:
    """Widen the data limits until nothing the figure drew sticks out of it.

    Node markers and labels are sized in points, so how much of the axes they
    cover depends on the limits — which is what this is setting. One pass
    measures what overflows and gives it room; giving it room shrinks the
    drawing slightly, so :func:`_place_and_fit` calls this more than once.

    Cheaper than it looks, and the alternative is worse: ``bbox_inches="tight"``
    also fits the content, by changing the figure's width — and an exported
    width that is not the column width is the one thing this module exists to
    guarantee.

    The nodes are measured from *extents_pt* rather than from their artists: a
    ``PathCollection`` reports an infinite window extent until it has been
    drawn, and a marker's size is known here exactly anyway.
    """
    from matplotlib.transforms import Bbox

    renderer = fig.canvas.get_renderer()
    boxes = []
    for nid, (x, y) in unit.items():
        px, py = ax.transData.transform((x, y))
        half = extents_pt[nid] * fig.dpi / 72.0
        boxes.append(Bbox([[px - half, py - half], [px + half, py + half]]))
    boxes += [
        box
        for box in (artist.get_window_extent(renderer) for artist in (*ax.texts, *ax.patches))
        if box.width and box.height
    ]
    drawn = Bbox.union(boxes)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [[drawn.x0, drawn.y0], [drawn.x1, drawn.y1]]
    )
    pad_x = FIGURE_PAD_IN / fig.get_figwidth() * (x1 - x0)
    pad_y = FIGURE_PAD_IN / fig.get_figheight() * (y1 - y0)
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y0 - pad_y, y1 + pad_y)


def _draw_edge_label(ax, start: tuple[float, float], end: tuple[float, float], prob: float) -> None:
    """Write *prob* where the edge bows, not on the straight line between nodes.

    The point at the curve's apex, which is where the drawn edge actually is —
    a label at the chord midpoint sits beside its own edge, and next to a
    different one on a graph with two-way traffic.
    """
    mid = np.array([(start[0] + end[0]) / 2, (start[1] + end[1]) / 2])
    chord = np.array(end) - np.array(start)
    perp = np.array([-chord[1], chord[0]])
    apex = mid + 0.5 * EDGE_CURVE * perp
    import matplotlib.pyplot as plt

    ax.annotate(
        f"{prob:.2f}".lstrip("0") if prob < 1 else "1",
        apex,
        ha="center",
        va="center",
        # A step below the labels, since it annotates the structure rather than
        # naming a node.
        fontsize=plt.rcParams["xtick.labelsize"],
        color=paper.MUTED,
        # A white pill under the number so it is readable where it crosses its
        # own edge, without an alpha (see the module docstring).
        bbox={"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none"},
        zorder=4,
    )


def behavior_graph_figure_files(**kwargs) -> dict[str, bytes] | None:
    """``{"pdf": …, "png": …}`` for one graph, or ``None`` with nothing to draw.

    The PNG is the on-page preview and the PDF is the file that goes into the
    paper; both come off one figure, so what a reader of the page approves is
    what they download. The figure is closed either way — a Streamlit page that
    leaks one per rerun runs the process out of them.

    Takes the same arguments as :func:`behavior_graph_figure`.
    """
    import matplotlib.pyplot as plt

    with paper.paper_rc():
        fig = behavior_graph_figure(**kwargs)
        if fig is None:
            return None
        try:
            return {"pdf": paper.rendered(fig, "pdf", 300), "png": paper.rendered(fig, "png", 300)}
        finally:
            plt.close(fig)
