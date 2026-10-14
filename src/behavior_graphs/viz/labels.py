"""Short forms of node names, for the text drawn *on* a graph.

A behavior graph's node names are written for prose — a video panel header, a
path chain in a caption, an inspector row — where "Behavior 12" reads better
than "B12". Inside the drawing they are the single biggest source of clutter:
every node carries one, the layout divides its horizontal room by how wide they
are (see ``_label_demand`` in the trajectory-tree view), and a tree of them
collides with itself long before the graph runs out of nodes.

So the drawn label is shortened, and only the drawn label: everything that
*reads* a name still gets the full one. What is dropped is only what carries no
information at all — the word "Behavior", which every circle in the graph is one
of, and the long tail of a rare-bucket summary, whose episode count is already in
the node's radius and its tooltip.

Nothing a reader might be looking for is dropped. The named nodes — START,
✓ SUCCESS, ✕ FAILURE, ■ END — keep their word even though their glyph repeats it:
where a run began and whether it ended well are the questions the graph is asked,
and they are worth the ink to answer without anyone having to learn a glyph
first.

A name that came from somewhere else — a VLM annotation, an experimenter's
``cluster_names`` — is left exactly as it is. It was written to be read, and
there is nothing in it we can claim is redundant.

The one thing done *to* such a name is numbering it when it repeats: see
:func:`number_repeats`. Two clusters described identically are still two
clusters, and a graph that draws them with the same words says the opposite.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

#: ``Behavior 12`` / ``Behavior 12 #2`` (a state-merging context copy). Digits
#: only: ``\S+`` also matched names someone wrote, turning "Behavior of the
#: gripper" into "Bof the gripper".
_BEHAVIOR = re.compile(r"^Behavior\s+(\d+)\b(.*)$")

#: ``+3 rare variants (7 eps)`` — see ``RARE_COLLAPSE_CLUSTER_ID``. The episode
#: count is in the node's radius and its tooltip; ``+3`` is what the label is for.
_RARE_BUCKET = re.compile(r"^(\+\d+)\s+rare variants?\b.*$")


def short_node_label(name: str) -> str:
    """``name`` as it should be drawn on the graph.

    >>> short_node_label("Behavior 12")
    'B12'
    >>> short_node_label("+3 rare variants (7 eps)")
    '+3'
    >>> short_node_label("START")
    'START'
    >>> short_node_label("FAILURE")
    'FAILURE'
    >>> short_node_label("Grasp the milk carton")
    'Grasp the milk carton'
    """
    text = str(name).strip()

    rare = _RARE_BUCKET.match(text)
    if rare:
        return rare.group(1)

    behavior = _BEHAVIOR.match(text)
    if behavior:
        # "Behavior 12 #2" -> "B12 #2": the copy suffix is the whole point of a
        # split state's name, so it survives.
        return f"B{behavior.group(1)}{behavior.group(2)}"

    return text


def number_repeats(names: Mapping[int, str]) -> dict[int, str]:
    """*names* with every repeated one numbered ``(1)``, ``(2)``, …

    A labeling model can describe two clusters in the same words — a run of
    ``set_table`` has two that are both "reaching for the red can" — and drawn
    that way the graph asserts something false: that one behavior appears twice
    where in fact two distinct clusters do, with their own episodes, their own
    transitions and their own outcomes. The number is there so a reader can see
    that they are two, and go and look at what separates them.

    *Every* member of a repeated group is numbered, not just the second one
    onward: leaving the first bare would make it read as the behavior and the
    rest as variants of it, and the reader could not tell an unnumbered name
    from one that never repeated.

    In cluster-id order, so the numbering is stable across reruns — and so a
    figure exported twice from one graph is the same figure.

    >>> number_repeats({5: "reaching for the can", 7: "reaching for the can"})
    {5: 'reaching for the can (1)', 7: 'reaching for the can (2)'}
    >>> number_repeats({0: "grasping the pumpkin", 1: "stalling"})
    {0: 'grasping the pumpkin', 1: 'stalling'}
    """
    by_name: dict[str, list[int]] = defaultdict(list)
    for node_id in sorted(names):
        by_name[names[node_id]].append(node_id)
    numbered = dict(names)
    for name, node_ids in by_name.items():
        if len(node_ids) < 2:
            continue
        for ordinal, node_id in enumerate(node_ids, start=1):
            numbered[node_id] = f"{name} ({ordinal})"
    return numbered


def figure_filename(rollout_dir: Path | str | None, run_id: str | None, view: str) -> str:
    """A name for a figure downloaded off this run: rollout, run, view.

    A paper's figure directory ends up holding one of these per task per view,
    and ``behavior_graph.pdf`` tells nobody which — worse, the browser silently
    numbers the second one rather than saying anything. All three parts earn
    their place: the rollout says which task and policy, the run id says which
    cell of the sweep drew it (so the figure in the paper can be traced back to
    the hyperparameters that produced it), and the view separates the two
    figures one page can export.

    >>> figure_filename("/data/rollouts/set_table_baseline", "17b0e615f5d5", "tree")
    'set_table_baseline_17b0e615f5d5_tree'
    >>> figure_filename(None, None, "markov")
    'behavior_graph_markov'
    """
    parts = [Path(rollout_dir).name if rollout_dir else "behavior_graph"]
    if run_id:
        parts.append(str(run_id))
    parts.append(view)
    return slug("_".join(str(part) for part in parts if part)) or "behavior_graph"


def slug(text: str) -> str:
    """*text* reduced to what is safe in a filename on any of the three OSes.

    Kept next to :func:`figure_filename` because both a graph and a comparison
    figure need it, and a second regex in a second module is how the two drift
    into naming the same thing differently.
    """
    return "_".join(
        re.sub(r"[^A-Za-z0-9._-]+", "-", part).strip("-") for part in str(text).split("_") if part
    )


def vlm_detail(vlm: dict | None, *, show_descriptions: bool = True) -> str | None:
    """The small print under a node's name: what the labeling model said about
    the node, and which model said it — or ``None`` when there is nothing to
    print.

    Two lines rather than one, because they answer different questions: the
    description is the claim, and ``gemini-3.5-flash · 95% confidence`` is where
    it came from and how sure it was.

    ``show_descriptions=False`` drops both. The user study passes that: reading
    a behavior out of the graph is the thing it measures, and a paragraph naming
    the behavior in words does a good part of that for the participant — with no
    equivalent in the rollout-video arm, so it is a difference between the arms
    rather than a help. The short label the graph draws is unaffected; that one
    is the node's name.

    >>> vlm_detail({"description": "Reaches for the handle.", "source": "m", "confidence": 0.5})
    'Reaches for the handle.  \\n_m · 50% confidence_'
    >>> vlm_detail({"description": "Reaches for the handle."}, show_descriptions=False) is None
    True
    """
    if not vlm or not show_descriptions:
        return None
    description = str(vlm.get("description", "")).strip()
    if not description:
        return None
    confidence = vlm.get("confidence")
    source = vlm.get("source", "VLM")
    provenance = (
        f"{source} · {confidence:.0%} confidence" if confidence is not None else str(source)
    )
    return f"{description}  \n_{provenance}_"
