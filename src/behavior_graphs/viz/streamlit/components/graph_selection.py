"""What a graph view has selected, and when that selection stops meaning anything.

A click on a node, an edge, or a path button is remembered in session state
under the view's ``key_prefix`` — the SVG component reads it back to grey out
everything else, and the video panels read it to decide what to show. That
state is keyed by the *view*, not by the graph, so it survives a reload of the
graph itself: change K (or any hyperparameter that rebuilds the graph) and the
new graph came up still greyed out around a node id belonging to the old one.

:func:`sync_graph_identity` is the fix — a view names the graph it is about to
draw, and when that name changes the stale selection is dropped. Identity is
deliberately the caller's call: for the trajectory tree, folding a branch away
changes which nodes exist without making it a different graph, so the tree
names itself by its labels rather than by its node set.
"""

from __future__ import annotations

import hashlib

import numpy as np
import streamlit as st

# Everything a click can leave behind, relative to a view's key_prefix. The
# fold sets are in here too: their keys are cluster-id paths, which mean
# something else entirely once the clustering changes.
#
# Deliberately NOT here: `graph_last_seq`, `toggle_last_processed` and
# `expand_last_processed`. Those are not selections — they are the
# already-handled markers for the SVG component's *persisted* click value,
# which it re-sends on every rerun. Clearing one makes the last click before
# the reset look new: the node the reset just dropped gets re-selected, or the
# last +/- badge click is re-applied through an id map that belongs to the
# graph we just left. Keeping them is what makes the reset stick.
_SELECTION_SUFFIXES = (
    "graph_selected",
    "graph_selected_edge",
    "highlighted_path",
    "path_ep_list",
    "path_label",
    "path_leaf_path",
    "path_synth_ids",
    "expanded_buckets",
    "manually_collapsed",
    "expanded_subclusters",
)


def reset_graph_selection(key_prefix: str) -> None:
    """Forget whatever ``key_prefix``'s graph view had selected."""
    for suffix in _SELECTION_SUFFIXES:
        st.session_state.pop(f"{key_prefix}_{suffix}", None)
    bump_render_token(key_prefix)


def bump_render_token(key_prefix: str) -> None:
    """Force the graph iframe to repaint on the next render.

    It keeps its own copy of the selection and only repaints when its args
    change, so anything that drops a selection from Python's side — a reset, a ✕
    on either panel — has to move this token or the iframe keeps showing what was
    just dismissed.
    """
    token_key = f"{key_prefix}_graph_render_token"
    st.session_state[token_key] = int(st.session_state.get(token_key, 0)) + 1


def sync_graph_identity(key_prefix: str, identity: str) -> None:
    """Record which graph ``key_prefix`` is drawing; reset if it changed.

    Idempotent within a run, so a view and something it delegates to can both
    call it with the same identity. The first call for a view never resets —
    there is nothing stale on the render that introduces the graph.
    """
    id_key = f"{key_prefix}_graph_identity"
    known = st.session_state.get(id_key)
    if known == identity:
        return
    st.session_state[id_key] = identity
    if known is None:
        return
    reset_graph_selection(key_prefix)


def labels_identity(labels: np.ndarray, metadata: list[dict]) -> str:
    """Identity for a view whose graph is derived from ``labels``.

    Reuses the episode index's content signature: a new clustering (new K, new
    run) gives a new signature, while pruning or folding — which change what is
    drawn but not the underlying labels — leaves it alone.
    """
    from behavior_graphs.viz.streamlit.episode_index import signature

    return signature(labels, metadata)


def graph_identity(labels: np.ndarray, metadata: list[dict], node_ids) -> str:
    """``labels_identity`` plus the graph's own node ids.

    For a graph handed to the view whole (the Markov graph, merged or not), the
    node set is part of its identity: state merging rebuilds ids from the same
    labels, and a selected id from the previous merge is exactly the stale
    selection this module exists to drop.
    """
    h = hashlib.blake2b(digest_size=8)
    for nid in sorted(node_ids):
        h.update(int(nid).to_bytes(8, "big", signed=True))
    return f"{labels_identity(labels, metadata)}|{h.hexdigest()}"
