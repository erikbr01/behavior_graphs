"""What the panel under a clicked node says about that node.

The labeling model writes two things per node: a short label, which is the name
the graph draws, and a paragraph describing it, which the panel printed along
with the model and its confidence. The study hides the paragraph — reading a
behavior out of the graph is the thing being measured, and the rollout-video arm
has no equivalent sentence — so it is a flag, and the demo keeps it.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import behavior_graphs
from behavior_graphs.viz.labels import vlm_detail

ENTRY = {
    "label": "reaching for the handle",
    "description": "The arm crosses the table and closes on the handle.",
    "source": "gemini-3.5-flash",
    "confidence": 0.95,
}


def test_the_detail_names_the_model_and_how_sure_it_was():
    detail = vlm_detail(ENTRY)

    assert ENTRY["description"] in detail
    assert "gemini-3.5-flash · 95% confidence" in detail


def test_the_study_drops_the_paragraph_and_its_provenance():
    assert vlm_detail(ENTRY, show_descriptions=False) is None


def test_a_node_with_only_a_label_has_no_detail():
    """The label is the node's name and the graph draws it; this is the rest."""
    assert vlm_detail({"label": "reaching for the handle"}) is None
    assert vlm_detail({"description": "   "}) is None
    assert vlm_detail(None) is None


def test_an_unscored_annotation_still_says_who_wrote_it():
    detail = vlm_detail({"description": "Something happened.", "source": "a person"})

    assert detail.endswith("_a person_")
    assert "confidence" not in detail


def test_the_study_pages_ask_for_the_paragraph_to_be_hidden():
    """The flag defaults to on, so a page that wants it off has to say so.

    Both study renderers do, and both read it from the study config — so a
    config can turn it back on for a pilot without a code change. One of them
    reaches it through `tree_kwargs_from_config`, the shared translation of a
    session's ``graph:`` block, which every variant since the order-effect
    mockup draws its graph through; the other still spells it out. Read the
    older one off the file rather than the imported module: importing a study
    page runs its page-level Streamlit setup.
    """
    from behavior_graphs.viz.streamlit.components.graph_config import tree_kwargs_from_config
    from behavior_graphs.viz.streamlit.components.trajectory_tree_view import (
        render_trajectory_tree,
    )

    assert (
        inspect.signature(render_trajectory_tree).parameters["show_vlm_descriptions"].default
        is True
    )

    assert tree_kwargs_from_config({})["show_vlm_descriptions"] is False
    # And a config that wants it back for a pilot gets it, without a code change.
    assert tree_kwargs_from_config({"show_vlm_descriptions": True})["show_vlm_descriptions"] is True

    evidence = Path(behavior_graphs.__file__).parent / "viz" / "streamlit" / "user_studies"
    assert "tree_kwargs_from_config(graph_params)" in (evidence / "evidence.py").read_text()
