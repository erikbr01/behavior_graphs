"""A study session's ``graph:`` block as :func:`render_trajectory_tree` kwargs.

The study configures its graph from YAML (``configs/viz/study/<session>.yaml``
merged under the variant's shared defaults), and every page that draws *that*
graph has to apply the same block the same way — a page that reads
``min_branch`` but forgets ``show_fold_toggles`` is showing a different display
from the one the participants were shown, which is exactly the comparison the
study rests on.

Only the settings that come out of config live here. The data a call needs
(labels, metadata, clips, node names) is the caller's business.
"""

from __future__ import annotations

from typing import Any

from behavior_graphs.viz.streamlit.components.panel_layout import panel_layout_from_config


def tree_kwargs_from_config(graph_params: dict) -> dict[str, Any]:
    """The config-derived keyword arguments for :func:`render_trajectory_tree`.

    Defaults are the interactive ones — a missing key means "as the component
    would have it" — except ``show_stats``, which is off: the path list and the
    "N nodes pruned" note under it are both about how the graph was built, which
    is the experimenter's business rather than the reader's.
    """
    return {
        "view_mode": graph_params.get("view_mode", "native_svg"),
        "min_branch": graph_params.get("min_branch", 2),
        "color_mode": graph_params.get("color_mode", "outcome"),
        "show_paths": graph_params.get("show_paths", True),
        "show_fold_toggles": graph_params.get("show_fold_toggles", True),
        "show_stats": graph_params.get("show_stats", False),
        "show_annotations": graph_params.get("show_behavior_annotations", True),
        "show_camera_labels": graph_params.get("show_camera_labels", True),
        # Off unless a config asks for it: the description is a paragraph the
        # labeling model wrote about the node, and reading a behavior out of the
        # graph is what the study is measuring. The provenance line under it
        # ("gemini-3.5-flash · 95% confidence") says the same thing about who
        # wrote it.
        "show_vlm_descriptions": graph_params.get("show_vlm_descriptions", False),
        "height_extra": int(graph_params.get("graph_height_extra", 0)),
        "panel_layout": panel_layout_from_config(graph_params),
    }
