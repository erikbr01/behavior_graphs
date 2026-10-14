"""The half of a paper-figure panel that is the same for every figure.

Two panels offer a figure now — the behavior graph
(:mod:`behavior_graphs.viz.streamlit.components.graph_figure_export`) and the
policy comparison
(:mod:`behavior_graphs.viz.streamlit.components.comparison_figure_export`) — and
what they have in common is the end of it: show the reader the PNG, then hand
them the PDF. The controls above differ, because a node-link diagram and a set
of violins have almost nothing to ask the same question about.

Shared so the preview sizing lives in one place. It looks like a detail and is
not: a portrait figure stretched to the panel's width is most of a screen, so
the reader scrolls past the thing they came to approve.
"""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

#: How tall a preview may get, in CSS pixels.
#:
#: Applied by putting a tall preview in a fixed-height scrolling container and
#: leaving the image at the container's width. Two earlier attempts were worse:
#: capping the height in CSS squashes the picture, because the image carries an
#: inline width from Streamlit that a stylesheet cannot outrank and a definite
#: width with a clamped height is the one combination that distorts; and giving
#: the image an explicit pixel width scales it correctly but turns off
#: Streamlit's ``useContainerWidth``, which is what its fullscreen affordance is
#: gated on — so clicking the preview to see it large stopped doing anything.
PREVIEW_MAX_HEIGHT_PX = 520

#: How wide the panel is taken to be when deciding whether a preview needs the
#: scrolling box, in CSS pixels. A nominal figure: the width is the browser's to
#: know and Python's to guess, and being wrong costs an unnecessary scroll box
#: on a figure that nearly fits, or a preview a little taller than the cap.
PREVIEW_PANEL_WIDTH_PX = 700


def preview_overflows(width_in: float, height_in: float) -> bool:
    """Would this figure, drawn at the panel's width, exceed the height cap?

    Only those get boxed. A wide figure — a row of violins, a flat graph — is
    nowhere near the cap and would otherwise sit in a fixed box with half of it
    empty.
    """
    if width_in <= 0 or height_in <= 0:
        return False
    return PREVIEW_PANEL_WIDTH_PX * height_in / width_in > PREVIEW_MAX_HEIGHT_PX


def paper_figure_downloads(
    files: Mapping[str, bytes],
    *,
    key: str,
    filename: str,
    width_in: float,
    height_in: float,
) -> None:
    """Preview *files* and offer them, one button each.

    The PNG is a preview of the PDF, not a second rendering: both come off one
    figure, so what is approved here is what is downloaded.
    """
    # At the container's width either way, so the image is never distorted and
    # Streamlit still offers the fullscreen view a reader clicks the preview
    # for. Only a preview that would run past the cap is boxed, and then it
    # scrolls rather than being shrunk.
    if preview_overflows(width_in, height_in):
        with st.container(height=PREVIEW_MAX_HEIGHT_PX, border=False):
            st.image(files["png"], width="stretch")
    else:
        st.image(files["png"], width="stretch")
    pdf_col, png_col = st.columns(2)
    pdf_col.download_button(
        "⬇ PDF (for the paper)",
        data=files["pdf"],
        file_name=f"{filename}.pdf",
        mime="application/pdf",
        width="stretch",
        key=f"{key}_pdf",
    )
    png_col.download_button(
        "⬇ PNG (for a slide)",
        data=files["png"],
        file_name=f"{filename}.png",
        mime="image/png",
        width="stretch",
        key=f"{key}_png",
    )
