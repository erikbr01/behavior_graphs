"""The style a figure has to be in to go into the paper, in one place.

Typography, colour and — the part that is not cosmetic — how the file is
written. Two figure families draw through this: the study's answer panels
(:mod:`behavior_graphs.viz.streamlit.user_study.figure_export`) and the behavior
graph itself (:mod:`behavior_graphs.viz.graph_figure`). They share it because the
requirements are the paper's, not the figure's: 8 pt type at the size it will be
printed, TrueType outlines, and the same serif as the template.

Three things here are load-bearing rather than taste:

* **``pdf.fonttype: 42``** embeds TrueType outlines. A figure with Type 3 fonts
  is what IEEE PDF eXpress bounces a submission over, and the fallback is
  silent — which is why the font list below is TrueType faces in preference
  order rather than whichever serif the machine has. Given a CFF-flavoured
  OpenType face (Nimbus Roman, the ghostscript one) the PDF backend quietly
  writes Type 3.
* **Width in inches, not a scale factor.** A figure exported at exactly the
  column width goes into the LaTeX at ``width=\\textwidth`` with no scaling,
  which is the only way 8 pt type in the figure comes out as 8 pt on the page.
  Anything else silently rescales the type.
* **No creation date.** :func:`rendered` strips it, so the same input exports
  byte-identical bytes and a figure that has not changed does not show up as a
  diff in the paper repo.

The style is applied through an ``rc_context`` and never set globally: this is
imported into a Streamlit process that draws other matplotlib figures, and
restyling those from here would be a surprise.
"""

from __future__ import annotations

import io
from collections.abc import Mapping

#: IEEE two-column text width, in inches (7.16 in / 21.6 pc) — a ``figure*``.
DOUBLE_COLUMN_WIDTH_IN = 7.16
#: One column of the same template (3.5 in), for a figure that fits beside text.
SINGLE_COLUMN_WIDTH_IN = 3.5

#: Near-black rather than black: at 8 pt, pure black type on white sparkles.
INK = "#1a1a1a"
#: For structure that should be read after the data — axes, ticks, thin rules.
MUTED = "#5c5c5c"

#: Typography and file format. Everything every paper figure needs and nothing
#: about axes, so a node-link diagram with no axes at all can use it unchanged.
BASE_RC: dict[str, object] = {
    "font.family": "serif",
    # Times, then the two TrueType fonts metric-compatible with it — Tinos
    # (croscore) and Liberation Serif — so the type matches the template's on a
    # machine with no Times licence. See the fonttype note above for why the
    # order matters.
    "font.serif": [
        "Times New Roman",
        "Tinos",
        "Liberation Serif",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    # 8 pt body, 7 pt on the axes: IEEE's own caption size is the floor a
    # figure's smallest text should sit at.
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "text.color": INK,
    "axes.labelcolor": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

#: The rest of what a figure *with axes* wants: light spines, a y grid under the
#: data, short ticks. Kept separate from :data:`BASE_RC` because a graph drawing
#: has no axes to style, and switching the grid on for one would draw it.
CHART_RC: dict[str, object] = {
    "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#dcdcdc",
    "grid.linewidth": 0.4,
}


def paper_rc(*overrides: Mapping[str, object]):
    """The paper style, as a context manager to draw *and save* inside.

    Both, not just the drawing: a font family and ``pdf.fonttype`` are resolved
    when the figure is rasterised or written, not when the artists are made, so
    a figure built in this context and saved outside it comes out in the default
    sans-serif and — worse — with Type 3 fonts.

    *overrides* are applied over :data:`BASE_RC` in order; pass
    :data:`CHART_RC` for a figure with axes.
    """
    import matplotlib.pyplot as plt

    rc: dict[str, object] = dict(BASE_RC)
    for override in overrides:
        rc.update(override)
    return plt.rc_context(rc)


def rendered(fig, fmt: str, dpi: int) -> bytes:
    """*fig* as bytes, with no timestamp in it.

    Call inside :func:`paper_rc` — see the note there about when a font is
    resolved.
    """
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format=fmt,
        dpi=dpi,
        # No creation date, so identical input exports identical bytes.
        metadata={"CreationDate": None} if fmt in ("pdf", "ps", "eps") else None,
    )
    return buffer.getvalue()
