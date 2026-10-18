"""The paper export must not emit Type 3 fonts, and must not resize the page.

`viz/paper.py` opens by explaining that `pdf.fonttype: 42` exists because a
figure with Type 3 fonts is rejected by IEEE PDF eXpress. The font type is
resolved when the figure is SAVED, not when it is drawn, and `export()` saved
after `paper_style()` had exited -- so every exported PDF carried exactly the
fonts the style was written to prevent.

The second assertion is the trap in the obvious fix: re-applying all of
PAPER_RC at save time also applies `savefig.bbox: "tight"`, which crops the
page to the ink and silently breaks the fixed column width.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from behavior_graphs.viz import paper_figures as pf  # noqa: E402


def _figure(width_in: float = 3.35, height_in: float = 2.0):
    with pf.paper_style():
        fig, ax = plt.subplots(figsize=(width_in, height_in))
        ax.plot([0, 1], [0, 1])
        ax.set_xlabel("progress")
    return fig


def test_the_exported_pdf_has_no_type3_fonts():
    fig = _figure()
    try:
        assert b"/Type3" not in pf.export(fig, "pdf")
    finally:
        plt.close(fig)


def test_the_page_box_is_the_figure_size_it_was_built_at():
    fig = _figure(width_in=3.35, height_in=2.0)
    try:
        blob = pf.export(fig, "pdf")
    finally:
        plt.close(fig)
    box = re.search(rb"/MediaBox\s*\[([^\]]*)\]", blob)
    assert box, "no MediaBox in the exported PDF"
    width_pt, height_pt = (float(v) for v in box.group(1).split()[2:4])
    assert abs(width_pt - 3.35 * 72) < 0.5, f"page cropped to {width_pt} pt"
    assert abs(height_pt - 2.0 * 72) < 0.5, f"page cropped to {height_pt} pt"
