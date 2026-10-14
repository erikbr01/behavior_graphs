"""Browser smoke test: the graph fills the width the layout gives it.

Moving the video panel from beside the graph to below it widens the graph's
container without changing a single component arg, so Streamlit resizes the
iframe and sends no render message. The SVG used to keep drawing at whatever
width it was first measured at — a two-thirds-wide graph in a full-width frame.
Invisible to AppTest, which never lays the component out at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# pytest puts this file's directory on sys.path (tests/ is not a package).
from cdp import GRAPH_DOC, Browser, label_locator

pytestmark = pytest.mark.browser

APP = Path(__file__).parent / "apps" / "panel_layout_app.py"
#: The same switch around a trajectory tree, which renders its own video panel.
TREE_APP = Path(__file__).parent / "apps" / "tree_layout_app.py"

#: The width the SVG is actually drawing at, from its viewBox.
DRAWN_WIDTH = f"""
(() => {{
  const d = {GRAPH_DOC};
  if (!d) return null;
  const svg = d.getElementById('graph');
  if (!svg) return null;
  const box = (svg.getAttribute('viewBox') || '').split(/\\s+/);
  return box.length === 4 ? Math.round(parseFloat(box[2])) : null;
}})()
"""

#: The width of the iframe the component is drawn in.
FRAME_WIDTH = f"""
(() => {{
  const d = {GRAPH_DOC};
  const frame = [...document.querySelectorAll('iframe')].find(f => {{
    try {{ return f.contentDocument === d }} catch (e) {{ return false }}
  }});
  return frame ? Math.round(frame.getBoundingClientRect().width) : null;
}})()
"""


#: The width of the page's content column, inside its padding — what a
#: "full width" element can actually occupy.
CONTENT_WIDTH = """
(() => {
  const el = document.querySelector('[data-testid="stMainBlockContainer"]')
    || document.querySelector('.stMainBlockContainer')
    || document.querySelector('[data-testid="block-container"]');
  if (!el) return null;
  const style = getComputedStyle(el);
  const pad = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  return Math.round(el.getBoundingClientRect().width - pad);
})()
"""


@pytest.fixture()
def page(serve):
    return serve(APP)


@pytest.fixture()
def tree_page(serve):
    return serve(TREE_APP)


def _pick_layout(page: Browser, label: str) -> None:
    page.click(label_locator(label))
    page.settle()


def test_graph_takes_the_full_width_when_the_panel_moves_below_it(page):
    beside = page.evaluate(DRAWN_WIDTH)
    assert beside == page.evaluate(FRAME_WIDTH), "the SVG should fill its frame"

    _pick_layout(page, "Below graph")

    below = page.evaluate(DRAWN_WIDTH)
    # The panel is no longer taking a third of the row, so the graph is wider —
    # and still exactly as wide as the frame it is in.
    assert below > beside * 1.2, f"expected a wider graph below, got {below} vs {beside}"
    assert below == page.evaluate(FRAME_WIDTH)

    _pick_layout(page, "Right of graph")

    # And it gives the width back rather than keeping it and overflowing.
    assert page.evaluate(DRAWN_WIDTH) == beside
    assert page.evaluate(DRAWN_WIDTH) == page.evaluate(FRAME_WIDTH)


def test_the_tree_takes_the_whole_row_when_its_panel_is_below(tree_page):
    """The tree renders its own panel, so it must not be split twice.

    It makes the layout's two slots itself and suppresses the graph renderer's
    panel; a second split inside would leave the tree in two thirds of its slot
    with an empty third beside it, in both layouts.
    """
    content = tree_page.evaluate(CONTENT_WIDTH)
    beside = tree_page.evaluate(FRAME_WIDTH)
    # Beside: the tree gets the two-thirds column, and all of it.
    assert 0.6 * content < beside < 0.72 * content, f"{beside} of {content}"

    _pick_layout(tree_page, "Below graph")

    below = tree_page.evaluate(FRAME_WIDTH)
    assert below >= 0.98 * content, f"expected the full row, got {below} of {content}"
    assert tree_page.evaluate(DRAWN_WIDTH) == below, "and the SVG should fill it"


def test_graph_follows_a_window_resize(page):
    """The other way the graph's width changes.

    Not the same path as the layout switch: no script rerun, no new args.
    Streamlit re-renders its components on a viewport change, which is what
    carries the new width in — worth pinning, because the component measures
    itself from `clientWidth` at render time and nothing else re-measures it.
    """
    wide = page.evaluate(DRAWN_WIDTH)
    assert wide == page.evaluate(FRAME_WIDTH)

    page.resize_viewport(900, 1000)
    page.wait_for(
        f"(() => {DRAWN_WIDTH.strip()} < {wide - 40})()",
        what="the graph to redraw narrower",
    )
    narrow = page.evaluate(DRAWN_WIDTH)
    assert narrow == page.evaluate(FRAME_WIDTH), "the SVG should still fill its frame"

    page.resize_viewport(1600, 1000)
    page.wait_for(
        f"(() => {DRAWN_WIDTH.strip()} > {narrow + 40})()",
        what="the graph to redraw wider again",
    )
    assert page.evaluate(DRAWN_WIDTH) == page.evaluate(FRAME_WIDTH)
