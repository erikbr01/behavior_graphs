"""Browser smoke test: the graph fits its frame, and labels share the width.

The graph is never drawn wider than the frame it is in — scrolling a graph
sideways costs the overview that is the reason to draw one. So the two things
that have to work inside a fixed width are wrapping (labels take a second line
instead of a wider slot) and allocation (each level's width is split by what its
labels need). Only measurable here: the widths are text metrics and the fitting
happens in the component's iframe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# pytest puts this file's directory on sys.path (tests/ is not a package).
from cdp import label_locator

pytestmark = pytest.mark.browser

APP = Path(__file__).parent / "apps" / "wide_labels_app.py"

#: Frame width, the width the graph draws at, whether anything scrolls, the most
#: lines any label wrapped onto, and the x-extent of the drawn nodes.
GEOMETRY = """
(() => {
  const frame = [...document.querySelectorAll('iframe')].find(f => {
    try { return f.contentDocument.querySelector('svg#graph') } catch (e) { return false }
  });
  if (!frame) return null;
  const doc = frame.contentDocument;
  const svg = doc.getElementById('graph');
  const box = (svg.getAttribute('viewBox') || '').split(/\\s+/);
  // Terminal labels always stack (outcome above share), so they say nothing
  // about whether a *name* needed a second line — which is what wrapping is for.
  const lines = [...doc.querySelectorAll('text.node-label:not(.node-label-terminal)')]
    .map(t => t.querySelectorAll('tspan').length);
  const circles = [...doc.querySelectorAll('circle.node-circle')]
    .map(c => parseFloat(c.getAttribute('cx') ?? '0'));
  return {
    frame: Math.round(svg.getBoundingClientRect().width),
    drawn: box.length === 4 ? Math.round(parseFloat(box[2])) : null,
    scrolls: doc.body.scrollWidth > doc.body.clientWidth,
    maxLines: lines.length ? Math.max(...lines) : 0,
    nodes: circles.length,
    widest: circles.length ? Math.round(Math.max(...circles)) : 0,
  };
})()
"""


@pytest.fixture()
def page(serve):
    # Deliberately narrow: this tree's labels need more room than a ~700px frame
    # has, which is the case the layout has to survive without scrolling.
    return serve(APP, width=1200)


@pytest.mark.parametrize("labels", ["long", "unbreakable", "short"])
def test_the_graph_never_outgrows_its_frame(page, labels):
    """Whatever the labels are, including names no wrapping can shorten."""
    if labels != "long":
        page.click(label_locator(labels))
        page.settle()

    geometry = page.evaluate(GEOMETRY)

    assert geometry["drawn"] == geometry["frame"], geometry
    assert not geometry["scrolls"], "nothing should scroll sideways"
    # And every node is inside the canvas, not pushed off the right edge.
    assert geometry["widest"] <= geometry["frame"], geometry


def test_long_labels_wrap_rather_than_asking_for_a_wider_slot(page):
    geometry = page.evaluate(GEOMETRY)

    assert geometry["maxLines"] > 1, "long names should wrap onto a second line"


def test_terse_labels_leave_the_layout_alone(page):
    """Same tree, short names: no name wraps and nothing is clipped."""
    page.click(label_locator("short"))
    page.settle()

    geometry = page.evaluate(GEOMETRY)

    assert geometry["maxLines"] == 1
    assert geometry["nodes"] > 0


def test_a_terminal_stacks_its_outcome_above_its_share(page):
    """Whatever the names are: they are the crowded levels' widest labels."""
    lines = page.evaluate(
        """
(() => {
  const frame = [...document.querySelectorAll('iframe')].find(f => {
    try { return f.contentDocument.querySelector('svg#graph') } catch (e) { return false }
  });
  if (!frame) return null;
  return [...frame.contentDocument.querySelectorAll('text.node-label-terminal')]
    .map(t => [...t.querySelectorAll('tspan')].map(s => s.textContent));
})()
"""
    )

    assert lines, "the tree should have terminal leaves"
    for outcome, share in lines:
        assert outcome in ("SUCCESS", "FAILURE", "END"), lines
        assert share.endswith("%"), lines
