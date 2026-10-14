"""Browser smoke test: the graph legend is a drawn figure, and it is inert."""

from __future__ import annotations

from pathlib import Path

import pytest

# pytest puts this file's directory on sys.path (tests/ is not a package).
from cdp import Browser

pytestmark = pytest.mark.browser

APP = Path(__file__).parent / "apps" / "graph_intro_app.py"
FIGURE = """
!!Array.from(document.querySelectorAll('iframe'))
   .find(f => f.contentDocument && f.contentDocument.querySelector('g.annotations'))
"""


@pytest.fixture()
def page(serve):
    browser: Browser = serve(APP)
    browser.wait_for(FIGURE, what="the annotated figure")
    return browser


def _figure(page: Browser) -> dict:
    return page.evaluate("""(() => {
      const f = Array.from(document.querySelectorAll('iframe'))
        .find(x => x.contentDocument && x.contentDocument.querySelector('g.annotations'));
      const d = f.contentDocument, svg = d.getElementById('graph');
      const ann = svg.querySelector('g.annotations');
      const box = svg.getBoundingClientRect();
      const labels = Array.from(ann.querySelectorAll('text')).map(t => {
        const r = t.getBoundingClientRect();
        return {text: t.textContent, left: r.left - box.left, right: r.right - box.left,
                top: r.top - box.top, bottom: r.bottom - box.top};
      });
      const radii = Array.from(svg.querySelectorAll('g.nodes circle'))
        .map(c => +c.getAttribute('r'));
      const instructions = d.getElementById('instructions');
      return {labels, radii, lines: ann.querySelectorAll('line').length,
              width: box.width, height: box.height,
              instructions_hidden: getComputedStyle(instructions).display === 'none',
              draggable: svg.querySelector('g.nodes > g') ?
                getComputedStyle(svg.querySelector('g.nodes > g')).cursor : '?'};
    })()""")


def test_every_channel_the_drawing_uses_gets_a_callout(page):
    """Five channels — two circle sizes, band width, green, red — and since the
    bullet list above the figure is gone, the figure is where they are named."""
    fig = _figure(page)
    said = " | ".join(label["text"] for label in fig["labels"])

    for phrase in ("large", "small", "wide band", "slim band", "green", "red"):
        assert phrase in said, (phrase, said)
    assert fig["lines"] == len(fig["labels"])


def test_the_callouts_fit_inside_the_canvas(page):
    """Reserved padding, not luck: a clipped callout is worse than none."""
    fig = _figure(page)

    for label in fig["labels"]:
        assert label["left"] >= -1, label
        assert label["right"] <= fig["width"] + 1, label
        assert label["top"] >= -1 and label["bottom"] <= fig["height"] + 1, label


def test_the_sizes_it_talks_about_are_actually_different(page):
    """ "Large circle" and "small circle" have to be visibly large and small."""
    radii = sorted(fig_radii := _figure(page)["radii"], reverse=True)
    assert len(fig_radii) >= 3
    assert radii[0] >= 2 * radii[-1], radii


def test_the_nodes_are_named_in_words(page):
    """ "B2" is the real graph's shorthand, for a graph with no room for words.

    The figure has the room, and naming what a node is happens to be its whole
    subject.
    """
    drawn = (
        page.evaluate("""
      Array.from(document.querySelectorAll('iframe'))
        .find(f => f.contentDocument && f.contentDocument.querySelector('g.annotations'))
        .contentDocument.querySelectorAll('g.labels text')
    """)
        if False
        else page.evaluate("""(() => {
      const f = Array.from(document.querySelectorAll('iframe'))
        .find(x => x.contentDocument && x.contentDocument.querySelector('g.annotations'));
      return Array.from(f.contentDocument.querySelectorAll('g.labels text'))
        .map(t => t.textContent).join(' | ');
    })()""")
    )

    assert "Behavior 1" in drawn and "Behavior 2" in drawn and "Behavior 3" in drawn, drawn
    assert "B1" not in drawn.replace("Behavior 1", ""), drawn


def test_a_callout_is_offset_from_what_it_points_at(page):
    """Straight above a node, a label reads as that node's name."""
    leaders = page.evaluate("""(() => {
      const f = Array.from(document.querySelectorAll('iframe'))
        .find(x => x.contentDocument && x.contentDocument.querySelector('g.annotations'));
      return Array.from(f.contentDocument.querySelectorAll('g.annotations line'))
        .map(l => Math.abs(+l.getAttribute('x2') - +l.getAttribute('x1')));
    })()""")

    assert leaders, "no leader lines"
    assert all(dx > 8 for dx in leaders), leaders


def test_a_band_callout_starts_on_the_band(page):
    """The band is drawn between the nodes' *rims* and then arcs, so a midpoint
    computed from the node centres lands beside it — by tens of pixels on the
    long edges, which are the ones worth a callout."""
    gaps = page.evaluate("""(() => {
      const f = Array.from(document.querySelectorAll('iframe'))
        .find(x => x.contentDocument && x.contentDocument.querySelector('g.annotations'));
      const svg = f.contentDocument.getElementById('graph');
      const paths = Array.from(svg.querySelectorAll('path.edge'));
      return Array.from(svg.querySelectorAll('g.annotations line')).map(l => {
        const x = +l.getAttribute('x1'), y = +l.getAttribute('y1');
        let best = 1e9;
        for (const p of paths) {
          const total = p.getTotalLength();
          for (let t = 0; t <= total; t += 2) {
            const q = p.getPointAtLength(t);
            best = Math.min(best, Math.hypot(q.x - x, q.y - y));
          }
        }
        return best;
      });
    })()""")

    # Two of the four callouts describe an edge; those two start on a band. The
    # other two start on a node's rim, which is nowhere near one.
    assert sum(1 for gap in gaps if gap < 2) == 2, gaps


def test_the_figure_is_not_a_control(page):
    fig = _figure(page)

    assert fig["instructions_hidden"]
    assert fig["draggable"] == "default"


def _boxes(page: Browser) -> dict:
    """Every drawn thing that carries ink, in canvas coordinates."""
    return page.evaluate("""(() => {
      const f = Array.from(document.querySelectorAll('iframe'))
        .find(x => x.contentDocument && x.contentDocument.querySelector('g.annotations'));
      const svg = f.contentDocument.getElementById('graph');
      const box = svg.getBoundingClientRect();
      const rect = (el, kind) => {
        const r = el.getBoundingClientRect();
        return {kind, text: (el.textContent || '').trim(),
                left: r.left - box.left, right: r.right - box.left,
                top: r.top - box.top, bottom: r.bottom - box.top};
      };
      return {
        labels: Array.from(svg.querySelectorAll('g.labels > g')).map(el => rect(el, 'label')),
        callouts: Array.from(svg.querySelectorAll('g.annotations text')).map(el => rect(el, 'callout')),
        markers: Array.from(svg.querySelectorAll('g.nodes > g')).map(el => rect(el, 'marker')),
      };
    })()""")


def _overlap(a: dict, b: dict) -> bool:
    return not (
        a["right"] <= b["left"]
        or b["right"] <= a["left"]
        or a["bottom"] <= b["top"]
        or b["bottom"] <= a["top"]
    )


# 900 as well as the default: the figure is drawn at three-quarters of the page,
# so a 1200px window gives it about this much, and the widths where the writing
# collides are the narrow ones. The callouts used to sit right of the branch
# nodes, where at this width they ran into the terminals' captions.
@pytest.mark.parametrize("width", [900, 1600])
def test_no_two_pieces_of_writing_land_on_each_other(serve, width):
    page: Browser = serve(APP, width=width)
    page.wait_for(FIGURE, what="the annotated figure")
    drawn = _boxes(page)

    writing = drawn["labels"] + drawn["callouts"]
    assert len(drawn["labels"]) == 8, [label["text"] for label in drawn["labels"]]
    for i, one in enumerate(writing):
        for other in writing[i + 1 :]:
            assert not _overlap(one, other), (one, other)
    # A caption sits under its own marker and the glyph's box reaches into it;
    # a *callout* has no such excuse.
    for callout in drawn["callouts"]:
        for marker in drawn["markers"]:
            assert not _overlap(callout, marker), (callout, marker)


def test_a_leaf_caption_leaves_room_for_the_leaf_below_it(page):
    """Four leaves, each writing SUCCESS / FAILURE and a share under itself.

    The gap between two captions is where the lower leaf's ✓ or ✕ is drawn, so
    it has to hold a marker and not just avoid the text: at the canvas height
    two leaves needed, 17px of it went to a ✕ that is 24px across.
    """
    captions = sorted(
        (
            label
            for label in _boxes(page)["labels"]
            if label["text"].startswith(("SUCCESS", "FAILURE"))
        ),
        key=lambda label: label["top"],
    )

    assert len(captions) == 4, captions
    for above, below in zip(captions, captions[1:], strict=False):
        assert below["top"] - above["bottom"] >= 30, (above, below)
