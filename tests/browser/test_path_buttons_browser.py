"""Browser smoke test: a path button reads as a list, not a paragraph.

With VLM annotations on, every step in a path is a sentence, and five of them
joined by arrows on one line is what made this block unreadable. The fix is a
markdown hard break per step plus container-scoped CSS — neither of which
``AppTest`` can check, since it sees the label string and never renders it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# pytest puts this file's directory on sys.path (tests/ is not a package).
from cdp import Browser

pytestmark = pytest.mark.browser

APP = Path(__file__).parent / "apps" / "path_buttons_app.py"
BUTTONS = "document.querySelectorAll('[class*=st-key-probe_paths] button').length >= 2"


@pytest.fixture()
def page(serve):
    browser: Browser = serve(APP)
    browser.wait_for(BUTTONS, what="the path buttons to render")
    return browser


def _labels(page: Browser) -> list[dict]:
    return page.evaluate("""
      Array.from(document.querySelectorAll('[class*=st-key-probe_paths] button')).map(el => {
        const p = el.querySelector('p');
        return {text: p.innerText, breaks: el.querySelectorAll('br').length,
                align: getComputedStyle(p).textAlign};
      })
    """)


def test_each_step_gets_its_own_line(page):
    labels = _labels(page)
    assert labels, "no path buttons rendered"
    for label in labels:
        lines = [ln for ln in label["text"].split("\n") if ln.strip()]
        # Count line, then one line per step: a collapsed newline would leave one.
        assert len(lines) >= 3, label
        assert label["breaks"] == len(lines) - 1, label
        assert lines[0].startswith(("100%", "80%", "60%", "40%", "20%")), label
        # First step unprefixed, every later step arrow-led.
        assert not lines[1].startswith("→"), label
        assert all(ln.startswith("→") for ln in lines[2:]), label


def test_the_lines_start_at_the_same_edge(page):
    """A centred label puts every line at a different x, which is the whole
    reason a five-step path was unreadable in the first place."""
    for label in _labels(page):
        assert label["align"] == "left", label


def test_one_episode_is_not_1_episodes(page):
    for label in _labels(page):
        head = label["text"].split("\n")[0]
        assert "(1 episodes)" not in head, head
