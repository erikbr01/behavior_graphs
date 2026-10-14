"""Browser smoke test: a real click on the graph, then a real graph reload.

Marked ``browser`` and excluded from the default run (see pyproject addopts);
run it with ``pytest -m browser``. It needs Chrome, ``requests`` and
``websocket-client``, and skips cleanly without them.

Why it exists: ``AppTest`` does not execute custom components, so the SVG
graph's click protocol — including the click value the component re-sends on
every rerun — is invisible to every other test in this repo. A reset that
cleared the dedup guard against that replay passed the whole unit suite while
re-selecting the node it had just dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# pytest puts this file's directory on sys.path (tests/ is not a package).
from cdp import (
    DIMMED_COUNT,
    SELECTED_COUNT,
    Browser,
    label_locator,
    node_locator,
    reported_value,
)

pytestmark = pytest.mark.browser

APP = Path(__file__).parent / "apps" / "graph_selection_app.py"
PROMPT = "Click a node or edge in the graph"


@pytest.fixture()
def page(serve):
    """The smoke-test app, served and open in a headless browser."""
    return serve(APP)


def _selected(browser: Browser) -> str:
    """The app's own report of what it has selected."""
    reported = browser.evaluate(reported_value("SELECTED"))
    if reported is None:
        raise AssertionError(f"app never reported a selection line:\n{browser.text()[:400]}")
    return reported


def test_clicking_a_node_selects_it_and_dims_the_rest(page):
    assert page.evaluate(SELECTED_COUNT) == 0
    assert page.evaluate(DIMMED_COUNT) == 0
    assert PROMPT in page.text()

    page.click(node_locator(1))

    assert page.wait_for(
        "(() => document.body.innerText.includes('SELECTED=1'))()",
        what="the app to register the click on node 1",
    )
    # The halo is on, unrelated nodes are dimmed, and the panel replaced the
    # prompt — i.e. the component and the Python side agree.
    assert page.evaluate(SELECTED_COUNT) == 1
    assert page.evaluate(DIMMED_COUNT) > 0
    assert PROMPT not in page.text()


def test_reloading_the_graph_clears_the_selection_for_real(page):
    """The regression an AppTest cannot see.

    After the reset drops the selection, the component still holds the click it
    reported and re-sends it on the next rerun. If the dedup marker went with
    the selection, that replay lands as a fresh click and re-selects the node —
    node 1 exists under both K values here precisely so that would succeed.
    """
    page.click(node_locator(1))
    page.wait_for(
        "(() => document.body.innerText.includes('SELECTED=1'))()",
        what="node 1 to be selected",
    )

    page.click(label_locator("4"))

    page.wait_for(
        "(() => document.body.innerText.includes('SELECTED=None'))()",
        what="the rebuilt graph to come up with nothing selected",
    )
    # And it stays dropped: no halo, nothing dimmed, prompt back.
    assert page.evaluate(SELECTED_COUNT) == 0
    assert page.evaluate(DIMMED_COUNT) == 0
    assert PROMPT in page.text()
    assert _selected(page) == "None"
