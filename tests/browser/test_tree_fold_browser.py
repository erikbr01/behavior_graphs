"""Browser smoke test: a collapsed branch must not follow the graph's reload.

The reset drops the fold sets, but a badge click also leaves a *pending* toggle
in session state next to an "already handled" marker. Clearing the marker while
the pending value stays makes the tree re-apply that click against the graph it
has just switched to — a subtree hidden behind a "+" nobody clicked, quietly
leaving its episodes out of what the participant sees.

Only reachable through a real browser: the badge lives in the SVG component, so
``AppTest`` cannot click it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# pytest puts this file's directory on sys.path (tests/ is not a package).
from cdp import OPEN_BADGE_COUNT, Browser, label_locator, open_badge_locator, reported_value

pytestmark = pytest.mark.browser

APP = Path(__file__).parent / "apps" / "tree_fold_app.py"


@pytest.fixture()
def page(serve):
    browser = serve(APP)
    browser.wait_for(OPEN_BADGE_COUNT, what="the tree to draw its fold badges")
    return browser


def _collapsed(browser: Browser) -> str:
    reported = browser.evaluate(reported_value("COLLAPSED"))
    if reported is None:
        raise AssertionError(f"app never reported a collapsed count:\n{browser.text()[:400]}")
    return reported


def _wait_for_collapsed(browser: Browser, count: int, *, what: str) -> None:
    browser.wait_for(f"(() => document.body.innerText.includes('COLLAPSED={count}'))()", what=what)


def test_collapsing_a_branch_by_badge_is_recorded(page):
    assert _collapsed(page) == "0"
    open_before = page.evaluate(OPEN_BADGE_COUNT)
    assert open_before > 0

    page.click(open_badge_locator())

    _wait_for_collapsed(page, 1, what="the badge click to collapse one branch")
    # One fewer branch showing its children than before.
    assert page.evaluate(OPEN_BADGE_COUNT) < open_before


def test_a_collapsed_branch_does_not_follow_the_graph_to_a_new_clustering(page):
    page.click(open_badge_locator())
    _wait_for_collapsed(page, 1, what="a branch to be collapsed")
    # Let the badge click's own follow-up reruns drain before changing the
    # graph under it. Landing the two inside one another is a real (if narrow)
    # race in the app — a stale fragment rerun re-runs with the previous
    # graph's closure, so the identity never changes and the reset never fires
    # — but it is not what this test is here to measure.
    page.settle()

    page.click(label_locator("4"))

    _wait_for_collapsed(page, 0, what="the rebuilt tree to come up with nothing collapsed")
    assert _collapsed(page) == "0"
