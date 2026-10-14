"""Shared browser-test fixtures: one Chrome check, one way to serve an app.

Each test module used to carry its own byte-identical copy of both, so a change
to how Chrome is launched or how readiness is decided was a four-file edit — and
the copies had already drifted in whether they called ``settle()``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

# pytest puts this directory on sys.path (tests/ is not a package).
from cdp import NODE_COUNT, Browser, StreamlitApp, chrome_binary


@pytest.fixture(scope="module")
def browser_binary() -> str:
    pytest.importorskip("requests")
    pytest.importorskip("websocket")
    binary = chrome_binary()
    if binary is None:
        pytest.skip("no Chrome/Chromium on PATH")
    return binary


@pytest.fixture()
def serve(browser_binary) -> Iterator[Callable[..., Browser]]:
    """``serve(script, width=…)`` → a browser with that app drawn and settled.

    Everything is torn down when the test ends, in the order it was started.
    Waits for the graph's nodes and then for the reruns behind them to drain: a
    test that acts on the first quiet moment can land its click between a
    fragment's reruns.
    """
    started: list = []

    def _serve(script: Path, *, width: int = 1600) -> Browser:
        app = StreamlitApp(script)
        started.append(app)
        app.wait_until_serving()
        browser = Browser(browser_binary, width=width)
        started.append(browser)
        browser.goto(app.url)
        browser.wait_for(NODE_COUNT, what="the graph component to draw its nodes")
        browser.settle()
        return browser

    try:
        yield _serve
    finally:
        for resource in reversed(started):
            resource.close()
