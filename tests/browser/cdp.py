"""Drive a Streamlit app in headless Chrome over CDP.

``AppTest`` runs the script but not custom components, so everything the SVG
behavior graph does — the click protocol, the persisted click value it re-sends
on every rerun, the halo and the dimming — is invisible to it. That blind spot
hid a real bug: the selection reset cleared the very dedup marker that stops the
component's persisted click from replaying, and every unit test still passed.

This is the smallest thing that can see it: a real browser, the real iframe, one
real click. Kept deliberately small — a page, an element, a click, an assertion —
rather than growing into a framework.
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CHROME_BINARIES = ("google-chrome", "chromium", "chromium-browser")
_REPO_ROOT = Path(__file__).resolve().parents[2]


def chrome_binary() -> str | None:
    for name in CHROME_BINARIES:
        found = shutil.which(name)
        if found:
            return found
    return None


def free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Browser:
    """One headless Chrome page, spoken to over the DevTools protocol."""

    def __init__(self, binary: str, width: int = 1600, height: int = 1400) -> None:
        import requests
        import websocket

        self._requests = requests
        port = free_port()
        self._proc = subprocess.Popen(
            [
                binary,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--mute-audio",
                f"--remote-debugging-port={port}",
                "--remote-allow-origins=*",
                f"--window-size={width},{height}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Keep polling until a *page* target exists, not just until the endpoint
        # answers: on a loaded machine Chrome serves /json before it has created
        # its tab, and treating that empty list as ready failed the run instead of
        # waiting a quarter second longer.
        page = None
        for _ in range(120):
            try:
                tabs = self._requests.get(f"http://127.0.0.1:{port}/json", timeout=2).json()
                page = next((t for t in tabs if t.get("type") == "page"), None)
                if page is not None:
                    break
            except Exception:
                pass
            time.sleep(0.25)
        if page is None:
            self.close()
            raise RuntimeError("Chrome did not expose a DevTools page target")
        self._ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
        self._msg_id = 0
        self._send("Page.enable")
        self._send("Runtime.enable")

    def _send(self, method: str, params: dict | None = None) -> dict:
        self._msg_id += 1
        self._ws.send(json.dumps({"id": self._msg_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self._ws.recv())
            if message.get("id") == self._msg_id:
                if "error" in message:
                    raise RuntimeError(f"{method} failed: {message['error']}")
                return message.get("result", {})

    def goto(self, url: str) -> None:
        self._send("Page.navigate", {"url": url})

    def evaluate(self, expression: str):
        """The value of a JS expression, brought back by value."""
        result = self._send(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": True},
        )
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            described = (details.get("exception") or {}).get("description")
            raise RuntimeError(f"JS raised: {described or details.get('text')}\nin: {expression}")
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, *, what: str, timeout: float = 40.0):
        """Poll ``expression`` until it returns something truthy, then return it."""
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = self.evaluate(expression)
            if last:
                return last
            time.sleep(0.25)
        raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what} (last: {last!r})")

    def text(self) -> str:
        return self.evaluate("document.body.innerText") or ""

    def wait_until_idle(self, *, timeout: float = 40.0) -> None:
        """Wait until Streamlit is not mid-rerun.

        Streamlit shows its status widget while a script or fragment is
        running. Clicking during that window is a genuine race — the click can
        land on an element about to be replaced, or arrive interleaved with the
        rerun it triggered — so every test click waits for quiet first.
        """
        self.wait_for(
            "(() => !document.querySelector('[data-testid=\"stStatusWidget\"]'))()",
            what="the app to finish running",
            timeout=timeout,
        )

    def settle(self, *, quiet: float = 1.0) -> None:
        """Wait for queued reruns to drain, not just for the current one.

        A click on the graph component triggers a fragment rerun, which itself
        publishes an SVG snapshot and can queue another. Idle only means "not
        running right now" — acting on the first quiet moment can land a second
        interaction between a fragment's reruns, where it is processed against
        the state the *previous* run left behind.
        """
        self.wait_until_idle()
        time.sleep(quiet)
        self.wait_until_idle()

    def click(self, locator: str) -> None:
        """Real mouse click at the centre of the element ``locator`` returns.

        ``locator`` is a JS expression evaluating to an element (or null).
        Real press/release events rather than ``el.click()``: Streamlit's own
        widgets are React-managed, and a synthetic event does not always take.
        """
        self.wait_until_idle()
        # A rect from inside an iframe is relative to *that* frame's viewport,
        # while Input.dispatchMouseEvent speaks top-page coordinates — so walk
        # the frame chain and add each host iframe's offset.
        box = self.evaluate(
            f"""
(() => {{
  const el = {locator};
  if (!el) return null;
  const frames = [];
  let doc = el.ownerDocument;
  while (doc !== document) {{
    const frame = doc.defaultView && doc.defaultView.frameElement;
    if (!frame) break;
    frames.push(frame);
    doc = frame.ownerDocument;
  }}
  for (const frame of [...frames].reverse()) frame.scrollIntoView({{block: 'center'}});
  el.scrollIntoView({{block: 'center', inline: 'center'}});
  const r = el.getBoundingClientRect();
  let x = r.left + r.width / 2;
  let y = r.top + r.height / 2;
  for (const frame of frames) {{
    const fr = frame.getBoundingClientRect();
    x += fr.left;
    y += fr.top;
  }}
  return {{x, y}};
}})()
"""
        )
        if not box:
            raise AssertionError(f"nothing to click for: {locator}")
        self._send("Input.dispatchMouseEvent", {"type": "mouseMoved", **box})
        for kind, buttons in (("mousePressed", 1), ("mouseReleased", 0)):
            self._send(
                "Input.dispatchMouseEvent",
                {"type": kind, **box, "button": "left", "buttons": buttons, "clickCount": 1},
            )

    def resize_viewport(self, width: int, height: int) -> None:
        """Change the viewport size, as dragging the window would.

        Unlike a layout change, this reaches the app through nothing at all:
        Streamlit sends no render message, so anything that has to follow the
        width has to notice by itself.
        """
        self._send(
            "Emulation.setDeviceMetricsOverride",
            {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False},
        )

    def screenshot(self, path: Path) -> None:
        shot = self._send(
            "Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True}
        )
        Path(path).write_bytes(base64.b64decode(shot["data"]))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ws.close()
        self._proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=10)


class StreamlitApp:
    """A Streamlit script served on a throwaway port for the length of a test."""

    def __init__(
        self, script: Path, env: dict[str, str] | None = None, log_dir: Path | None = None
    ) -> None:
        self.port = free_port()
        # A file, not a pipe: nothing drains a pipe during the test, and
        # Streamlit fills the ~64 KB buffer within a few reruns — at which
        # point the app blocks mid-rerun and looks, from the browser, exactly
        # like a click that did nothing. Outside the repo unless asked
        # otherwise, so a test run leaves no files behind in the tree.
        log_dir = Path(log_dir) if log_dir else Path(tempfile.mkdtemp(prefix="pg-browser-test-"))
        self.log_path = log_dir / f"streamlit-{self.port}.log"
        self._log_file = self.log_path.open("w")
        run_env = {
            **os.environ,
            # The worktree's src tree, not whatever a shared venv resolves
            # behavior_graphs to.
            "PYTHONPATH": os.pathsep.join(
                [str(_REPO_ROOT / "src"), os.environ.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            **(env or {}),
        }
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(script),
                "--server.port",
                str(self.port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=run_env,
            text=True,
        )

    def log(self) -> str:
        """Whatever the app has logged so far — the first thing to read on failure."""
        self._log_file.flush()
        return self.log_path.read_text(errors="replace")

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/"

    def wait_until_serving(self, timeout: float = 90.0) -> None:
        import requests

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"streamlit exited early:\n{self.log()}")
            try:
                if requests.get(self.url, timeout=2).status_code == 200:
                    return
            except Exception:
                pass
            # Outside the except: a 503 from a warming-up server is a reason to
            # wait, not to spin the loop for the whole timeout.
            time.sleep(0.5)
        raise RuntimeError(f"streamlit did not serve {self.url} within {timeout:.0f}s")

    def close(self) -> None:
        self._proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            self._proc.wait(timeout=10)
        with contextlib.suppress(Exception):
            self._log_file.close()


# ── JS helpers for reaching into the graph component's iframe ────────────────
# components.html renders same-origin srcdoc iframes, so the top frame can read
# and click straight through contentDocument.

GRAPH_DOC = """
(() => {
  for (const f of document.querySelectorAll('iframe')) {
    let d;
    try { d = f.contentDocument } catch (e) { continue }
    if (d && d.querySelector('.node-circle')) return d;
  }
  return null;
})()
"""

#: How many nodes carry the selection halo — the translucent ring the component
#: draws behind the selected node. Not `.node-circle.selected`: that class is
#: only set on the *special* (START/terminal) shapes, so a cluster node shows as
#: selected purely by its halo.
SELECTED_COUNT = f"""
(() => {{
  const d = {GRAPH_DOC};
  if (!d) return -1;
  return [...d.querySelectorAll('g.node-group circle')].filter(
    c => (c.getAttribute('stroke') || '').startsWith('rgba(80, 200, 255')
  ).length;
}})()
"""

#: How many node groups the component has dimmed — the "greyed out" symptom.
DIMMED_COUNT = f"""
(() => {{
  const d = {GRAPH_DOC};
  if (!d) return -1;
  return [...d.querySelectorAll('g[opacity]')].filter(g => parseFloat(g.getAttribute('opacity')) < 1).length;
}})()
"""

#: The number of drawn nodes, used to wait for the component to be ready.
NODE_COUNT = (
    f"(() => {{const d = {GRAPH_DOC}; return d ? d.querySelectorAll('.node-circle').length : 0}})()"
)


#: How many fold badges currently read "−" (a branch showing its children).
OPEN_BADGE_COUNT = f"""
(() => {{
  const d = {GRAPH_DOC};
  if (!d) return -1;
  return [...d.querySelectorAll('g.fold-badge')].filter(b => b.textContent.includes('\\u2212')).length;
}})()
"""


def open_badge_locator() -> str:
    """A JS expression for the first "−" fold badge — one that can collapse."""
    return (
        f"(() => {{const d = {GRAPH_DOC}; if (!d) return null;"
        "return [...d.querySelectorAll('g.fold-badge')]"
        ".find(b => b.textContent.includes('\\u2212')) || null;})()"
    )


def node_locator(node_id: int) -> str:
    """A JS expression for the clickable area of the node with ``node_id``.

    The whole ``g.node-group`` carries the click listener, and its last
    ``.node-circle`` is the transparent oversized hit area — click that, so a
    press lands well inside the target rather than on its stroke.
    """
    return (
        f"(() => {{const d = {GRAPH_DOC}; if (!d) return null;"
        f"const g = d.querySelector('g.node-group[data-node-id=\"{node_id}\"]');"
        f"if (!g) return null;"
        f"const circles = g.querySelectorAll('.node-circle');"
        f"return circles[circles.length - 1] || g;}})()"
    )


def label_locator(text: str) -> str:
    """A JS expression for the radio/checkbox label reading exactly ``text``.

    Streamlit's widget labels are the only handle a test has on its controls, and
    every browser test here drives the app through one.
    """
    return f"[...document.querySelectorAll('label')].find(l => l.innerText.trim() === {text!r})"


def reported_value(name: str) -> str:
    """A JS expression for ``name``'s value in an app's ``NAME=value`` readout.

    The browser test apps write their state to the page that way, because it is
    the only part of a custom component's effect a test can read as text.
    """
    return (
        f"(() => {{const p = {name!r} + '=';"
        " const l = document.body.innerText.split('\\n').find(t => t.startsWith(p));"
        " return l ? l.slice(p.length).trim() : null;})()"
    )
