"""Loading a ``scripts/`` entry point from a test.

Written once here because `scripts/` is not an importable package, and several
test modules had grown their own copy of the incantation — copies that drifted:
some re-executed the script on every call rather than once per module, and some
skipped the ``sys.modules`` registration a script's own ``dataclass``/``pickle``
use needs.

Deliberately not ``conftest.py``: ``tests/browser/`` has one, and two modules
called ``conftest`` on ``sys.path`` shadow each other under pytest's prepend
import mode.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_script(name: str, path: str | Path | None = None):
    """Import ``scripts/<name>.py`` (or *path*) as a module named *name*.

    Call at module scope: executing a script re-reads and re-runs it, which is
    wasted work per test and, for a script with import side effects, not
    idempotent.
    """
    target = (
        REPO / "scripts" / (path or f"{name}.py")
        if not Path(path or "").is_absolute()
        else Path(path)
    )
    spec = importlib.util.spec_from_file_location(name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
