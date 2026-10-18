"""Expand ``${VAR}`` in default paths, and refuse to run on an unexpanded one.

argparse defaults are built at import, so a literal ``Path("${DATA_PATH}/…")``
is never expanded by anything: it becomes a cwd-relative directory whose first
component is the eleven characters ``${DATA_PATH}``. Nothing errors. The script
finds no datasets, writes an empty result, and exits 0 -- a successful-looking
no-op, which is the worst shape a missing-data failure can take.

``env_path`` expands at import where it can, and ``require_expanded`` turns the
leftover case into a message naming the variable instead of a silent empty run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["env_path", "require_expanded"]


def env_path(template: str) -> Path:
    """``Path`` with ``${VAR}`` expanded from the environment."""
    return Path(os.path.expandvars(template))


def require_expanded(*paths: Path | str) -> None:
    """Exit with a usable message if any path still carries a ``${VAR}``."""
    unexpanded = [str(p) for p in paths if "${" in str(p)]
    if not unexpanded:
        return
    names = sorted({s[s.index("${") + 2 : s.index("}")] for s in unexpanded if "}" in s})
    sys.exit(
        f"unset environment variable(s): {', '.join(names)}. "
        f"Paths still carrying a placeholder: {', '.join(unexpanded)}. "
        "Export them (see docs/data.md) or pass the paths explicitly."
    )
