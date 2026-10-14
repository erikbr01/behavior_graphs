"""Shared subprocess and path helpers for environment adapters."""

from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from collections import deque
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger(__name__)

_LOG_TAIL_LINES = 50


def default_repo_root() -> Path:
    """Walk up from this file to the repo root (first dir with pyproject.toml)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not locate the repo root above {here}; pass repo_root explicitly")


def run_streaming(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path | None = None,
    extra_env: dict | None = None,
) -> None:
    """Run a subprocess, streaming output to stdout (and ``log_path`` if given).

    ``extra_env`` entries are overlaid on the inherited environment.
    Raises ``RuntimeError`` with the tail of the output on failure.
    """
    import os

    env = None
    if extra_env:
        env = {**os.environ, **{k: str(v) for k, v in extra_env.items()}}
    cmd = [str(c) for c in cmd]
    log.info(
        "running: %s (cwd=%s%s)", shlex.join(cmd), cwd, f", log={log_path}" if log_path else ""
    )
    tail: deque = deque(maxlen=_LOG_TAIL_LINES)
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if log_file is not None:
                log_file.write(line)
            sys.stdout.write(line)
            tail.append(line)
        returncode = proc.wait()
    finally:
        if log_file is not None:
            log_file.close()
    if returncode != 0:
        raise RuntimeError(
            f"subprocess failed with exit code {returncode}: {shlex.join(cmd)}\n"
            f"last {len(tail)} lines:\n{''.join(tail)}"
        )
