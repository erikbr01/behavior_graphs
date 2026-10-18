"""Every shipped file must parse at the Python version pyproject claims.

`requires-python = ">=3.10"`, but the venv this suite runs in is 3.13 and the
one the training submodule uses is 3.9 -- so nothing here has ever executed on
the oldest interpreter the package promises to support. Syntax added after
3.10 would pass every check we run and fail at import for a user who took the
floor at its word.

Two checks, because they fail on different things and neither subsumes the
other:

* `ast.parse(..., feature_version=(3, 10))` runs from any interpreter and
  gates the GRAMMAR. It rejects `except*` (3.11) and PEP 695 type parameters
  (3.12) -- real work, and free.
* compiling under a real 3.10 catches what `feature_version` cannot. PEP 701
  (nested same-type quotes in f-strings, 3.12) is a TOKENIZER change, and
  `ast.parse('x = f"{"a":>3s}"', feature_version=(3, 10))` accepts it happily
  on 3.13 while 3.10 itself says `SyntaxError: f-string: expecting '}'`.

A neighbouring project on this machine hit exactly the f-string case: a module
only their newer checks had ever parsed, found as 13 consecutive failed runs.
Their correction is why the second check is here -- the first one alone would
have reported this tree clean with a 3.12-only f-string sitting in it.
"""

from __future__ import annotations

import ast
import pathlib
import shutil
import subprocess

import pytest
import tomllib

REPO = pathlib.Path(__file__).resolve().parents[1]
SHIPPED = ("src", "scripts", "apps", "tests", "reproduce")


def _floor() -> tuple[int, int]:
    spec = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["requires-python"]
    major, _, minor = spec.lstrip(">=~^ ").partition(".")
    return int(major), int(minor.split(".")[0])


def _python_files() -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for top in SHIPPED:
        root = REPO / top
        if root.is_dir():
            out += [p for p in root.rglob("*.py") if ".venv" not in p.parts]
    return sorted(out)


def test_the_floor_is_what_we_think_it_is():
    assert _floor() == (3, 10)


def test_every_shipped_file_parses_at_the_floor():
    floor = _floor()
    files = _python_files()
    assert len(files) > 200, f"only found {len(files)} files; the glob is wrong"

    broken: list[str] = []
    for path in files:
        try:
            ast.parse(path.read_text(), filename=str(path), feature_version=floor)
        except SyntaxError as error:
            broken.append(f"{path.relative_to(REPO)}:{error.lineno}: {error.msg}")
    assert not broken, "syntax newer than python{}.{}:\n  ".format(*floor) + "\n  ".join(broken)


def test_the_grammar_check_would_actually_catch_something():
    """A guard that cannot fail is worse than none -- pin that it can."""
    with pytest.raises(SyntaxError):
        ast.parse("type Alias = int\n", feature_version=_floor())
    # ... and pin what it does NOT catch, so nobody retires the slower check
    # below believing this one covers it.
    ast.parse('x = f"{"a":>3s}"\n', feature_version=_floor())


def _floor_interpreter() -> str | None:
    """A real interpreter at the declared floor, if this machine has one."""
    major, minor = _floor()
    found = shutil.which(f"python{major}.{minor}")
    if found:
        return found
    uv = shutil.which("uv")
    if uv:
        out = subprocess.run(
            [uv, "python", "find", f"{major}.{minor}"], capture_output=True, text=True
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


def test_every_shipped_file_compiles_under_a_real_floor_interpreter():
    """The authoritative check: tokenizer changes only a real 3.10 rejects."""
    interpreter = _floor_interpreter()
    if interpreter is None:
        pytest.skip("no python{}.{} on this machine".format(*_floor()))

    out = subprocess.run(
        [interpreter, "-m", "compileall", "-q", "-x", r"(\.venv|third_party)", *SHIPPED],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, (
        f"{interpreter} cannot compile the tree:\n{out.stdout[-3000:]}{out.stderr[-3000:]}"
    )
