#!/usr/bin/env python
"""Diff a freshly computed results section against the published one.

The point of a reproduction script is to *check*, not to render. This recomputes
a section of ``results/paper_results.json`` and reports every value that differs
from the one the paper reports, rather than printing numbers and leaving the
reader to compare by eye.

    python reproduce/verify_against_published.py policies
    python reproduce/verify_against_published.py progress --tolerance 1e-6

Exits non-zero if anything differs, so it works in CI.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = REPO / "src" / "behavior_graphs" / "results" / "paper_results.json"


def diff(a, b, path: str = "", tol: float = 1e-9) -> list[str]:
    """Every leaf where *a* (published) and *b* (fresh) disagree."""
    if type(a) is not type(b):
        return [f"{path}: type {type(a).__name__} vs {type(b).__name__}"]
    out: list[str] = []
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}/{k}: only in fresh")
            elif k not in b:
                out.append(f"{path}/{k}: only in published")
            else:
                out += diff(a[k], b[k], f"{path}/{k}", tol)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > tol:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a!r} vs {b!r}")
    return out


def _verify_curves(args) -> int:
    """`progress_curves.json` -- the 42,000 numbers the site's charts consume.

    It ships beside paper_results.json and had no reproduction check of its
    own: every other published artifact was diffed against a re-derivation
    and this one was taken on trust. It is written as a side effect of the
    `progress` export rather than to stdout, so it needs its own path through
    here: export to a temporary file and diff that.
    """
    import tempfile

    from behavior_graphs.results import CURVES_PATH

    if not CURVES_PATH.exists():
        print(f"no published curves at {CURVES_PATH}", file=sys.stderr)
        return 2
    published = json.loads(CURVES_PATH.read_text())

    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "paper_results.json"
        curves = pathlib.Path(tmp) / "progress_curves.json"
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/export_paper_results.py",
                "--sections",
                "progress",
                "--out",
                str(out),
                "--curves",
                str(curves),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stderr.strip()[-1500:], file=sys.stderr)
            return proc.returncode
        if not curves.exists():
            print("the export wrote no curves file", file=sys.stderr)
            return 2
        fresh = json.loads(curves.read_text())

    differences = diff(published, fresh, tol=args.tolerance)
    if not differences:
        print(f"curves: reproduces exactly ({len(json.dumps(published)):,} bytes compared)")
        return 0
    print(f"curves: {len(differences)} value(s) differ from the published curves")
    for line in differences[: args.max_shown]:
        print("   ", line)
    if len(differences) > args.max_shown:
        print(f"    ... and {len(differences) - args.max_shown} more")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("section", help="policies | study | adjusted | progress | curves")
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--max-shown", type=int, default=20)
    args = ap.parse_args()

    if args.section == "curves":
        return _verify_curves(args)

    if not PUBLISHED.exists():
        print(f"no published results at {PUBLISHED}", file=sys.stderr)
        return 2
    published = json.loads(PUBLISHED.read_text()).get(args.section)
    if published is None:
        print(f"published results carry no {args.section!r} section", file=sys.stderr)
        return 2

    proc = subprocess.run(
        [sys.executable, "scripts/export_paper_results.py", "--sections", args.section, "--stdout"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip()[-1500:], file=sys.stderr)
        return proc.returncode
    try:
        fresh = json.loads(proc.stdout)[args.section]
    except Exception as exc:  # noqa: BLE001 -- report, don't mask
        print(f"could not parse the export's output: {exc}", file=sys.stderr)
        return 2

    differences = diff(published, fresh, tol=args.tolerance)
    if not differences:
        print(f"{args.section}: reproduces exactly ({len(json.dumps(published))} bytes compared)")
        return 0
    print(f"{args.section}: {len(differences)} value(s) differ from the published results")
    for line in differences[: args.max_shown]:
        print("   ", line)
    if len(differences) > args.max_shown:
        print(f"    ... and {len(differences) - args.max_shown} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
