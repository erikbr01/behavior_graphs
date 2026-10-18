#!/usr/bin/env python
"""Check Table I's claims survive a re-fit, rather than its digits.

Table I comes from a Bayesian mixed-effects fit -- four NUTS chains. The sampler
IS seeded (`_SEED` in viz/streamlit/user_study/mixed_effects.py), so the same
machine and the same library versions reproduce it exactly; a different pymc,
numpy or BLAS does not, because the seed fixes the random stream and not the
floating-point arithmetic underneath it. Measured across two environments here,
the third significant figure moves. Diffing digits would therefore fail a
perfectly good reproduction, which is why this checks the claims the paper
actually makes:

  * temporal demand and effort are reduced, and their intervals exclude zero
  * confidence and success-rate regret are unchanged, intervals include zero
  * each point estimate agrees with the published one within MCMC tolerance

Run it after `export_paper_results.py --sections adjusted --stdout > fresh.json`,
or let it do that itself.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PUBLISHED = REPO / "src" / "behavior_graphs" / "results" / "paper_results.json"

#: Outcomes the paper claims a reduction in, with intervals excluding zero.
REDUCED = {"temporal_demand", "effort"}
#: Outcomes the paper claims are unchanged, with intervals including zero.
UNCHANGED = {"confidence", "regret"}


def _label(row: dict) -> str:
    for key in ("outcome", "label", "name", "metric"):
        if key in row:
            return str(row[key]).strip().lower()
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", type=pathlib.Path, help="a previously exported adjusted section")
    ap.add_argument(
        "--point-tolerance",
        type=float,
        default=2.0,
        help="rating points a point estimate may move under re-fitting",
    )
    args = ap.parse_args()

    pub = json.loads(PUBLISHED.read_text())["adjusted"]["rows"]
    if args.fresh:
        fresh = json.loads(args.fresh.read_text())
        fresh = fresh.get("adjusted", fresh)["rows"]
    else:
        proc = subprocess.run(
            [
                sys.executable,
                "scripts/export_paper_results.py",
                "--sections",
                "adjusted",
                "--stdout",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stderr.strip()[-1200:], file=sys.stderr)
            return proc.returncode
        fresh = json.loads(proc.stdout)["adjusted"]["rows"]

    by_label = {_label(r): r for r in fresh}
    problems: list[str] = []
    print(f"  {'outcome':<22} {'published':>18}  {'re-fit':>18}   claim")
    for prow in pub:
        label = _label(prow)
        frow = by_label.get(label)
        if frow is None:
            problems.append(f"{label}: absent from the re-fit")
            continue
        pd_, fd = prow.get("difference"), frow.get("difference")
        lo, hi = frow.get("lower"), frow.get("upper")
        excludes_zero = lo is not None and hi is not None and (lo > 0) == (hi > 0)
        if label in REDUCED:
            claim = "reduced, excludes 0"
            if not (fd is not None and fd < 0 and excludes_zero):
                problems.append(
                    f"{label}: expected a reduction with an interval excluding 0, "
                    f"got {fd} [{lo}, {hi}]"
                )
        elif label in UNCHANGED:
            claim = "unchanged, includes 0"
            if excludes_zero:
                problems.append(f"{label}: expected an interval including 0, got [{lo}, {hi}]")
        else:
            # Not a pass: an unrecognised label means this script checked
            # nothing for that row, which is worse than failing loudly. The
            # first draft of this file silently skipped two of the four rows
            # because it matched on prose labels and the data uses keys.
            claim = "UNRECOGNISED"
            problems.append(
                f"{label}: not in REDUCED or UNCHANGED, so no claim was checked "
                f"-- update this script rather than ignoring the row"
            )
        if pd_ is not None and fd is not None and abs(pd_ - fd) > args.point_tolerance:
            problems.append(
                f"{label}: point estimate moved {abs(pd_ - fd):.2f} points "
                f"({pd_} -> {fd}), beyond the {args.point_tolerance} tolerance"
            )
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"  {label:<22} {fmt(pd_):>18}  {fmt(fd):>18}   {claim}")

    if problems:
        print(f"\n  {len(problems)} problem(s):")
        for line in problems:
            print("   ", line)
        return 1
    print(
        "\n  every claim Table I makes survives the re-fit; point estimates within "
        f"{args.point_tolerance} rating points (the sampler is seeded, but the "
        "digits still move across library versions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
