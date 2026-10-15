#!/usr/bin/env python
"""Check the published progress table against itself.

Table II's inputs -- the benchmark's per-dataset ``*_metrics.json`` -- are not
carried in this repo, so the table cannot be recomputed from a plain checkout.
What *can* be checked, and is worth checking, is that the table is internally
consistent: every per-domain mean should be the mean of the per-dataset rows it
claims to summarise, and the dataset counts should add up to the 43 the paper
reports. A transcription slip or a stale export shows up here.
"""

from __future__ import annotations

import json
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "src" / "behavior_graphs" / "results" / "paper_results.json"
TOL = 5e-3  # the export rounds; this is tight enough to catch a real slip


def main() -> int:
    p = json.loads(RESULTS.read_text())["progress"]
    rows, means = p["rows"], p["means"]
    metrics = [m["key"] for m in p["metrics"]]
    problems: list[str] = []

    total = sum(d["datasets"] for d in p["domains"])
    if total != p["datasets"]:
        problems.append(f"domain dataset counts sum to {total}, header says {p['datasets']}")
    shape = ", ".join(f"{d['label']} {d['datasets']}" for d in p["domains"])
    print(f"  datasets: {p['datasets']} across {len(p['domains'])} domains ({shape})")

    checked = 0
    for mean in means:
        domain, method = mean["domain"], mean["method"]
        for metric in metrics:
            if mean.get(metric) is None:
                continue
            # "Average" is the paper's all-43 row: a pseudo-domain with no
            # per-dataset rows of its own, so it averages over every real one.
            vals = [
                r[metric]
                for r in rows
                if r.get("method") == method
                and r.get(metric) is not None
                and (domain == "Average" or r.get("domain") == domain)
            ]
            if not vals:
                problems.append(f"{domain}/{method}/{metric}: mean published, no rows behind it")
                continue
            got = statistics.fmean(vals)
            if abs(got - mean[metric]) > TOL:
                problems.append(
                    f"{domain}/{method}/{metric}: mean {mean[metric]:.4f} "
                    f"but rows average {got:.4f} over {len(vals)}"
                )
            checked += 1

    n_avg = sum(1 for m in means if m["domain"] == "Average")
    print(
        f"  means checked against their rows: {checked} "
        f"({n_avg} of them the all-{p['datasets']} Average row)"
    )
    if problems:
        print(f"\n  {len(problems)} inconsistency(ies):")
        for line in problems[:20]:
            print("   ", line)
        return 1
    print("  the published table is internally consistent")
    print("\n  note: this does not re-derive the numbers. Table II's per-dataset")
    print("  metric files are not in this repo -- see REPRODUCE.md, step 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
