#!/usr/bin/env python3
"""Reproducibly fit and export the user-study mixed-effects summaries.

Example:
    uv run --extra analysis python scripts/analysis/user_study_mixed_effects.py \
        study_results_20260913_1521 --output-dir outputs/user_study_statistics

The response directory contains one JSON file per study attempt.  When the
retrained-policy rollout data are mounted, the script also scores and analyzes
data-collection regret.  The fixed seed, sampling settings, input directory,
and measured arm rates are recorded in the Markdown report.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from behavior_graphs.viz.streamlit.user_study import mixed_effects, regret
from behavior_graphs.viz.streamlit.user_study.response_store import LocalResponseStore


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "outcome",
        "estimate",
        "credible_interval_lower",
        "credible_interval_upper",
        "unit",
        "n",
        # Sampler diagnostics, not results: r_hat says whether the chains agree,
        # ess_bulk how many independent draws the estimate is really worth, and
        # mcse_mean how much of the estimate is Monte-Carlo error. The last one
        # is what licenses a digit -- at an mcse of 0.1 rating points, quoting
        # an effect to two decimals reports the sampler, not the study.
        "r_hat",
        "ess_bulk",
        "mcse_mean",
    ]
    with path.open("w", newline="") as stream:
        # A fit whose ArviZ did not report a diagnostic leaves the cell empty
        # rather than dropping the row.
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _read_rates(path: Path) -> dict:
    rates = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rate = regret.SuccessRate(
                dataset=row["dataset"],
                successes=int(row["successes"]),
                trials=int(row["trials"]),
            )
            rates[rate.dataset] = rate
    return rates


def _write_rates(path: Path, rates: dict) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "successes", "trials"])
        writer.writeheader()
        writer.writerows(
            {
                "dataset": rate.dataset,
                "successes": rate.successes,
                "trials": rate.trials,
            }
            for rate in sorted(rates.values(), key=lambda rate: rate.dataset)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "responses", type=Path, help="Directory containing exported response JSON files"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for CSV and Markdown outputs"
    )
    parser.add_argument("--draws", type=int, default=2_000, help="Posterior draws per chain")
    parser.add_argument("--tune", type=int, default=2_000, help="NUTS tuning draws per chain")
    parser.add_argument(
        "--skip-regret", action="store_true", help="Do not read retrained-policy outcomes"
    )
    parser.add_argument(
        "--arm-rates-csv",
        type=Path,
        help="CSV with dataset,successes,trials, exported by an earlier analysis run",
    )
    args = parser.parse_args()

    responses = LocalResponseStore(args.responses).list_all()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rating_effects = mixed_effects.continuous_effects(
        responses,
        draws=args.draws,
        tune=args.tune,
    )
    rows = [
        {
            "outcome": effect["outcome"],
            "estimate": effect["mean"],
            "credible_interval_lower": effect["lower"],
            "credible_interval_upper": effect["upper"],
            "unit": "rating points (Graph minus Videos)",
            "n": sum(
                row["outcome"] == effect["outcome"]
                for row in mixed_effects.assigned_rating_rows(responses)
            ),
            **{name: effect[name] for name in ("r_hat", "ess_bulk", "mcse_mean") if name in effect},
        }
        for effect in rating_effects
    ]

    regret_effect = None
    rates = {}
    if not args.skip_regret:
        rates = (
            _read_rates(args.arm_rates_csv)
            if args.arm_rates_csv
            else regret.measured_success_rates()
        )
        if rates:
            _write_rates(args.output_dir / "measured_arm_rates.csv", rates)
        regret_effect = mixed_effects.regret_effect(
            responses,
            rates,
            draws=args.draws,
            tune=args.tune,
        )
        if regret_effect is not None:
            rows.append(
                {
                    "outcome": "data_collection_regret",
                    "estimate": regret_effect["mean"],
                    "credible_interval_lower": regret_effect["lower"],
                    "credible_interval_upper": regret_effect["upper"],
                    "unit": "success-rate points (Graph minus Videos)",
                    "n": regret_effect["n"],
                    **{
                        name: regret_effect[name]
                        for name in ("r_hat", "ess_bulk", "mcse_mean")
                        if name in regret_effect
                    },
                }
            )

    _write_csv(args.output_dir / "mixed_effects_summary.csv", rows)
    completed = {row["participant"] for row in mixed_effects.assigned_rating_rows(responses)}
    report = [
        "# User-study mixed-effects analysis",
        "",
        f"Input directory: `{args.responses}`",
        f"Completed randomized-stage participants: {len(completed)}",
        f"Posterior draws per chain: {args.draws}",
        f"Tuning draws per chain: {args.tune}",
        "Random seed: 20260913",
        "",
        "The rating and regret models use visualization, task, study stage, and experience as fixed effects, plus a participant random intercept.",
        "",
        "## Effects",
        "",
        "| Outcome | Graph minus Videos | 95% credible interval | Unit | N | r_hat | ESS | MCSE |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        report.append(
            f"| {row['outcome']} | {row['estimate']:.2f} | "
            f"[{row['credible_interval_lower']:.2f}, {row['credible_interval_upper']:.2f}] | "
            f"{row['unit']} | {row['n']} | "
            f"{row.get('r_hat', float('nan')):.3f} | "
            f"{row.get('ess_bulk', float('nan')):.0f} | "
            f"{row.get('mcse_mean', float('nan')):.2f} |"
        )
    worst_mcse = max((row.get("mcse_mean", 0.0) for row in rows), default=0.0)
    report.extend(
        [
            "",
            f"Largest Monte-Carlo standard error on an estimate: {worst_mcse:.2f} points. "
            "Quoting an effect or an interval bound to finer than that reports the "
            "sampler rather than the study — the fit is seeded, so that error is "
            "reproducible rather than absent. `r_hat` above 1.01 is refused outright "
            f"(see `mixed_effects._MAX_R_HAT`); these are all {max((row.get('r_hat', 0.0) for row in rows), default=0.0):.3f} or below.",
        ]
    )
    if not args.skip_regret:
        rate_source = args.arm_rates_csv or "mounted retrained-policy outcomes"
        report.extend(
            [
                "",
                f"Measured retrained-policy arms available: {len(rates)}",
                f"Arm-rate source: `{rate_source}`",
            ]
        )
        if regret_effect is None:
            report.append(
                "Regret was not fit because the available responses and arm rates yielded no scorable randomized-stage choices."
            )
    (args.output_dir / "mixed_effects_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
