#!/usr/bin/env python
"""Write the paper's reported results to the JSON the reviewer page reads.

``try.behavior-graphs.com`` serves results, and the data behind them lives in
three places this repo does not carry: the retrained policies' scored rollouts,
the user study's response store, and the progress benchmark's metric files. The
reviewer container mounts none of the three (see the ``reviewer`` service in
``deploy/docker-compose.yml``), and fitting the study's mixed model takes four
MCMC chains. So the numbers are exported once, by this script, into
:mod:`behavior_graphs.results` — and the page reads that.

Everything here is *read* from an artifact on disk. Nothing is typed in from the
paper: a number on the website that no file produced is a number nobody can
re-check, and the whole point of the page is that a reviewer can.

Sections, and what each one needs:

``policies``   ``$DATA_PATH/rollouts`` — the scored rollout collections of the
               baseline and retrained policies. Pairwise exact tests run here,
               so the page does not.
``study``      ``$STUDY_RESULTS_DIR`` — the response store. Participants are
               re-keyed to ``P01…`` on the way out; the store's session ids do
               not leave it.
``adjusted``   The same store, plus the optional ``analysis`` extra (bambi,
               arviz), because this is the mixed-effects fit. Not installed in a
               plain checkout; it *is* installed in the deployment image, whose
               code mounts are read-only, so that run goes through a copy::

                   docker cp src/behavior_graphs/results/paper_results.json \
                       <container>:/tmp/paper_results.json
                   docker exec -e STUDY_RESULTS_DIR=/app/study_results \
                       -e DATA_PATH=/data <container> \
                       python /app/scripts/export_paper_results.py \
                       --sections adjusted --out /tmp/paper_results.json
                   docker cp <container>:/tmp/paper_results.json \
                       src/behavior_graphs/results/paper_results.json

               ``--stdout`` is no help for this one: PyMC writes its sampling
               progress to stdout, so the document comes back with a table of
               draw counts in front of it.
``progress``   ``$PROGRESS_RESULTS_ROOT`` — the benchmark's
               ``{dataset}_metrics.json`` files for the table, and its
               ``*_curves.npz`` files for the predicted curves. This section
               writes **two** files: the metrics into the document, and the
               curves into ``progress_curves.json`` beside it (see
               :mod:`behavior_graphs.results` for why they are apart).

Sections merge: a run with ``--sections progress`` rewrites that key and leaves
the rest of the file as it was, which is what lets the three sources be read on
the three machines that have them.

Usage:
    python scripts/export_paper_results.py                      # all four
    python scripts/export_paper_results.py --sections progress
    python scripts/export_paper_results.py --sections adjusted --stdout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from behavior_graphs.results import CURVES_PATH, DEFAULT_PATH, SCHEMA  # noqa: E402

#: Metric keys the progress section carries, in the order the table shows them.
#: A subset of what the benchmark writes: ``pearson`` correlates with ``voc`` at
#: r = 0.989 over these 43 datasets and picks a different winner on one of them,
#: so it is very nearly VOC printed twice.
PROGRESS_METRICS = ("voc", "kendall", "mae", "auroc")

#: Lower is better for exactly one of them, and a chart that does not know
#: which ranks MAE upside-down.
PROGRESS_HIGHER_BETTER = {"voc": True, "kendall": True, "mae": False, "auroc": True}

#: The four methods the benchmark table compares, ours first. Fixed rather than
#: read off the files: the table's whole claim is one apples-to-apples
#: comparison, and it only counts datasets every one of these scored.
PROGRESS_METHODS = ("graph_stage", "mlp", "robometer_4b", "topreward_ll")

#: What to call each of them. ``graph_stage`` is the piecewise-linear readout on
#: the graph's behaviors; ``mlp`` reads the *same* policy embedding the graph is
#: built on, which is what makes it the comparison that isolates the structure.
PROGRESS_METHOD_LABELS = {
    "graph_stage": "Behavior Graph",
    "mlp": "MLP",
    "robometer_4b": "ROBOMETER-4B",
    "topreward_ll": "TOPReward",
}

#: Which domain each dataset belongs to, by name prefix, and the order the
#: table lists them. A ``droid`` prefix exists nowhere in the artifacts: the
#: real-robot sets are named by task, so anything reading the domain off a
#: filename scatters them over three rows.
PROGRESS_DOMAINS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("LIBERO", ("libero",)),
    ("Real-world", ("set_table", "prepare_tray", "cooking")),
    ("Robomimic", ("robomimic", "transport_mh_salvage250")),
    ("MimicGen", ("mimicgen",)),
)

#: Datasets the benchmark table excludes. The SAFE real-robot sets carry no
#: video, so no VLM baseline can run on them, and a row including them would
#: compare two methods on one subset and four on another.
PROGRESS_EXCLUDED = ("safe_franka", "safe_widowx")

#: Points the predicted progress curves are resampled onto, spanning a rollout
#: from its first frame to its last.
#:
#: Resampled rather than stored as written, for two reasons. Rollouts differ in
#: length, so a mean over them needs a common x; and the methods predict at
#: different native resolutions -- the graph labels sliding windows, the MLP
#: labels frames -- while the artifacts do not record the window geometry
#: needed to expand one onto the other. Fraction of the rollout is the x the
#: four of them are mutually comparable on.
#:
#: 41 points rather than a round 20: the graph's estimator is piecewise linear
#: over the behaviors it found, and at 5% resolution its kinks alias into a
#: smooth curve, which is the one feature of its shape worth seeing.
CURVE_GRID = 41

#: How many rollouts per collection carry their curves into the export, split
#: evenly between the two outcomes. A cap, because the 43 collections hold
#: 7,682 rollouts between them and one LIBERO pool has 1,000 of them; filled
#: by rollout index, so which ones appear is reproducible.
#:
#: The two outcomes are never mixed into one curve. A rollout that succeeded
#: ran the task to the end, so true progress is the fraction elapsed; a failed
#: one stopped early, so its true progress plateaus. Both are drawn, one
#: rollout at a time, and the outcome is on the row.
CURVE_ROLLOUTS = 12

SECTIONS = ("policies", "study", "adjusted", "progress")


# ── policies: the retrained policies' success rates ──────────────────────────


def export_policies() -> dict:
    """The comparison figure's numbers: counts, posteriors, letters, p-values.

    The violins are a Beta posterior over ``successes``/``trials``, so the page
    draws them from the counts rather than from a sampled shape. The letters and
    the adjusted p-values are exported because they are the expensive part —
    Barnard's exact test is a grid search per pair — and because they must not
    come out differently on the website than in the paper.
    """
    from behavior_graphs.stats.policy_comparison import compare_policies
    from behavior_graphs.viz.streamlit import policy_groups
    from behavior_graphs.viz.streamlit.rollout_outcomes import discover_datasets, load_outcomes

    datasets = discover_datasets()
    groups = policy_groups.default_groups(list(datasets))
    if not groups:
        raise SystemExit(
            "No rollout collection of the compared policies is on this machine. "
            "Set DATA_PATH (or BEHAVIOR_GRAPHS_RUNS_ROOT) to the data root holding "
            "rollouts/<dataset>/."
        )

    tasks = []
    for label, names in groups.items():
        arms = [
            load_outcomes(name, datasets[name]).as_arm(policy_groups.paper_label(name) or name)
            for name in names
        ]
        result = compare_policies(arms, pin_first=policy_groups.BASELINE_LABEL)
        # `result.order` is where the arms sit (baseline first, then by rate);
        # the letters are named by rate regardless, so `a` is the best arm in
        # every panel and not the leftmost one.
        tasks.append(
            {
                "label": label,
                "arms": [
                    {
                        "dataset": names[i],
                        "label": result.labels[i],
                        "successes": arms[i].successes,
                        "trials": arms[i].trials,
                        "letters": result.letters[i],
                        "baseline": result.labels[i] == policy_groups.BASELINE_LABEL,
                    }
                    for i in result.order
                ],
                # Row and column follow the arm order above, so a page that has
                # only this JSON can label the matrix without re-deriving it.
                "pvalues": [
                    [_finite(result.adjusted[i][j]) for j in result.order] for i in result.order
                ],
            }
        )
    first = next(iter(groups))
    sample = compare_policies(
        [
            load_outcomes(name, datasets[name]).as_arm(policy_groups.paper_label(name) or name)
            for name in groups[first]
        ],
        pin_first=policy_groups.BASELINE_LABEL,
    )
    return {
        "tasks": tasks,
        "prior": list(sample.prior),
        "test": sample.test,
        "correction": sample.correction,
        "alpha": sample.alpha,
        "credible_level": sample.credible_level,
    }


def _finite(value: object) -> float | None:
    """A JSON number, or ``None`` where the test could not be run."""
    number = float(value)  # type: ignore[arg-type]
    return None if math.isnan(number) else number


# ── study: what the participants answered ────────────────────────────────────


def _responses() -> list[dict]:
    from behavior_graphs.viz.streamlit.user_study import claims
    from behavior_graphs.viz.streamlit.user_study.response_store import (
        LocalResponseStore,
        results_dir,
    )

    store = LocalResponseStore(results_dir())
    reported = claims.study_responses(store.list_all())
    if not reported:
        raise SystemExit(
            f"No completed response of the reported layout under {results_dir()}. "
            "Set STUDY_RESULTS_DIR to the response store."
        )
    return reported


def _person(store_id: str) -> str:
    """One spelling of a participant key, whichever module produced it.

    ``analytics.participant`` returns ``session:<id>`` and
    ``mixed_effects._participant`` returns the bare ``<id>``, on purpose: one is
    a label for a table and the other a grouping key for a model. Both name the
    same person, and the two exported row sets have to agree about who that is,
    or a paired line in the ratings panel and a dot in the regret panel come off
    one participant under two numbers.
    """
    return store_id.removeprefix("session:")


def _participant_keys(responses: Sequence[dict]) -> dict[str, str]:
    """``{participant: "P01"}`` — the ids the store keys by do not leave it.

    Keyed by :func:`_person`, and sorted by it rather than by anything a
    participant answered, so the numbering is stable across runs and carries no
    ordering a reader could mistake for the order they took part in.
    """
    from behavior_graphs.viz.streamlit.user_study import analytics

    ids = sorted({_person(analytics.participant(response)) for response in responses})
    return {name: f"P{index:02d}" for index, name in enumerate(ids, start=1)}


def export_study() -> dict:
    from behavior_graphs.viz.streamlit.user_study import analytics, claims, mixed_effects, regret

    responses = _responses()
    keys = _participant_keys(responses)
    rates = regret.measured_success_rates()
    scored = regret.regret_rows(responses, rates)
    if not scored:
        raise SystemExit(
            "No data-collection answer could be scored: that needs the retrained "
            "policies' rollouts on disk as well as the responses. Set DATA_PATH."
        )

    return {
        "participants": len(responses),
        "preference": claims.preference_claim(responses),
        "paired": claims.paired_rating_claims(responses),
        # Long format, one row per rating, because the panels want the dots and
        # the paired lines as well as the boxes. These are the randomized
        # stages only -- the ones the adjusted fit is run on -- so a reader
        # checking a box against an interval is looking at the same evidence.
        "ratings": [
            {
                "participant": keys[_person(row["participant"])],
                "task": row["task"],
                "stage": row["stage"],
                "display": "graph" if row["graph"] else "videos",
                "outcome": row["outcome"],
                "score": row["score"],
            }
            for row in mixed_effects.assigned_rating_rows(responses)
            if _person(row["participant"]) in keys
        ],
        # Every scored answer, with the flag that says whether its display was
        # drawn for the participant or picked by them: the display comparison is
        # the randomized stages, and the page has to be able to say so.
        "regret": [
            {
                "participant": keys.get(_person(row["participant"]), ""),
                "task": row["task"],
                "task_label": row["task_label"],
                "display": row["phase"],
                "randomized": row.get("assignment", "") != "chosen",
                "choice_label": row["choice_label"],
                "dataset": row["dataset"],
                "regret": row["regret"],
                "chose_best": row["chose_best"],
            }
            for row in scored
        ],
        # How often each option was picked off each display, every option of
        # every task getting a row -- including the ones nobody picked, which is
        # the interesting thing a display can do.
        "choices": [
            {
                "task": row["task"],
                "task_label": row["task_label"],
                "display": row["phase"],
                "dataset": row["dataset"],
                "choice_label": row["choice_label"],
                "picks": row["picks"],
                "answers": row["answers"],
            }
            for row in regret.choice_share_rows(responses, randomized_only=True)
        ],
        "tasks": [
            {"key": task.task, "label": task.label}
            for task in regret.TASK_ARMS
            if task.task in {row["task"] for row in scored}
        ],
        "outcomes": [
            {"key": key, "label": analytics.TLX_LABELS.get(key, key), "better": "down"}
            for key in ("temporal_demand", "effort")
        ]
        + [{"key": "confidence", "label": analytics.TLX_LABELS["confidence"], "better": "up"}],
    }


def export_adjusted() -> dict:
    """The mixed-effects fit: one adjusted estimate per outcome.

    Needs the optional ``analysis`` extra. Seeded
    (:data:`behavior_graphs.viz.streamlit.user_study.mixed_effects._SEED`), so two
    runs on one store agree to the digit.
    """
    from behavior_graphs.viz.streamlit.user_study import claims, mixed_effects, regret

    responses = _responses()
    rates = regret.measured_success_rates()
    rows = claims.adjusted_rating_claims(mixed_effects.continuous_effects(responses))
    decision = claims.adjusted_decision_claim(
        mixed_effects.decision_regret_effect(responses, rates)
    )
    if decision is not None:
        rows.append(decision)
    if not rows:
        raise SystemExit("The model fit returned no estimate.")
    return {"rows": rows}


# ── progress: the benchmark's metric files ───────────────────────────────────


def _progress_roots(given: str | None) -> list[Path]:
    raw = given or os.environ.get("PROGRESS_RESULTS_ROOT", "") or "progress_results"
    return [Path(part).expanduser() for part in raw.split(":") if part.strip()]


def _domain(dataset: str) -> str | None:
    for label, prefixes in PROGRESS_DOMAINS:
        if dataset.startswith(prefixes):
            return label
    return None


def export_progress(roots: Iterable[Path]) -> tuple[dict, dict]:
    """``(metrics section, curves document)`` over the benchmark's results.

    Two returns because they go to two files: the metrics are a table a person
    checks against the paper, and the curves are 42,000 numbers a chart
    consumes. See :mod:`behavior_graphs.results` for why that matters.

    Only datasets every compared method scored are counted. A mean over
    "whatever each method happened to be run on" is not a comparison, and the
    coverage is uneven in exactly the way that matters: the VLM baselines were
    run on a subset of the zoo.

    A single metric can be missing from a dataset that otherwise has all four
    methods -- AUROC is undefined on a collection whose rollouts all succeeded
    -- so the means are taken per metric over the datasets that define it,
    which is why a domain's AUROC can rest on fewer datasets than its VOC.
    """
    per_dataset: dict[str, dict[str, dict[str, float]]] = {}
    for root in _progress_roots(None) if roots is None else roots:
        for path in sorted(Path(root).glob("*_metrics.json")):
            dataset = path.name[: -len("_metrics.json")]
            if dataset.startswith(PROGRESS_EXCLUDED) or _domain(dataset) is None:
                continue
            blob = json.loads(path.read_text())
            if not any(isinstance(value, dict) for value in blob.values()):
                # The flat, single-method convention: the method is in the file
                # name, and every method this table compares is in the nested
                # files the benchmark runner writes.
                continue
            slot = per_dataset.setdefault(dataset, {})
            for method, metrics in blob.items():
                if method not in PROGRESS_METHODS:
                    continue
                slot[method] = {
                    key: value
                    for key in PROGRESS_METRICS
                    if (value := _metric(metrics.get(key))) is not None
                }
    complete = sorted(
        dataset
        for dataset, methods in per_dataset.items()
        if all(method in methods for method in PROGRESS_METHODS)
    )
    if not complete:
        raise SystemExit(
            "No dataset here was scored by every compared method. Set "
            "PROGRESS_RESULTS_ROOT to the benchmark's results directory."
        )

    rows = [
        {
            "dataset": dataset,
            "domain": _domain(dataset),
            "method": method,
            **per_dataset[dataset][method],
        }
        for dataset in complete
        for method in PROGRESS_METHODS
    ]
    domains = [
        {"label": label, "datasets": sum(1 for d in complete if _domain(d) == label)}
        for label, _ in PROGRESS_DOMAINS
    ]
    return {
        "methods": [
            {"key": key, "label": PROGRESS_METHOD_LABELS[key], "ours": key.startswith("graph")}
            for key in PROGRESS_METHODS
        ],
        "metrics": [
            {
                "key": key,
                "label": key.upper() if key != "kendall" else "Kendall",
                "higher_better": PROGRESS_HIGHER_BETTER[key],
            }
            for key in PROGRESS_METRICS
        ],
        "domains": [domain for domain in domains if domain["datasets"]],
        "datasets": len(complete),
        "rows": rows,
        "means": _progress_means(rows),
    }, _curve_bands(list(roots), complete)


def _curve_bands(roots: Sequence[Path], datasets: Sequence[str]) -> dict:
    """Per collection, the predicted progress curves of individual rollouts.

    One entry per rollout, carrying every method's curve resampled onto the
    shared grid, plus the rollout's index and outcome so a page can find its
    video. Progress is a per-rollout quantity and this keeps it one: an
    earlier version of this function exported a median across rollouts with a
    10-to-90 band around it, and both were quantities the method never
    produces -- the estimator is deterministic and gives exactly one curve per
    rollout, so the band was spread between *rollouts* in the costume of a
    confidence interval and the median was an average of curves nobody ever
    sees.

    A sample rather than all of them: there are 7,682 rollouts across the 43
    collections, and LIBERO alone has 1,000 in one. :data:`CURVE_ROLLOUTS` is
    the cap, filled with the first failures and the first successes by rollout
    index, so which rollouts appear is reproducible and can be said in a
    clause.
    """
    import numpy as np

    from behavior_graphs.progress_estimation.io import load_curves, resolve_curve_path

    grid = np.linspace(0.0, 1.0, CURVE_GRID)
    out = []
    for dataset in datasets:
        per_method: dict[str, dict[int, list[float]]] = {}
        flags: dict[int, bool] = {}
        indices: list[int] = []
        for method in PROGRESS_METHODS:
            path = next(
                (p for root in roots if (p := resolve_curve_path(Path(root), dataset, method))),
                None,
            )
            if path is None:
                continue
            loaded = load_curves(path)
            episode_ids = [int(i) for i in np.asarray(loaded.idx).ravel()]
            outcomes = np.asarray(loaded.flags, dtype=bool)
            slot = per_method.setdefault(method, {})
            for position, raw in enumerate(loaded.curves):
                values = np.asarray(raw, dtype=float).ravel()
                # A one-frame rollout has no curve to interpolate, and a
                # non-finite one cannot be drawn.
                if values.size < 2 or not np.all(np.isfinite(values)):
                    continue
                episode = episode_ids[position]
                source = np.linspace(0.0, 1.0, values.size)
                slot[episode] = [round(float(v), 3) for v in np.interp(grid, source, values)]
                flags[episode] = bool(outcomes[position])
                if episode not in indices:
                    indices.append(episode)
        if not per_method:
            continue
        # Only rollouts every method scored: a chart missing one of its four
        # lines invites reading the gap as a result.
        complete = sorted(i for i in indices if all(i in slot for slot in per_method.values()))
        failures = [i for i in complete if not flags[i]][: CURVE_ROLLOUTS // 2]
        successes = [i for i in complete if flags[i]][: CURVE_ROLLOUTS // 2]
        chosen = sorted(failures + successes)
        out.append(
            {
                "dataset": dataset,
                "domain": _domain(dataset),
                "episodes": len(indices),
                "successes": sum(1 for i in indices if flags.get(i)),
                "rollouts": [
                    {
                        "idx": index,
                        "success": flags[index],
                        "curves": {m: per_method[m][index] for m in per_method},
                    }
                    for index in chosen
                ],
            }
        )
    return {"grid": [round(float(v), 4) for v in grid], "datasets": out}


def _metric(value: object) -> float | None:
    """One metric, from either ``[mean, std]`` or a bare scalar.

    ``None`` where the benchmark wrote a non-finite value: AUROC is undefined
    on a rollout collection with one outcome class, and averaging a NaN in
    erases the whole column.
    """
    if value is None:
        return None
    number = float(value[0]) if isinstance(value, list | tuple) else float(value)
    return None if math.isnan(number) or math.isinf(number) else number


def _progress_means(rows: Sequence[dict]) -> list[dict]:
    """Per-domain and overall means, weighted by dataset.

    Every dataset counts once regardless of how many episodes it holds, which
    is what makes a column comparable across a zoo of very differently sized
    collections.
    """
    out = []
    groups = [label for label, _ in PROGRESS_DOMAINS] + ["Average"]
    for label in groups:
        subset = [row for row in rows if label == "Average" or row["domain"] == label]
        if not subset:
            continue
        for method in PROGRESS_METHODS:
            scored = [row for row in subset if row["method"] == method]
            cell: dict[str, object] = {
                "domain": label,
                "method": method,
                "datasets": len(scored),
            }
            for metric in PROGRESS_METRICS:
                values = [row[metric] for row in scored if metric in row]
                cell[metric] = statistics.fmean(values) if values else None
                cell[f"{metric}_datasets"] = len(values)
            out.append(cell)
    return out


# ── assembly ─────────────────────────────────────────────────────────────────


def build(
    sections: Sequence[str], existing: dict, progress_roots: list[Path]
) -> tuple[dict, dict | None]:
    """``(document, curves or None)`` — the two files, for the sections asked for."""
    out = dict(existing)
    out["schema"] = SCHEMA
    curves = None
    if "policies" in sections:
        out["policies"] = export_policies()
    if "study" in sections:
        out["study"] = export_study()
    if "adjusted" in sections:
        out["adjusted"] = export_adjusted()
    if "progress" in sections:
        out["progress"], curves = export_progress(progress_roots)
        curves["schema"] = SCHEMA
    return out, curves


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_PATH, help="where to write (default: %(default)s)"
    )
    parser.add_argument(
        "--sections",
        default="all",
        help=f"comma-separated subset of {','.join(SECTIONS)} (default: all)",
    )
    parser.add_argument(
        "--progress-root",
        default=None,
        help="colon-separated dirs holding {dataset}_metrics.json (default: $PROGRESS_RESULTS_ROOT)",
    )
    parser.add_argument(
        "--curves",
        type=Path,
        default=None,
        help=f"where the progress curves go (default: {CURVES_PATH.name})",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help=(
            "print the merged document instead of writing it, for a run inside a "
            "container. Writes no curves file"
        ),
    )
    args = parser.parse_args(argv)

    requested = (
        SECTIONS
        if args.sections == "all"
        else tuple(part.strip() for part in args.sections.split(",") if part.strip())
    )
    unknown = [name for name in requested if name not in SECTIONS]
    if unknown:
        parser.error(f"unknown section(s): {', '.join(unknown)}; pick from {','.join(SECTIONS)}")

    existing = json.loads(args.out.read_text()) if args.out.is_file() else {}
    document, curves = build(requested, existing, _progress_roots(args.progress_root))
    text = json.dumps(document, indent=1, sort_keys=True) + "\n"
    if args.stdout:
        print(text, end="")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} ({len(text)} bytes): {', '.join(requested)}", file=sys.stderr)
    if curves is not None:
        # One line per series rather than one per number: nobody reads this by
        # eye, and pretty-printed it was six times the size of the document
        # next to it. `separators` drops the space after every comma, which on
        # 42,000 numbers is 42,000 bytes.
        curve_path = args.curves or CURVES_PATH
        curve_text = json.dumps(curves, sort_keys=True, separators=(",", ":")) + "\n"
        curve_path.parent.mkdir(parents=True, exist_ok=True)
        curve_path.write_text(curve_text)
        print(f"wrote {curve_path} ({len(curve_text)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
