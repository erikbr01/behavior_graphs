"""Progress Estimation — cross-dataset metric table + per-episode curve browser.

Two views over the artifacts ``ProgressRunner`` writes:

  ``{dataset}_metrics.json``            {method: {metric: [mean, std]}} over folds
  ``{dataset}_{infix}_curves.npz``      per-episode predicted curves (+ flags, idx)

**Metrics table** — one row per method, one column per metric. The dataset
selector is a MULTI-select: one dataset shows its across-fold spread, several
aggregate over exactly that subset. The aggregate is an unweighted mean over datasets (each dataset counts once
regardless of episode count, which is what makes the column comparable across a
zoo of very differently-sized sets) and the ± is then the spread ACROSS datasets,
not the fold spread. Best cell per metric is bolded using each metric's own
``higher_better``, so MAE is not silently ranked upside-down.

**Episode browser** — pick a dataset and episode, play the rollout video, and see
every method's predicted curve underneath with a cursor at the scrubbed frame.

Two honest limitations, both surfaced in the UI:

* The clip and the curves share one iframe
  (:func:`~behavior_graphs.viz.streamlit.components.video_curve_player`), which is
  what lets a ``timeupdate`` listener move a cursor line ON the curves. A
  ``st.plotly_chart`` renders in a component of its own that the video's script
  cannot reach, which is why this used to be a separate progress bar under the
  video instead. Click the plot to seek.
* Methods predict at different native resolutions (the graph labels sliding
  WINDOWS, the MLP labels FRAMES), and the npz does not record the window
  geometry needed to expand them. Curves are therefore plotted against *fraction
  of episode* rather than frame index, which keeps them mutually comparable
  without inventing an alignment.

Environment:
    PROGRESS_RESULTS_ROOT    colon-separated dirs holding the artifacts above
                             (default ./progress_results)
    PROGRESS_ROLLOUTS_ROOT   colon-separated dirs holding <dataset>/ rollout dirs,
                             used for videos and episode metadata
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from behavior_graphs.progress_estimation.io import (
    METHOD_REGISTRY,
    _episode_mp4,
    canonical_method,
    discover_curve_methods,
    load_curves,
    resolve_curve_path,
)
from behavior_graphs.progress_estimation.metrics import (
    MAE,
    VOC,
    KendallTau,
    Pearson,
    SuccessAUROC,
)
from behavior_graphs.progress_estimation.report import chose_k_off_eval, latex_escape
from behavior_graphs.viz.streamlit.components.latex_export import latex_export_panel
from behavior_graphs.viz.streamlit.components.video_curve_player import video_curve_player

_METRICS = [VOC(), KendallTau(), Pearson(), MAE(), SuccessAUROC()]
_HIGHER_BETTER = {m.name: m.higher_better for m in _METRICS}

# Trivial reference curves, NOT methods. `frame_index` is normalized time and
# `dummy` is the harness no-op, so on all-success demos both score a perfect
# VOC/Kendall/Pearson/MAE -- the ranking metrics are saturated by a stopwatch.
# They are shown (they are the bar every real method has to clear) but excluded
# from the best-per-metric highlight, which would otherwise always name a
# baseline as the winner.
_BASELINES = {"frame_index", "dummy"}

#: Dataset families the page does not list, matching the sweep zoo's own
#: exclusion. The SAFE real-robot sets carry no video, so no VLM baseline can
#: run on them and any table including them compares two methods on one subset
#: and four on another. Their graph/MLP artifacts are still on disk.
_HIDDEN_DATASET_PREFIXES = ("safe_franka", "safe_widowx")

#: Dataset rubrics for the grouped table below: a family name and the dataset
#: prefixes that belong to it. A rubric rather than a per-dataset row because
#: the zoo is 40-odd sets and the question the table answers is per FAMILY --
#: and because a `droid` family exists nowhere in the artifacts: the real-robot
#: sets are named by task (`set_table`, `prepare_tray`, `cooking`), so anything
#: that reads them off the filenames scatters them across three rows.
#:
#: "Real-world" rather than "droid": these are rollouts from our own
#: DROID-style Franka setup, and a row labelled DROID reads as the public DROID
#: dataset -- which none of this is, and which a reader would reasonably assume.
#:
#: `safe_*` is absent for the reason `_HIDDEN_DATASET_PREFIXES` gives: those
#: sets carry no video, so no VLM baseline can be run on them and a row for
#: them could only ever hold two of the four methods. `transport_mh_salvage250`
#: is robomimic transport in sim despite the name, so it belongs in `robomimic`
#: rather than in a long-horizon bucket of its own.
_RUBRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("libero", ("libero",)),
    ("Real-world", ("set_table", "prepare_tray", "cooking")),
    ("robomimic", ("robomimic", "transport_mh_salvage250")),
    ("mimicgen", ("mimicgen",)),
)

#: The four the rubric table compares, fixed rather than taken from the method
#: picker: this table's whole point is one apples-to-apples comparison, so it
#: only counts datasets scored by ALL of them. `robometer_fft` is left out
#: because it is a LIBERO finetune and would flatter itself on the libero row;
#: `graph_absorb` because it answers a different question (P(absorb@SUCCESS))
#: and is not a progress estimate.
_RUBRIC_METHODS = ("graph_stage", "mlp", "robometer_4b", "topreward_ll")

#: Metric keys as they should be SET, not as they are keyed. The metric classes
#: name themselves in lower case because the name is a dict key and a filename
#: stem; three of these are acronyms and reading "voc"/"mae"/"auroc" in a paper
#: table is wrong rather than merely informal.
_METRIC_LABELS = {
    "voc": "VOC",
    "kendall": "Kendall",
    "pearson": "Pearson",
    "mae": "MAE",
    "auroc": "AUROC",
}

#: The rubric column's header, and the index level's name with it.
#:
#: "Source", because every row names where the rollouts came from -- three sim
#: benchmarks and our own robot -- and it is the one short word honest about all
#: four: `Benchmark` is wrong for the real-world row, and `Suite` is LIBERO's
#: own word for goal/object/spatial, so it reads as a grouping *within* LIBERO.
#:
#: "(#datasets)" rather than "(#tasks)", in the header rather than only in the
#: caption, because the parenthetical is otherwise a puzzle to a reader who
#: looks at the table alone -- and because on three of the four rows datasets
#: and tasks are not the same number: the 19 real-world sets are 3 tasks
#: (set_table, prepare_tray, cooking) with data and policy variants, the 10
#: robomimic ones are 5 tasks in {ph, mh}, and a LIBERO suite is 10 tasks
#: apiece. Datasets is also the unit the means weight by, so the column's
#: arithmetic matches the caption's total.
_RUBRIC_INDEX_NAME = "Source (#datasets)"

#: Shown by default in the rubric tables. Pearson is offered but not
#: preselected: measured over the 43 datasets x 4 methods the table counts, it
#: correlates with VOC at r = 0.989 and picks a different winning method on 1
#: dataset out of 43, so it is very nearly VOC printed twice. What it could
#: uniquely contribute -- sensitivity to the curve's SHAPE rather than its
#: order -- is what MAE already measures directly. Kendall stays: it is the
#: least redundant of the three (r = 0.961 with VOC) and the only one that
#: changes a conclusion, picking graph-stage on LIBERO where VOC and Pearson
#: both pick the MLP. Re-adding Pearson is one click and recomputes nothing.
_RUBRIC_DEFAULT_METRICS = ("voc", "kendall", "mae", "auroc")

#: Methods that are this repo's own, marked "(ours)" in the rubric tables. By
#: prefix rather than a list of one, because `graph_absorb` and the demo-fit
#: variant are ours too and a table that marked only `graph_stage` would imply
#: the others are somebody else's.
_OURS_PREFIX = "graph"


#: Not offered at all. `topreward` is this repo's in-house reimplementation,
#: kept only long enough to establish that the vendored one matched; the
#: faithful `topreward_ll` (the authors' own code, byte-identical core) is the
#: one that gets reported, so showing both invites reading the wrong row.
_HIDDEN_METHODS = {"topreward"}

#: Preselected: the graph estimator, the MLP, both ROBOMETER variants,
#: TOPReward, and normalized time as the reference curve on the plot. The MLP
#: belongs in the default view rather than a click away -- it reads the SAME
#: policy embedding the graph does, so it is the comparison that separates the
#: graph's structure from its representation, which graph-vs-VLM alone cannot.
#: Everything else the artifacts carry -- graph-absorb, warp-rm -- stays opt-in.
#: Baselines are plotted but never tabled, so `frame_index` here only affects
#: the episode browser.
_DEFAULT_METHODS = (
    "graph_stage_demofit",
    "graph_stage",
    "mlp",
    "robometer_4b",
    "robometer_fft",
    "topreward_ll",
    "frame_index",
)

#: Readable names for artifact prefixes that are pure abbreviation. The prefix
#: is what the files are named and stays the value everywhere -- this only
#: changes what is drawn. Anything not listed passes through unchanged, so a new
#: dataset is never given a label someone guessed at.
_DATASET_NAMES = {
    "lg_pool500": "libero_goal (500-episode pool)",
    "libero_object_pool500": "libero_object (500-episode pool)",
    "libero_spatial_pool500": "libero_spatial (500-episode pool)",
    "sqmh": "robomimic_square_mh",
    "trmh": "robomimic_transport_mh",
}


def dataset_name(prefix: str) -> str:
    """Label for a dataset prefix; the prefix itself if it is already readable."""
    return _DATASET_NAMES.get(prefix, prefix)


def split_note(splits: dict[str, dict], picked: list[str]) -> str:
    """How the selected datasets were split, in one sentence.

    Read off each run's own record rather than assumed, because the runs in one
    results dir do not agree: the robomimic pair is k-fold (every episode
    predicted once, by a model fit on the others) while the LIBERO pools are
    ALIGNED (a fixed eval set, the graph fit on a disjoint pool). Either way the
    numbers are scored on data the model did not fit.
    """
    rows = [splits[d] for d in picked if d in splits]
    if not rows:
        return ""
    eps = sum(r.get("n_eps") or 0 for r in rows)
    succ = sum(r.get("n_succ") or 0 for r in rows)
    over = "" if len(rows) == 1 else f" over {len(rows)} datasets"
    counts = f"{eps} eval episodes{over} ({succ} success / {eps - succ} failure)"

    schemes = {r.get("scheme") for r in rows}
    if schemes == {"aligned"}:
        fit = sum(r.get("n_fit") or 0 for r in rows)
        return f"Aligned split, graph fit on {fit} held-out episodes; {counts}."
    if schemes == {"demofit"}:
        fit = sum(r.get("n_fit") or 0 for r in rows)
        # Every rollout is scored, and that is not leakage: the graph was fit on
        # DEMONSTRATIONS, which are disjoint from the rollouts by construction.
        # K is still chosen on a validation split of those demos.
        return (
            f"Demo-fit, graph fit on {fit} training demonstrations and scored on "
            f"all {counts.replace('eval episodes', 'rollout episodes')}; "
            f"disjoint from the fit data by construction, so no fold is held back."
        )
    if schemes == {"kfold"}:
        folds = {r.get("folds") for r in rows if r.get("folds")}
        scheme = f"{folds.pop()}-fold CV" if len(folds) == 1 else "cross-validation"
        # Whether K was chosen on an inner split or on the fold being reported
        # is the difference between a held-out number and an optimistic one, so
        # it belongs in the caption rather than in a commit message.
        vf = {r.get("val_frac") for r in rows if r.get("val_frac")}
        if all(r.get("selection") == "nested" for r in rows) and len(vf) == 1:
            how = f" K chosen on a {vf.pop():.0%} validation split of each fold's train half."
        else:
            how = ""
        return f"{scheme}, scored on the held-out fold; {counts}.{how}"
    # A selection spanning both: say so rather than picking one label, since the
    # two are not the same experiment and the mean runs across them.
    n_al = sum(1 for r in rows if r.get("scheme") == "aligned")
    return (
        f"Mixed splits -- {n_al} aligned, {len(rows) - n_al} k-fold -- so the "
        f"mean runs across two designs; {counts}."
    )


def method_name(method: str) -> str:
    """Registry display name for a method key, or the key if it is unknown."""
    info = METHOD_REGISTRY.get(canonical_method(method))
    return info.display if info else method


def _roots(env: str, default: str) -> list[Path]:
    raw = os.environ.get(env, "").strip() or default
    return [Path(p).expanduser() for p in raw.split(":") if p.strip()]


def _as_pair(value) -> tuple[float, float]:
    """(mean, std) from either ``[mean, std]`` or a bare scalar."""
    if isinstance(value, list | tuple):
        mean = float(value[0])
        return mean, (float(value[1]) if len(value) > 1 else 0.0)
    return float(value), 0.0


def _split_method(stem: str) -> tuple[str, str] | None:
    """Split ``sim_bottles_topreward_ll`` -> ``("sim_bottles", "topreward_ll")``.

    Needed because the single-method writers put the method in the FILENAME and
    the json itself is a flat ``{metric: value}`` map, so the method name exists
    nowhere else. Candidate tokens come from the method registry (keys plus their
    historical infixes); longest match wins so ``robometer_fft`` is not truncated
    to ``robometer``.
    """
    cands: list[tuple[str, str]] = []
    for key, info in METHOD_REGISTRY.items():
        for token in (key, *getattr(info, "infixes", ())):
            cands.append((token.replace(":", "_"), key))
    for token, key in sorted(cands, key=lambda t: -len(t[0])):
        suffix = "_" + token
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], key
    return None


@st.cache_data(show_spinner=False)
def load_metrics(roots: tuple[str, ...]) -> dict[str, dict]:
    """{dataset: {method: {metric: (mean, std)}}} over every results root.

    Handles both on-disk conventions. ``ProgressRunner`` writes a NESTED
    ``{dataset}_metrics.json`` holding ``{method: {metric: [mean, std]}}``, while
    the single-method scripts write a FLAT ``{dataset}_{method}_metrics.json``
    holding ``{metric: scalar}`` with no fold spread. Reading the flat form with
    the nested reader silently treats metric names as method names, so the two
    are distinguished by whether any top-level value is a dict.

    Later roots win on collision, so a local override dir can shadow a shared one.
    """
    out: dict[str, dict] = {}
    for root in roots:
        for path in sorted(Path(root).glob("*_metrics.json")):
            stem = path.name[: -len("_metrics.json")]
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                st.warning(f"skipping {path}: {exc}")
                continue
            if any(isinstance(v, dict) for v in blob.values()):
                dataset, per_method = stem, blob
            else:
                split = _split_method(stem)
                if split is None:
                    st.warning(
                        f"skipping {path.name}: flat metrics file whose method suffix "
                        "is not in METHOD_REGISTRY, so the method is unidentifiable"
                    )
                    continue
                dataset, method = split
                per_method = {method: blob}
            slot = out.setdefault(dataset, {})
            for method, per_metric in per_method.items():
                slot[method] = {
                    k: _as_pair(v) for k, v in per_metric.items() if k in _HIGHER_BETTER
                }
    return out


#: `sweep_progress_allds.py` names the graph schemes it stores by; the metrics
#: are bare scalars per (dataset, K) rather than a mean and a fold spread.
_SWEEP_SCHEMES = ("graph_direct:stage", "graph_direct:absorb", "frame_index")

#: The two split headers a benchmark run writes into `{dataset}_report.md`, and
#: they are not the same experiment. K-fold predicts every episode once from a
#: model fit on the others; ALIGNED fits on a disjoint pool and predicts a fixed
#: eval set, which is the stricter of the two. Both are in this results dir --
#: the robomimic pair is k-fold, the LIBERO pools are aligned -- so the caption
#: has to read it rather than assume one.
_SPLIT_ALIGNED = re.compile(
    r"ALIGNED eval on (\d+) eps \([^)]*?(\d+)succ/(\d+)fail\), graph fit on (\d+) held-out"
)
# `[^)]*` before the close paren: the runner appends `, eval=M` after the fold
# count, and anchoring on `)` immediately after `-fold` matched nothing.
_SPLIT_KFOLD = re.compile(r"\((\d+) eps, (\d+)succ/(\d+)fail(?:, (\d+)-fold)?[^)]*\)")


@st.cache_data(show_spinner=False)
def sweep_leakage(roots: tuple[str, ...]) -> dict[str, int]:
    """``{run: how many datasets were dropped for selecting on the eval fold}``.

    Counted rather than silently skipped: a run that vanishes looks like a
    missing mount, and the reader should know the numbers exist and why they
    are not shown.
    """
    dropped: dict[str, int] = {}
    for root in roots:
        base = Path(root)
        for path in sorted([*base.glob("sweep_results.json"), *base.glob("*/sweep_results.json")]):
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            n = sum(
                1
                for entry in blob.values()
                if isinstance(entry, dict) and "per_k" in entry and not chose_k_off_eval(entry)
            )
            if n:
                dropped[path.parent.name] = n
    return dropped


@st.cache_data(show_spinner=False)
def load_sweep_runs(roots: tuple[str, ...]) -> dict[str, dict]:
    """``{run name: {dataset: {method: {metric: (mean, std)}}}}`` for each sweep.

    ``sweep_progress_allds.py`` writes one ``sweep_results.json`` per run rather
    than the per-dataset metrics/curves pair, which is why its datasets could
    not be displayed here at all. Its numbers ARE per-dataset metrics, just
    nested under the chosen K, so they are lifted into the same shape the rest
    of the page reads. There are no curves in it -- the episode browser cannot
    follow.

    **A dataset whose K was chosen on the eval fold is dropped.** Early runs of
    that script maximised stage VOC over K across the held-out folds and then
    reported the maximum, so the published number was the best of five readings
    of the data it called unseen. Those entries are on disk and are not
    displayable: a table cannot carry both kinds of number and stay a table, and
    a footnote is not enough when the figure is what gets quoted. Runs whose
    entry says it nested the choice (``selection: nested``, or the ``best_k:
    "selected"`` the nested path writes) load normally. :func:`sweep_leakage`
    counts what was dropped so the page can say so.
    """
    runs: dict[str, dict] = {}
    for root in roots:
        base = Path(root)
        for path in sorted([*base.glob("sweep_results.json"), *base.glob("*/sweep_results.json")]):
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                st.warning(f"skipping {path}: {exc}")
                continue
            per_dataset: dict[str, dict] = {}
            for dataset, entry in blob.items():
                if not isinstance(entry, dict) or "per_k" not in entry:
                    continue  # skipped or errored during the sweep
                if not chose_k_off_eval(entry):
                    continue  # K chosen on the eval fold; see the docstring
                # per_k keys are ints in memory, strings once through JSON.
                per_k = {str(k): v for k, v in entry["per_k"].items()}
                best = per_k.get(str(entry.get("best_k")))
                if not best:
                    continue
                per_dataset[dataset] = {
                    scheme: {m: _as_pair(v) for m, v in vals.items() if m in _HIGHER_BETTER}
                    for scheme, vals in best.items()
                    if scheme in _SWEEP_SCHEMES
                }
            if per_dataset:
                runs[path.parent.name] = per_dataset
    return runs


@st.cache_data(show_spinner=False)
def load_demofit_runs(roots: tuple[str, ...]) -> tuple[dict[str, dict], dict[str, dict]]:
    """``({dataset: {method: {metric: (mean, std)}}}, {dataset: split})`` for demo-fit.

    A different EXPERIMENT, not a different method: the graph is fit on the
    policy's own training demonstrations and then scores every rollout, instead
    of being fit on a held-out fold of the rollouts. `demo_fit_progress.py`
    records one `demo_fit_results.json` per output dir rather than the
    per-dataset metrics files the benchmark writes, so its numbers are lifted
    into the shape the rest of the page reads.

    Several such dirs are merged into ONE view (DROID and robomimic/mimicgen
    were run separately), because the question the table answers -- how does
    demo-fitting compare to k-fold, on the datasets where demos exist -- spans
    both. There is no fold spread to report: one fit, so std is absent rather
    than zero.
    """
    metrics: dict[str, dict] = {}
    splits: dict[str, dict] = {}
    for root in roots:
        base = Path(root)
        for path in sorted(
            [*base.glob("demo_fit_results.json"), *base.glob("*/demo_fit_results.json")]
        ):
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                st.warning(f"skipping {path}: {exc}")
                continue
            for dataset, entry in blob.items():
                if not isinstance(entry, dict):
                    continue
                cells = {
                    k: _as_pair(v)
                    for k, v in (entry.get("ordering_on_successes") or {}).items()
                    if k in _HIGHER_BETTER and v is not None
                }
                # A one-sided rollout set has no detection number to report --
                # `nut_assembly_d0` never succeeded, `can_ph` never failed -- so
                # the cell is left absent and renders as an em dash rather than
                # as a 0 that would drag the column mean down.
                auroc = entry.get("detection_auroc")
                if auroc is not None and np.isfinite(auroc):
                    cells["auroc"] = _as_pair(auroc)
                if not cells:
                    continue
                metrics.setdefault(dataset, {})["graph_stage_demofit"] = cells
                splits[dataset] = {
                    "n_eps": entry.get("n_rollouts"),
                    "n_succ": entry.get("n_succ"),
                    "n_fail": entry.get("n_fail"),
                    "folds": None,
                    "scheme": "demofit",
                    "n_fit": entry.get("n_demos"),
                }
    return metrics, splits


@st.cache_data(show_spinner=False)
def load_splits(roots: tuple[str, ...]) -> dict[str, dict]:
    """``{dataset: {n_eps, n_succ, n_fail, folds}}`` from each run's own report.

    Parsed rather than assumed: the benchmark writes the header
    "sqmh (50 eps, 41succ/9fail, 3-fold)" beside its metrics, and that is the
    only place the fold count is recorded.
    """
    out: dict[str, dict] = {}
    # A run written by bench_allds_with_vlms.py keeps its own index, which
    # records the counts, the fold count, and how K was chosen -- strictly more
    # than the report header carries, and already structured. Read first so it
    # wins over the header parse below.
    for root in roots:
        for path in sorted(Path(root).glob("index_*.json")):
            try:
                blob = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for dataset, e in blob.items():
                if not isinstance(e, dict) or "n_eps" not in e:
                    continue
                out.setdefault(
                    dataset,
                    {
                        "scheme": "kfold",
                        "folds": e.get("folds"),
                        "n_fit": None,
                        "n_eps": e.get("n_eps"),
                        "n_succ": e.get("n_succ"),
                        "n_fail": e.get("n_fail"),
                        "selection": e.get("selection"),
                        "val_frac": e.get("val_frac"),
                    },
                )
    for root in roots:
        for path in sorted(Path(root).glob("*_report.md")):
            try:
                head = path.read_text().split("\n", 1)[0]
            except OSError:
                continue
            aligned = _SPLIT_ALIGNED.search(head)
            kfold = None if aligned else _SPLIT_KFOLD.search(head)
            if aligned:
                eps, succ, fail, fit = aligned.groups()
                entry = {"scheme": "aligned", "folds": None, "n_fit": int(fit)}
            elif kfold:
                eps, succ, fail, folds = kfold.groups()
                entry = {
                    "scheme": "kfold",
                    "folds": int(folds) if folds else None,
                    "n_fit": None,
                }
            else:
                continue
            entry |= {"n_eps": int(eps), "n_succ": int(succ), "n_fail": int(fail)}
            out.setdefault(path.name[: -len("_report.md")], entry)
    return out


@st.cache_data(show_spinner=False)
def sweep_splits(roots: tuple[str, ...], run: str) -> dict[str, dict]:
    """The same, for a sweep run: counts come from each dataset's own entry."""
    out: dict[str, dict] = {}
    for root in roots:
        for path in [Path(root) / run / "sweep_results.json", Path(root) / "sweep_results.json"]:
            if not path.is_file() or path.parent.name != run:
                continue
            title = path.parent / "sweep_table.md"
            m = (
                re.search(r"(\d+)-fold", title.read_text(errors="ignore")[:400])
                if title.is_file()
                else None
            )
            folds = int(m.group(1)) if m else None
            for dataset, entry in json.loads(path.read_text()).items():
                if isinstance(entry, dict) and "per_k" in entry:
                    out.setdefault(
                        dataset,
                        {
                            "n_eps": entry.get("n_eps"),
                            "n_succ": entry.get("n_succ"),
                            "n_fail": entry.get("n_fail"),
                            # Not in the per-dataset entry; the run stamps it into
                            # its table title instead ("..., 3-fold").
                            "folds": folds,
                            "scheme": "kfold",
                            "n_fit": None,
                        },
                    )
    return out


@st.cache_data(show_spinner=False)
def find_curves_dir(roots: tuple[str, ...], dataset: str) -> str | None:
    """First results root that holds any ``{dataset}_*_curves.npz``."""
    for root in roots:
        if any(Path(root).glob(f"{dataset}_*_curves.npz")):
            return str(root)
    return None


@st.cache_data(show_spinner=False)
def find_rollout_dir(roots: tuple[str, ...], dataset: str) -> str | None:
    """``<root>/<dataset>`` that looks like a rollout dir (has episode metadata)."""
    for root in roots:
        cand = Path(root) / dataset
        if (cand / "episodes" / "metadata.yaml").is_file():
            return str(cand)
    return None


@st.cache_data(show_spinner=False)
def load_episode_meta(rollout_dir: str) -> list[dict]:
    import yaml

    meta = yaml.safe_load((Path(rollout_dir) / "episodes" / "metadata.yaml").read_text())
    return list(meta.get("episodes", []))


@st.cache_data(show_spinner=False)
def load_method_curves(curves_dir: str, dataset: str, methods: tuple[str, ...]) -> dict:
    """{method: {episode_idx: curve}} for the methods that actually have a file."""
    out: dict[str, dict[int, np.ndarray]] = {}
    for method in methods:
        path = resolve_curve_path(Path(curves_dir), dataset, method)
        if path is None:
            continue
        c = load_curves(path)
        # `idx` holds ORIGINAL episode ids: the runner concatenates per-fold eval
        # splits, so position in the array is not the episode number.
        out[method] = {int(i): np.asarray(v, float) for i, v in zip(c.idx, c.curves)}
    return out


@st.cache_data(show_spinner=False)
def video_path(rollout_dir: str, idx: int, success: bool | None) -> str | None:
    """This episode's clip, flat layout or per-camera.

    Delegates to the library resolver so the page cannot drift from what the
    methods read. Sim dirs write ``media/ep0000_success.mp4``; DROID dirs write
    ``media/<camera>/ep0000_success.mp4``, and searching only the flat form left
    the browser with no video on all 13 real-robot datasets.
    """
    mp4 = _episode_mp4(Path(rollout_dir), idx, success)
    return str(mp4) if mp4 else None


def dataset_value(d: dict, method: str, metric: str):
    """One dataset's single (mean, std) for ``method``/``metric``, or None.

    One dataset can hold the SAME method under two spellings, because
    `runner._write_reports` merges rather than overwrites `{dataset}_metrics.json`
    -- a set scored by an older scheme-keyed run and re-scored by the current
    runner carries both `graph_direct:stage` and `graph_stage`. Flattening over
    `d.items()` then made that dataset contribute TWICE to a mean documented as
    "each dataset counts once", and showed whichever spelling happened to come
    first (the older one). Prefer the key that IS the registry name, so the
    newest write wins deterministically.

    Module level because both tables aggregate over datasets and the
    two-spellings rule has to be the same in each, or the rubric table and the
    method table would disagree about what one dataset contributed.
    """
    cands = [
        (k, v[metric])
        for k, v in d.items()
        if canonical_method(k) == method and metric in v and np.isfinite(v[metric][0])
    ]
    if not cands:
        return None
    cands.sort(key=lambda kv: kv[0] != method)  # exact registry key first
    return cands[0][1]


def metric_frame(
    metrics: dict[str, dict], datasets: list[str], methods: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str | None]]:
    """Method x metric table over one or several datasets, as NUMBERS.

    Returns ``(mean, std, best)``: two float frames sharing an index, and the
    winning row label per metric. Numbers rather than pre-formatted strings
    because ``st.dataframe`` sorts a column by its underlying values, and a
    column of ``"0.812+-0.041"`` sorts lexicographically -- ``"0.9"`` above
    ``"0.12"``, and a bolded ``"**0.812**"`` above every unbolded number,
    since ``*`` precedes the digits. :func:`styled_frame` renders them for the screen and
    the shared LaTeX panel renders them for export, both from these numbers, so
    the two cannot drift.

    With a single dataset the +- is the across-FOLD spread the runner recorded.
    With several it is the spread ACROSS the selected datasets, from an unweighted
    mean -- each dataset counts once regardless of episode count, which is what
    makes the number comparable over very differently-sized sets. A method or
    metric missing from a dataset simply does not contribute.
    """
    sub = {d: metrics[d] for d in datasets if d in metrics}
    single = len(sub) == 1
    rows: dict[str, dict] = {}
    # `frame_index` is normalized time and `dummy` is the harness no-op. They
    # saturate every ranking metric on all-success sets, which says more about
    # the metric than about any method, so they are not rows here. The episode
    # browser still plots them, where a reference curve is worth seeing.
    # `methods` holds canonical keys; metrics.json may name a method by scheme.
    # Grouped by CANONICAL method, not by the key each file happens to use. One
    # method is spelled two ways across the runs mounted here -- the older
    # artifacts key the graph rows by scheme (`graph_direct:stage`), the newer
    # ones by method (`graph_stage`) -- so selecting datasets from both used to
    # build two rows that then carried the SAME display label. A duplicated
    # index broke the bolding (the winner's label matched two rows) and made the
    # method picker look like it was doing nothing.
    scored_methods = sorted(
        {
            canonical_method(m)
            for d in sub.values()
            for m in d
            if canonical_method(m) in methods and canonical_method(m) not in _BASELINES
        }
    )

    for method in scored_methods:
        cells = {}
        for metric in _HIGHER_BETTER:
            pairs = [
                pair for d in sub.values() if (pair := dataset_value(d, method, metric)) is not None
            ]
            if not pairs:
                cells[metric] = None
            elif single:
                cells[metric] = (float(pairs[0][0]), float(pairs[0][1]), 1)
            else:
                vals = [float(v[0]) for v in pairs]
                cells[metric] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        rows[method] = cells

    # Bold the winner per metric, honouring each metric's direction.
    best: dict[str, str | None] = {}
    for metric, higher in _HIGHER_BETTER.items():
        scored = [(m, c[metric][0]) for m, c in rows.items() if c.get(metric)]
        if scored:
            best[metric] = (max if higher else min)(scored, key=lambda kv: kv[1])[0]

    means, stds, labels = {}, {}, {}
    for method, cells in rows.items():
        # metrics.json may key a row by scheme (`graph_direct:stage`); the
        # registry is keyed by method. Without this the row is labelled with the
        # raw scheme string instead of its tag.
        # The same label the Methods picker draws, so a row and a selection are
        # recognisably the same thing.
        label = method_name(method)
        labels[method] = label
        # NaN, not a placeholder string: it keeps the column numeric, and both
        # renderers turn it back into the em dash the reader sees.
        means[label] = {m: (cells[m][0] if cells.get(m) else np.nan) for m in _HIGHER_BETTER}
        # `n == 1` with std 0 means one contributing dataset and no spread to
        # show, which is not the same as a spread that happens to be zero.
        stds[label] = {
            m: (cells[m][1] if cells.get(m) and (cells[m][2] > 1 or cells[m][1]) else np.nan)
            for m in _HIGHER_BETTER
        }
    mean_frame = pd.DataFrame(means).T.astype(float)
    std_frame = pd.DataFrame(stds).T.astype(float)
    # Named, or Streamlit draws the index under a blank header and the table
    # looks like it has no method column at all.
    mean_frame.index.name = std_frame.index.name = "method"
    best_labels = {m: (labels[w] if w else None) for m, w in best.items()}
    return mean_frame, std_frame, best_labels


@st.cache_data(show_spinner=False)
def fold_counts(roots: tuple[str, ...]) -> tuple[int, ...]:
    """How many folds each run was scored over, from its own report header.

    For the caption, so the number in a paper is the number the artifacts were
    written with rather than one typed in once and left. `ProgressRunner` puts
    it in the report's first line (`... 1000 eps, 3 fold(s), eval=334`) and
    nowhere machine-readable, which is why this parses prose.
    """
    found: set[int] = set()
    for root in roots:
        for path in sorted(Path(root).glob("*_report.md")):
            head = path.read_text(errors="ignore").split("\n", 1)[0]
            match = re.search(r"(\d+) fold\(s\)", head)
            if match:
                found.add(int(match.group(1)))
    return tuple(sorted(found))


def rubric_method_label(method: str) -> str:
    """:func:`method_name`, with our own methods marked."""
    label = method_name(method)
    return f"{label} (ours)" if method.startswith(_OURS_PREFIX) else label


def rubric_datasets(
    metrics: dict[str, dict], methods: tuple[str, ...] = _RUBRIC_METHODS
) -> list[tuple[str, list[str], list[str]]]:
    """``[(rubric, complete, dropped)]`` — which datasets each rubric counts.

    "Complete" means scored by every method in *methods*: the rubric table and
    the averages under it have to be computed over the same sets, or the
    averages would not be the averages of the rows above them.
    """
    out = []
    for label, prefixes in _RUBRICS:
        in_rubric = [
            d
            for d in sorted(metrics)
            if d.startswith(prefixes) and not d.startswith(_HIDDEN_DATASET_PREFIXES)
        ]
        complete = [
            d
            for d in in_rubric
            if all(
                any(dataset_value(metrics[d], m, metric) is not None for metric in _HIGHER_BETTER)
                for m in methods
            )
        ]
        out.append((label, complete, [d for d in in_rubric if d not in complete]))
    return out


def rubric_average_frame(
    metrics: dict[str, dict],
    methods: tuple[str, ...] = _RUBRIC_METHODS,
    *,
    by_rubric: bool = False,
) -> tuple[pd.DataFrame, int, int]:
    """One row per method, averaged over everything the rubric table counts.

    *by_rubric* picks what gets equal weight. Dataset-weighted is the page's
    convention elsewhere, but the rubrics here are wildly uneven — 19 real-robot
    sets against 6 LIBERO ones — so a dataset-weighted mean is largely a
    statement about the biggest family, and the rubric-weighted one is the
    average of the blocks as drawn. Neither is wrong; which one is meant is a
    choice, so it is a control rather than a default nobody sees.

    Returns ``(frame, n_datasets, n_rubrics)``.
    """
    selected = [
        (label, complete) for label, complete, _ in rubric_datasets(metrics, methods) if complete
    ]
    every = [d for _, complete in selected for d in complete]
    rows: dict[str, dict[str, float]] = {}
    for method in methods:
        cells: dict[str, float] = {}
        for metric in _HIGHER_BETTER:
            if by_rubric:
                per_rubric = [
                    np.mean(vals)
                    for _, complete in selected
                    if (
                        vals := [
                            pair[0]
                            for d in complete
                            if (pair := dataset_value(metrics[d], method, metric)) is not None
                        ]
                    )
                ]
                cells[metric] = float(np.mean(per_rubric)) if per_rubric else float("nan")
            else:
                vals = [
                    pair[0]
                    for d in every
                    if (pair := dataset_value(metrics[d], method, metric)) is not None
                ]
                cells[metric] = float(np.mean(vals)) if vals else float("nan")
        rows[rubric_method_label(method)] = cells
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "method"
    return frame, len(every), len(selected)


def rubric_frame(
    metrics: dict[str, dict], methods: tuple[str, ...] = _RUBRIC_METHODS
) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    """Rubric x method table: one block of rows per family, one row per method.

    Only datasets scored by EVERY method in *methods* contribute. A mean over
    "whatever each method happened to be run on" is not a comparison, and the
    coverage is uneven in exactly the way that matters: the VLM baselines were
    run on a subset of the zoo, so an unfiltered mean would put the graph's
    score on 40 datasets beside ROBOMETER's on 20 and invite reading the gap as
    a result.

    Returns ``(frame, counted, skipped)``: the numbers, how many datasets each
    rubric contributed, and the datasets that matched a rubric but were dropped
    for missing a method — surfaced in the UI rather than silently absent.
    """
    rows: dict[tuple[str, str], dict[str, float]] = {}
    counted: dict[str, int] = {}
    skipped: list[str] = []
    for label, complete, dropped in rubric_datasets(metrics, methods):
        skipped += dropped
        counted[label] = len(complete)
        # A rubric no dataset here satisfies is not a row: four rows of dashes
        # per absent family reads as "measured and empty" rather than "not in
        # these results", and on a root holding one family it would be three
        # empty blocks under the one real one.
        if not complete:
            continue
        for method in methods:
            cells: dict[str, float] = {"datasets": float(len(complete))}
            for metric in _HIGHER_BETTER:
                vals = [
                    pair[0]
                    for d in complete
                    if (pair := dataset_value(metrics[d], method, metric)) is not None
                ]
                cells[metric] = float(np.mean(vals)) if vals else float("nan")
            rows[(f"{label} ({len(complete)})", rubric_method_label(method))] = cells
    columns = ["datasets", *_HIGHER_BETTER]
    frame = pd.DataFrame(
        list(rows.values()) or None,
        index=pd.MultiIndex.from_tuples(rows or [("", "")], names=[_RUBRIC_INDEX_NAME, "method"])
        if rows
        else pd.MultiIndex.from_arrays([[], []], names=[_RUBRIC_INDEX_NAME, "method"]),
        columns=columns,
    )
    # A count, not a measurement: as a float it reaches the LaTeX export as
    # `6.000`, and the on-screen formatter cannot fix what the exporter reads.
    frame["datasets"] = frame["datasets"].astype(int)
    return frame, counted, skipped


def _cell_text(mean: float, std: float) -> str:
    """One cell, shared by the screen and the LaTeX export."""
    if mean != mean:
        return "—"
    return f"{mean:.3f}" if std != std else f"{mean:.3f}±{std:.3f}"


def styled_frame(mean: pd.DataFrame, std: pd.DataFrame, best: dict[str, str | None]):
    """The same table for the screen: float values, real bold, sortable.

    The +- has to be baked in per cell because pandas' column formatters see
    only the value, never its row -- but the frame underneath stays float, so
    clicking a header sorts by the number rather than by its rendering.
    """
    styler = mean.style
    for row in mean.index:
        for metric in mean.columns:
            spread = std.loc[row, metric]
            styler = styler.format(
                lambda v, s=spread: _cell_text(v, s), subset=(row, metric), na_rep="—"
            )
    return styler.apply(
        lambda col: ["font-weight: bold" if best.get(col.name) == row else "" for row in col.index],
        axis=0,
    )


st.title("Progress Estimation")

results_roots = _roots("PROGRESS_RESULTS_ROOT", "./progress_results")
rollout_roots = _roots("PROGRESS_ROLLOUTS_ROOT", "./runs")
bench_metrics = load_metrics(tuple(str(p) for p in results_roots))
sweeps = load_sweep_runs(tuple(str(p) for p in results_roots))
# Said out loud rather than left as an absence: a sweep whose datasets were all
# dropped disappears from the picker, which looks like a missing mount.
_leaked = sweep_leakage(tuple(str(p) for p in results_roots))
if _leaked:
    st.caption(
        "Not shown: "
        + ", ".join(f"`{run}` ({n} dataset{'s' if n != 1 else ''})" for run, n in _leaked.items())
        + " — those runs chose K by scoring every candidate on the held-out "
        "folds and reporting the best, so the number is a maximum over the data "
        "it calls unseen. Re-run `scripts/sweep_progress_allds.py`, which now "
        "only ever nests the choice."
    )

#: What a benchmark run is called in the Results picker, versus a sweep's dir.
_BENCH_RUN = "benchmark runs (per-episode)"
#: The demo-fit experiment, shown as its own results source. Named for what makes
#: it different from `_BENCH_RUN`: what the graph was FIT on, not what it scored.
_DEMOFIT_RUN = "demo-fit (graph fit on training demos)"

demofit_metrics, demofit_splits = load_demofit_runs(tuple(str(p) for p in results_roots))

run_options = (
    ([_BENCH_RUN] if bench_metrics else [])
    + ([_DEMOFIT_RUN] if demofit_metrics else [])
    + sorted(sweeps)
)
if len(run_options) > 1:
    run_choice = st.selectbox(
        "Results",
        run_options,
        key="pe_run",
        help="A benchmark run stores per-episode curves, so it can be browsed "
        "episode by episode. A cross-dataset sweep stores one metric set per "
        "dataset -- many more datasets, no curves to play against. Only sweeps "
        "that chose their hyperparameters off the eval fold are listed.",
    )
else:
    run_choice = run_options[0] if run_options else None

is_demofit = run_choice == _DEMOFIT_RUN
# A sweep has no per-episode curves; demo-fit DOES write them, so it stays
# browsable and only the sweep branch disables the episode player.
is_sweep = run_choice is not None and run_choice not in (_BENCH_RUN, _DEMOFIT_RUN)
if is_demofit:
    # Restricted to datasets that HAVE demos, and carrying the k-fold rows for
    # those same datasets alongside the demo-fit row -- the comparison is the
    # point, so a table with only one method in it would answer nothing.
    metrics = {ds: {**bench_metrics.get(ds, {}), **cells} for ds, cells in demofit_metrics.items()}
    # The demo-fit split describes the demo-fit row; where a dataset also has a
    # k-fold row the caption would be describing two designs at once, which
    # `split_note` already refuses to average over silently.
    splits = demofit_splits
elif is_sweep:
    metrics = sweeps[run_choice]
    splits = sweep_splits(tuple(str(p) for p in results_roots), run_choice)
else:
    metrics = bench_metrics
    splits = load_splits(tuple(str(p) for p in results_roots))

if not metrics:
    st.warning(
        "No `*_metrics.json` found under "
        + ", ".join(f"`{p}`" for p in results_roots)
        + ". Point `PROGRESS_RESULTS_ROOT` at a progress-benchmark output dir."
    )
    st.stop()

# ------------------------------------------------------------------ metrics table
st.header("Metrics")
all_datasets = [d for d in sorted(metrics) if not d.startswith(_HIDDEN_DATASET_PREFIXES)]
# A 69-entry multiselect is unusable for "all of them", and Streamlit has no
# select-all of its own; the checkbox drives the list rather than sitting beside
# it, so the two cannot disagree about what is shown.
use_all = st.checkbox("All datasets", value=True, key="pe_all_datasets")
picked_widget = st.multiselect(
    "Datasets",
    all_datasets,
    default=all_datasets,
    format_func=dataset_name,
    disabled=use_all,
    key="pe_metric_datasets",
    help="Pick one to see its across-fold spread, or several to aggregate over "
    "exactly that subset (unweighted mean; ± becomes the spread across datasets).",
)
picked = all_datasets if use_all else picked_widget
# One method selection for the whole page: the table and the episode browser are
# two views of the same comparison, and a picker each meant reading a metric off
# one and a curve off the other without noticing they showed different methods.
# Options come from every dataset, not just the picked ones, so changing dataset
# does not silently reset the choice.
# Both halves of the artifacts, canonicalised to one spelling: metrics.json
# names methods by scheme (`graph_direct:stage`) and the curve files by infix
# (`graph_direct_stage`), and a method can appear in only one of the two.
all_methods = sorted(
    (
        {canonical_method(m) for d in metrics.values() for m in d}
        | {
            m
            for ds in metrics
            for cd in [find_curves_dir(tuple(str(p) for p in results_roots), ds)]
            if cd
            for m in discover_curve_methods(Path(cd), ds)
        }
    )
    - _HIDDEN_METHODS,
    key=method_name,
)
default_methods = [m for m in _DEFAULT_METHODS if m in all_methods] or all_methods
chosen_methods = st.multiselect(
    "Methods",
    all_methods,
    default=default_methods,
    format_func=method_name,
    # Keyed by the results source, not a fixed string. A keyed widget ignores its
    # `default` once it holds state, so a single key meant switching source kept
    # the previous source's selection -- and the demo-fit row, which only exists
    # under the demo-fit source, never appeared until the reader ticked it by
    # hand. Different sources offer different methods, so a fresh widget per
    # source is also the honest behaviour. Table and browser still share it.
    key=f"pe_methods_{run_choice}",
    help="Applies to both the table and the episode browser. `frame_index` and "
    "`dummy` are reference curves: the browser plots them, the table never lists "
    "them, because they saturate every ranking metric on an all-success set.",
)
arrows = ", ".join(f"{m} {'↑' if h else '↓'}" for m, h in _HIGHER_BETTER.items())

if not picked or not chosen_methods:
    st.info("Select at least one dataset and one method to show the metrics table.")
else:
    mean_f, std_f, best_f = metric_frame(metrics, picked, tuple(chosen_methods))
    st.dataframe(styled_frame(mean_f, std_f, best_f), use_container_width=True)
    # What ± means is a property of the aggregation, so say it once and let the
    # on-screen caption and the exported LaTeX caption share it -- they used to
    # be worded separately, and the LaTeX one is the copy that ends up in a paper.
    # One dataset has only folds to vary over; several are pooled across.
    spread = "across folds" if len(picked) == 1 else "across datasets"
    st.caption(f"{arrows}. Cells are mean ± std {spread}. {split_note(splits, picked)}")

    # The mean_/std_ frame the exporter wants, so it can merge each pair into
    # one `$0.812 \\pm 0.041$` cell and bold the winner itself -- rather than
    # being handed strings with `**` already in them.
    tex_frame = pd.concat([mean_f.add_prefix("mean_"), std_f.add_prefix("std_")], axis=1)[
        [f"{p}{m}" for m in mean_f.columns for p in ("mean_", "std_")]
    ]
    # reset_index, because booktabs_table emits COLUMNS and never an index --
    # exporting the frame as-is produced a table of numbers with no method
    # names down the side.
    tex_frame = tex_frame.reset_index()
    latex_export_panel(
        tex_frame,
        [c for c in tex_frame.columns if c != "method"],
        filename=f"progress_{'-'.join(picked) if len(picked) <= 3 else str(len(picked)) + 'sets'}.tex",
        default_caption=(
            "Progress-estimation metrics on "
            + latex_escape(", ".join(dataset_name(d) for d in picked))
            + "."
        ),
        default_label="tab:progress-estimation",
        key="pe_tex",
    )

# ------------------------------------------------------------------ rubric table
# The same numbers as the table above, asked the other way round: that one is
# one row per method over a dataset selection, this one is one block per dataset
# FAMILY with a row per method inside it, so "how does each method do on the
# real-robot sets" is read off a block instead of by re-picking the selector.
st.header("By dataset rubric")

_rubric_mean, _rubric_counted, _rubric_skipped = rubric_frame(metrics)
# Which metrics to show. Five columns is already a wide table for a paper, and
# the three ranking metrics say nearly the same thing on these sets, so
# narrowing it is the usual case rather than a rare one. Selecting changes only
# which columns are drawn, never the numbers, so a metric can be added back
# without recomputing anything. Outside the branch below because a picker that
# appears and disappears with the data loses its selection.
_rubric_metrics = st.multiselect(
    "Metrics",
    list(_METRIC_LABELS),
    default=[m for m in _METRIC_LABELS if m in _RUBRIC_DEFAULT_METRICS],
    format_func=lambda m: _METRIC_LABELS[m],
    key="pe_rubric_metrics",
    help="Applies to the table and to its average block.",
)
# What the average block at the foot of the table gives equal weight to. The
# rubrics are very uneven -- 19 real-robot sets against 6 LIBERO ones -- so a
# dataset-weighted mean is largely a statement about the biggest family, while a
# rubric-weighted one is the average of the blocks as drawn. Neither is wrong;
# which is meant is a choice, so it is a control rather than a silent default.
_rubric_weight = st.radio(
    "Average weights each",
    ["dataset", "rubric"],
    horizontal=True,
    key="pe_rubric_weight",
    help="Dataset: each of the datasets counts once. Rubric: each block counts "
    "once, whatever its size.",
)
if not sum(_rubric_counted.values()):
    st.info(
        "No dataset in these results is scored by all four of "
        + ", ".join(rubric_method_label(m) for m in _RUBRIC_METHODS)
        + ", so there is nothing to compare rubric by rubric."
    )
elif not _rubric_metrics:
    # Not `st.stop()`: that would take the episode browser below off the page
    # as well, which is not what clearing a picker up here asks for.
    st.info("Pick at least one metric to show the rubric table.")
else:
    _shown = [_METRIC_LABELS[m] for m in _rubric_metrics]
    _by_rubric = _rubric_weight == "rubric"
    _avg, _n_datasets, _n_rubrics = rubric_average_frame(metrics, by_rubric=_by_rubric)
    # The average as the table's LAST BLOCK rather than a section of its own:
    # it is the row a reader looks for after the per-family ones, the midrule
    # above it already says it is not another family, and in LaTeX it comes out
    # as one more `\multirow` group for free.
    _avg_label = f"Average ({_n_rubrics} rubrics)" if _by_rubric else f"Average ({_n_datasets})"
    _avg_block = _avg.copy()
    _avg_block.index = pd.MultiIndex.from_product(
        [[_avg_label], list(_avg.index)], names=_rubric_mean.index.names
    )
    # No count column: it is already in the rubric label, and a column of
    # four identical numbers per block is four times the ink for it.
    _rubric_view = pd.concat(
        [_rubric_mean.rename(columns=_METRIC_LABELS), _avg_block.rename(columns=_METRIC_LABELS)]
    )[_shown]
    _better = {_METRIC_LABELS[m]: _HIGHER_BETTER[m] for m in _rubric_metrics}

    def _best_in_rubric(col: pd.Series) -> list[str]:
        """Bold the winning method WITHIN each rubric block, per metric.

        Per rubric rather than per column: a column spans every family, and
        bolding its single best cell would be comparing libero against droid,
        which is a difference in task difficulty rather than in method. Each
        metric's own direction, so MAE is not ranked upside-down, and any column
        that is not a scored metric is left alone.
        """
        if col.name not in _better:
            return [""] * len(col)
        pick = "idxmax" if _better[col.name] else "idxmin"
        winners = {
            getattr(group, pick)()
            for _, group in col.groupby(level=0, sort=False)
            if group.notna().any()
        }
        return ["font-weight: 700" if idx in winners else "" for idx in col.index]

    def _midrules(frame: pd.DataFrame) -> pd.DataFrame:
        """A rule above every rubric block but the first.

        A block is four rows and the only thing otherwise separating two of them
        is the rowspan cell, which a reader scanning the numbers is not looking
        at. This is the `\\midrule` booktabs would put there.
        """
        starts = ~frame.index.get_level_values(0).duplicated()
        rule = "border-top: 1px solid rgba(250, 250, 250, 0.35)"
        css = []
        for position, is_start in enumerate(starts):
            # Not above the first block: the header rule already closes the top.
            mark = rule if (is_start and position) else ""
            css.append([mark] * len(frame.columns))
        return pd.DataFrame(css, index=frame.index, columns=frame.columns)

    # HTML rather than `st.dataframe`, for the multirow: pandas emits a
    # `rowspan` for a sparsified MultiIndex, so a rubric is ONE cell spanning
    # its four method rows. `st.dataframe` repeats the label on every row
    # instead, which reads as four separate one-method groups.
    _styler = (
        _rubric_view.style.format({label: "{:.3f}" for label in _better}, na_rep="—")
        .apply(_best_in_rubric, axis=0)
        .apply(_midrules, axis=None)
        .set_table_styles(
            [
                {"selector": "", "props": [("border-collapse", "collapse"), ("width", "100%")]},
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "left"),
                        ("padding", "0.35rem 0.6rem"),
                        ("vertical-align", "middle"),
                        ("font-weight", "600"),
                    ],
                },
                {
                    "selector": "thead tr:last-child th",
                    "props": [("border-bottom", "1px solid rgba(250, 250, 250, 0.45)")],
                },
                {
                    "selector": "td",
                    "props": [
                        ("text-align", "right"),
                        ("padding", "0.35rem 0.6rem"),
                        ("font-variant-numeric", "tabular-nums"),
                    ],
                },
            ]
        )
    )
    st.markdown(_styler.to_html(sparse_index=True), unsafe_allow_html=True)
    _arrows = ", ".join(
        f"{_METRIC_LABELS[m]} {'↑' if _HIGHER_BETTER[m] else '↓'}" for m in _rubric_metrics
    )
    st.caption(
        f"{_arrows}. Unweighted mean over the datasets in each rubric, each counting "
        "once; bold is the best method within a rubric. Only datasets scored by "
        "**all four** methods are included — the VLM baselines were run on a "
        "subset of the zoo, and a mean over whatever each method happened to get "
        "is not a comparison. The count beside each rubric is how many "
        "datasets it contributed, and the last block averages over "
        + (f"the {_n_rubrics} blocks above it." if _by_rubric else f"all {_n_datasets} of them.")
    )
    if _rubric_skipped:
        with st.expander(f"{len(_rubric_skipped)} dataset(s) left out for missing a method"):
            st.write(
                ", ".join(f"`{d}`" for d in _rubric_skipped)
                + " — matched a rubric but is not scored by all four methods."
            )
    _outside = [
        d
        for d in sorted(metrics)
        if not d.startswith(_HIDDEN_DATASET_PREFIXES)
        and not any(d.startswith(prefixes) for _, prefixes in _RUBRICS)
    ]
    if _outside:
        with st.expander(f"{len(_outside)} dataset(s) in no rubric"):
            st.write(", ".join(f"`{d}`" for d in _outside))

    # reset_index so the rubric and the method are COLUMNS: booktabs_table
    # emits columns only, and `group_column` then spans the rubric one exactly
    # as the screen's rowspan does.
    _tex = _rubric_view.reset_index()
    latex_export_panel(
        _tex,
        [c for c in _tex.columns if c not in (_RUBRIC_INDEX_NAME, "method")],
        group_column=_RUBRIC_INDEX_NAME,
        filename="progress_by_rubric.tex",
        default_caption=(
            "Progress-estimation metrics over "
            f"{sum(_rubric_counted.values())} datasets in total."
            # The fold count comes from the runs, not from this line: a caption
            # that says "3" because someone typed 3 is wrong the first time the
            # benchmark is re-run with a different split. Silent when the runs
            # in view disagree, since one sentence cannot honestly cover both.
            + (
                f" Reported values are means over {_folds[0]} cross-validation folds."
                if len(_folds := fold_counts(tuple(str(p) for p in results_roots))) == 1
                else ""
            )
        ),
        default_label="tab:progress-by-rubric",
        key="pe_rubric_tex",
    )

# --------------------------------------------------------------- episode browser
st.header("Episode browser")

if is_sweep:
    st.info(
        f"`{run_choice}` is a cross-dataset sweep: it records each dataset's "
        "metrics and no per-episode curves, so there is nothing to browse. Pick a "
        "benchmark run above to play episodes."
    )
    st.stop()

# `all_datasets`, not `metrics`: the browser hides the same datasets the table
# does, or the two panels disagree about which datasets the page is about.
browsable = [d for d in all_datasets if find_rollout_dir(tuple(str(p) for p in rollout_roots), d)]
if not browsable:
    st.info(
        "No dataset under "
        + ", ".join(f"`{p}`" for p in rollout_roots)
        + " has `episodes/metadata.yaml`. Set `PROGRESS_ROLLOUTS_ROOT` to browse videos."
    )
    st.stop()

bdataset = st.selectbox("Dataset", browsable, format_func=dataset_name, key="pe_browse_dataset")
rollout_dir = find_rollout_dir(tuple(str(p) for p in rollout_roots), bdataset)
curves_dir = find_curves_dir(tuple(str(p) for p in results_roots), bdataset)
episodes = load_episode_meta(rollout_dir)

if not episodes:
    st.warning(f"`{rollout_dir}` lists no episodes.")
    st.stop()

# From the curve files, not from metrics.json: this view plots curves, and a
# method can have one without the other.
methods = (
    tuple(m for m in discover_curve_methods(Path(curves_dir), bdataset) if m in chosen_methods)
    if curves_dir
    else ()
)
curves = load_method_curves(curves_dir, bdataset, methods) if curves_dir else {}


def _ep_label(i: int) -> str:
    ep = episodes[i]
    ok = ep.get("success")
    return f"ep {ep.get('index', i):04d} · {'success' if ok else 'failure'} · {ep.get('length', '?')} steps"


# Only episodes something actually predicted on. A rollout dir is the whole pool
# (1000 for the LIBERO sets) while a benchmark pass scores a sample of it, so
# most of the list used to open on an empty chart.
scored = [
    i
    for i, ep in enumerate(episodes)
    if any(by_ep.get(int(ep.get("index", i))) is not None for by_ep in curves.values())
]
if not scored:
    st.warning(
        f"None of `{bdataset}`'s {len(episodes)} episodes has a curve from the selected "
        "methods. Select more methods, or check that the curve files cover this dataset."
    )
    st.stop()

pos = st.selectbox("Episode", scored, format_func=_ep_label, key="pe_browse_episode")
ep = episodes[pos]
ep_idx = int(ep.get("index", pos))
n_frames = int(ep.get("length") or 0)

# Whatever the page-wide picker chose that this dataset actually scored. The
# x axis is fraction of episode, not frame index -- see the module docstring.
shown = list(curves)

mp4 = video_path(rollout_dir, ep_idx, ep.get("success"))

if not shown:
    st.info("Select at least one method to plot its curve.")
    st.stop()

fig = go.Figure()
for method in shown:
    curve = curves.get(method, {}).get(ep_idx)
    if curve is None or not len(curve):
        continue
    info = METHOD_REGISTRY.get(canonical_method(method))
    # Normalized x: see the module docstring -- native resolutions differ.
    x = np.linspace(0.0, 1.0, len(curve))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=curve,
            name=method_name(method),
            mode="lines",
            line=dict(color=info.color if info else None, width=2),
            hoverinfo="skip",
        )
    )

# Ground truth on all-success demos is normalized time; drawing it makes the
# "is this estimator doing more than a stopwatch?" question visible at a glance.
fig.add_trace(
    go.Scatter(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        name="normalized time (GT ramp)",
        mode="lines",
        line=dict(color="#888888", width=1, dash="dot"),
        hoverinfo="skip",
    )
)
fig.update_layout(
    height=380,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(title="fraction of episode", tickformat=".0%", range=[0, 1]),
    yaxis=dict(title="predicted progress", range=[-0.02, 1.02]),
    legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    # No hover at all: the cursor is what this chart is read with, and a tooltip
    # chasing the pointer competes with it. Click still seeks.
    hovermode=False,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

# One iframe for the clip and the curves, so the playhead can be a line ON the
# curves rather than a separate bar under the video. Falls back to the plain
# chart when the episode has no clip to sync to.
played = video_curve_player(mp4, fig, key=f"pe_{bdataset}_{ep_idx}") if mp4 else False
if not played:
    if mp4:
        st.info("This episode's clip could not be served; showing the curves alone.")
    else:
        st.info("No mp4 for this episode; showing the curves alone.")
    st.plotly_chart(fig, use_container_width=True)

missing = [m for m in shown if curves.get(m, {}).get(ep_idx) is None]
if missing:
    st.caption(
        "No saved curve for this episode from: "
        + ", ".join(f"`{m}`" for m in missing)
        + ". A method only has curves for episodes that appeared in an eval fold."
    )
