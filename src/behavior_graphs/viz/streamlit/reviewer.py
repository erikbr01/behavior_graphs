"""What the anonymous reviewer page offers, and which run it draws.

The page (``apps/reviewer/Home.py``) draws graphs in two places: the study's
three tasks, in the walkthrough of how a graph is built, and any dataset at all
in the explorer at the bottom. Everything either of them needs to decide *what*
to draw lives here, away from Streamlit, because the decisions are the part
worth testing: which datasets are grouped under which family, which K values
exist for one of them, and which of a dataset's ~170 swept runs is the one to
show.

:func:`study_graphs` is the three the study used, in its own order, so the
walkthrough shows the displays the paper reports on rather than re-tuned
lookalikes. The results beside them come from ``behavior_graphs.results``;
everything here is live off the mounted runs.

The rule throughout: **a predefined config wins, and nothing is chosen
automatically where one exists.** A session yaml under ``configs/viz/study/``
pins a run for its dataset — which fixes the run, and with it K — and the
study's three stages are the ones that matter most, being what the participants
read. Only a dataset no config speaks for gets a run and a K picked for it.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_log = logging.getLogger(__name__)

#: The study variant whose stages seed the page. Same default as the figure
#: exports, so the trees on both match the ones the paper reports.
DEFAULT_VARIANT = "three_stage_v1"

#: How the graph is drawn when the dataset is not one the study used, and when
#: the study's own configuration cannot be read at all. The controls for
#: *rebuilding* the graph are all off — a reviewer wants the graph, and the
#: knobs a researcher reaches for are on the demo app.
DISPLAY_FALLBACK: dict = {
    "view_mode": "native_svg",
    "min_branch": 2,
    "color_mode": "outcome",
    # The clips go under the graph rather than beside it, so the graph gets the
    # full width — it is the thing being shown.
    "video_panel": "below",
    "show_paths": False,
    "show_stats": False,
    "show_fold_toggles": False,
    # On, unlike the study: the timeline under each clip, banded and labelled by
    # behavior, is what shows a reader that the graph's decomposition lines up
    # with what the robot is doing — the claim the page exists to let them check.
    # The study withheld it because a participant reading the segmentation off
    # the video is no longer reading it off the graph, which is the thing that
    # study measured; a reviewer is not being measured.
    "show_behavior_annotations": True,
    "show_camera_labels": False,
    "graph_height_extra": 90,
}

#: What a ``configs/data/`` group is called on the page. A group with no entry
#: here shows its file name, which is also what the demo app's picker shows, so
#: a benchmark added without touching this is unlabelled rather than missing.
FAMILY_LABELS: dict[str, str] = {
    "droid": "Real robot — DROID (Franka)",
    "prepare_tray": "Real robot — prepare tray",
    "safe": "Real robot — SAFE (WidowX, Franka)",
    "libero": "LIBERO",
    "mimicgen": "MimicGen",
    "robomimic": "RoboMimic",
    "vla_foundry": "VLA Foundry",
    "dino_scan": "DINO scan",
}


def family_label(group: str) -> str:
    """The page's name for a ``configs/data/`` group."""
    return FAMILY_LABELS.get(group, group)


#: The keys a pinned config supplies to this page: the ones that decide what the
#: tree *is*. The rest of a session's ``graph:`` block is chrome — where the
#: clips go, which badges are drawn — and that is the page's own business, not
#: something a study session should reach in and set on a page it knows nothing
#: about. Today the two agree (the study's blocks turn the same badges off that
#: :data:`DISPLAY_FALLBACK` does), which is exactly when a boundary is cheap to
#: draw.
CONFIG_KEYS = ("view_mode", "min_branch", "color_mode", "graph_height_extra")


@dataclass(frozen=True)
class PinnedGraph:
    """One graph a config pins: a dataset, a run, and how to draw it.

    *task* is the sweep task dir the run lives under, which is also the key the
    dataset selector and ``configs/data/`` use (see ``data.py::_effective_task``)
    — so a pinned config resolves to an option in the dataset dropdown rather
    than to a name nothing else knows.
    """

    task: str
    label: str
    run_dir: Path
    graph_params: dict
    #: The session's own ``task_description``, or "" where it has none. What
    #: the robot is supposed to do, in the words the study showed participants
    #: -- a graph of a task a reader cannot name is a graph of nothing.
    description: str = ""


def pinned_graphs(
    rows: Sequence[dict] = (), variant_id: str = DEFAULT_VARIANT
) -> list[PinnedGraph]:
    """Every graph a config pins, most authoritative first.

    The study variant's stages lead, in the order the variant lists them: those
    are the graphs the participants read and the paper reports, so they outrank
    everything. Then the remaining session yamls under ``configs/viz/study/``,
    by filename — each pins a run for its dataset, which is a decision somebody
    made on purpose and this page should not silently re-make. Where two
    sessions pin the same dataset (``transport_mh_jan28`` and its ``_k10``
    sibling), the first by that order wins and the other is ignored.

    Graph parameters are merged the way the study merges them — the variant's
    ``graph:`` block, then the session's own on top, then anything the stage
    overrides — because a tree drawn at a different ``min_branch`` from the one
    the participants read is a different tree.

    *rows* is the run index, and it is how a pin learns which dataset it is for:
    the index has already worked that out per run (``data.py::_effective_task``),
    and the two layouts it handles do not agree on where the name is.
    ``sweeps/<dataset>/<hash>`` has it in the parent, which is where the study's
    three stages keep it; ``runs/<name>`` — the older sessions — has it in the
    run dir itself, and reading the parent there yields the literal "runs".
    Without *rows* the parent is all there is to go on, which is right for the
    sweep layout and a wrong guess for the flat one; harmless either way, since
    a pin whose dataset is not in the index cannot be drawn at all.

    Never raises: a checkout without ``configs/viz/`` still gets a working page,
    just one that picks every run itself.
    """
    try:
        from behavior_graphs.core.yaml_io import load_yaml
        from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
        from behavior_graphs.viz.streamlit.user_study.sessions import (
            resolve_run_dir,
            sessions_dir,
        )
    except ImportError as e:  # pragma: no cover — a partial install, not a config error
        _log.warning("cannot import the study configuration (%s)", e)
        return []

    def read(path: Path) -> dict | None:
        try:
            return load_yaml(path.read_text()) or {}
        except (OSError, ValueError) as e:
            _log.warning("session config %s is unreadable (%s)", path, e)
            return None

    directory = sessions_dir()
    variant = get_variant(variant_id)
    if variant is None:
        _log.warning("no study variant %r — the page will not open on the study", variant_id)

    # The index's own answer for the dataset a run belongs to, by run dir name —
    # the name being a config hash for a swept run, and unique in practice.
    task_of = {Path(r["path"]).name: r["task"] for r in rows}

    out: list[PinnedGraph] = []
    seen: set[Path] = set()
    claimed: set[str] = set()

    def add(path: Path, label: str, extra: dict) -> None:
        seen.add(path)
        sess = read(path)
        if sess is None:
            return
        run_dir = resolve_run_dir(sess)
        if run_dir is None:
            _log.warning("session config %s names no run_dir", path)
            return
        # Never the session's own `task:` field: that is the session's name
        # ("set_table_contrastive") and no dataset is called that.
        task = task_of.get(run_dir.name) or run_dir.parent.name
        if task in claimed:
            # An earlier config already speaks for this dataset, and two graphs
            # of one dataset is a choice this page should not be making.
            _log.info("%s also pins %s; keeping the earlier config's graph", path.name, task)
            return
        claimed.add(task)
        out.append(
            PinnedGraph(
                task=task,
                label=label,
                run_dir=run_dir,
                graph_params={
                    **((variant.graph_params if variant else {}) or {}),
                    **(sess.get("graph") or {}),
                    **extra,
                },
                description=" ".join(str(sess.get("task_description") or "").split()),
            )
        )

    for stage in variant.stages if variant else ():
        add(
            directory / f"{stage.session_name}.yaml",
            stage.label or stage.task,
            stage.graph_params or {},
        )
    for path in sorted(directory.glob("*.yaml")) if directory.is_dir() else ():
        if path not in seen:
            add(path, path.stem, {})
    return out


def study_graphs(rows: Sequence[dict] = (), variant_id: str = DEFAULT_VARIANT) -> list[PinnedGraph]:
    """Only the graphs the study's own stages pin, in the order it lists them.

    :func:`pinned_graphs` returns these first and then every other session yaml,
    which is the right order for *choosing a run* and the wrong list for a page
    that wants to say "these are the three tasks the study used". Taking the
    first three of that list would say it by position, and would go on saying it
    after a fourth session yaml was added or the variant went missing.

    Empty where the variant cannot be read, which is a page that should say so
    rather than show three arbitrary graphs under a sentence about the study.
    """
    try:
        from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
    except ImportError as e:  # pragma: no cover - a partial install
        _log.warning("cannot import the study registry (%s)", e)
        return []
    variant = get_variant(variant_id)
    if variant is None:
        _log.warning("no study variant %r, so no study graphs", variant_id)
        return []
    # Matched on the label rather than on ``task``: a pin's ``task`` is the
    # *dataset* the run index resolved ("cooking_baseline_dit_200"), while a
    # stage names the task ("cooking"), so the two never compare equal. The
    # label is what ``pinned_graphs`` copies off the stage, and every other
    # session yaml is labelled with its file stem instead.
    wanted = [stage.label or stage.task for stage in variant.stages]
    by_label = {graph.label: graph for graph in pinned_graphs(rows, variant_id)}
    # A stage whose run is not mounted here has no pin at all and is left out.
    return [by_label[label] for label in wanted if label in by_label]


def display_params(task: str, pinned: Sequence[PinnedGraph]) -> dict:
    """How to draw *task* — a pinned config's settings over the page's defaults.

    A config's settings follow its dataset to every K, not just the one it
    pinned: ``prepare_tray`` is read at ``min_branch: 1`` because that is what
    makes its tree legible, and that does not stop being true one notch along
    the K slider.

    Only :data:`CONFIG_KEYS` are taken from it — see there.
    """
    for graph in pinned:
        if graph.task == task:
            return {
                **DISPLAY_FALLBACK,
                **{k: v for k, v in graph.graph_params.items() if k in CONFIG_KEYS},
            }
    return dict(DISPLAY_FALLBACK)


def families_with_runs(families: dict[str, list[str]], tasks: set[str]) -> dict[str, list[str]]:
    """*families* restricted to datasets that have a run, empty families dropped.

    ``configs/data/`` describes what exists somewhere; the run index describes
    what this deployment can actually draw. Offering the difference is offering
    a dropdown entry that resolves to "no run matches".
    """
    out: dict[str, list[str]] = {}
    for group, names in families.items():
        present = [n for n in names if n in tasks]
        if present:
            out[group] = present
    return out


def orphan_tasks(families: dict[str, list[str]], tasks: set[str]) -> list[str]:
    """Tasks with runs that no family claims.

    A dataset swept but never added to a ``configs/data/`` group would otherwise
    be unreachable from a page whose only route to a dataset is through a
    family. The demo app's flat picker has no such hole, which is why this is
    needed here and not there.
    """
    claimed = {name for names in families.values() for name in names}
    return sorted(tasks - claimed)


def k_values(task: str, rows: list[dict]) -> list[int]:
    """The K values *task* has a run for, ascending."""
    return sorted({int(r["k"]) for r in rows if r["task"] == task and r["k"] is not None})


def preferred_k(
    task: str,
    rows: list[dict],
    pinned: Sequence[PinnedGraph] = (),
    best_per_k: dict[str, dict[int, dict]] | None = None,
) -> int | None:
    """The K to show for *task* when nobody has chosen one.

    **A predefined config decides it wherever there is one** — the K of the run
    that config pins, which for the study's three tasks is the graph the
    participants read and the paper reports. K is only chosen automatically for
    a dataset no config speaks for, and then it is the best-scoring K of the ones
    swept: a judgement the model-selection pass has already made and recorded.

    Deliberately not the smallest, which is what an ascending list's first
    element gives you: K=5 on a dataset nobody chose it for is a coarse graph
    presented as the intended one.
    """
    for graph in pinned:
        if graph.task == task and (k := _pinned_k(graph, rows)) is not None:
            return k
    ks = k_values(task, rows)
    scored = (best_per_k or {}).get(task) or {}
    if best := [k for k in ks if k in scored]:
        return max(best, key=lambda k: (scored[k].get("score", -1), k))
    return ks[0] if ks else None


def _pinned_k(graph: PinnedGraph, rows: list[dict]) -> int | None:
    """The K of the run *graph* pins, read off the index, or ``None``.

    ``None`` covers the pins this deployment cannot serve: the older sessions
    name runs under ``$DATA_PATH/runs``, and a stack scanning only
    ``/data/sweeps`` has none of them indexed. Those datasets fall through to
    the automatic choice, which is the right answer — the pinned run is not
    there to be drawn.
    """
    task_rows = [r for r in rows if r["task"] == graph.task and r["k"] is not None]
    found = _named_run(str(graph.run_dir), task_rows)
    if found is None:
        return None
    return next(int(r["k"]) for r in task_rows if r["path"] == str(found))


def resolve_run(
    task: str,
    k: int,
    rows: list[dict],
    best_per_k: dict[str, dict[int, dict]] | None = None,
    pinned: Sequence[PinnedGraph] = (),
) -> Path | None:
    """The run to draw for *task* at *k*, or ``None`` if there is none.

    The run a config pins wherever there is one, so a reviewer who has not
    touched the controls is looking at the tree the participants read and the
    paper reports, not a run that merely shares its dataset and K.

    Otherwise the best-scoring run of that (task, K): K is the reviewer's choice
    and the rest is not something to make them tune. That comes from the
    model-selection pass by path rather than by hyperparameters, because the
    hyperparameters do not identify a run — ``set_table_baseline_pretrained_dit``
    has 1 368 runs at K=11, and matching on the handful of params the picker
    exposes lands on an arbitrary one of them. That is not a cosmetic
    difference: the run the study pinned carries ``node_labels.json``, and most
    of its 1 367 twins do not, so the tree comes out labelled "B6" instead of
    with the behavior the node is.

    Falls back to any run at that K, preferring one that finished — a crashed
    job retried under a new hash leaves two runs with identical hyperparameters,
    one of them missing ``metrics.json``.
    """
    candidates = [r for r in rows if r["task"] == task and r["k"] == k]
    if not candidates:
        return None
    for graph in pinned:
        if graph.task == task and (found := _named_run(str(graph.run_dir), candidates)):
            return found
    best = (best_per_k or {}).get(task, {}).get(k) or {}
    if winner := best.get("path"):
        if found := _named_run(winner, candidates):
            return found
    complete = [r for r in candidates if r.get("has_metrics")]
    return Path((complete or candidates)[0]["path"])


def _named_run(wanted: str, candidates: list[dict]) -> Path | None:
    """*wanted* among *candidates*, matched by path or, failing that, by name.

    By name as well because a run's recorded path is not always the path this
    process sees it at: ``best_per_k.csv`` is written by the sweep on the host
    (``/mnt/…/data/sweeps/…``) and read by an app that has the same tree
    bind-mounted somewhere else (``/data/sweeps/…``), so a strict comparison
    never matches in the deployment and always matches in a test. The name is a
    hash of the run's config, and the candidates are already one dataset at one
    K, so it identifies the run on its own.
    """
    by_path = {r["path"]: r for r in candidates}
    if wanted in by_path:
        return Path(wanted)
    name = Path(wanted).name
    for row in candidates:
        if Path(row["path"]).name == name:
            return Path(row["path"])
    return None


def default_selection(
    pinned: Sequence[PinnedGraph],
    families: dict[str, list[str]],
    rows: list[dict],
    best_per_k: dict[str, dict[int, dict]] | None = None,
) -> tuple[str, str, int] | None:
    """``(family, dataset, K)`` the page opens on, or ``None`` if nothing loads.

    The first pinned graph this deployment can draw — the study's first stage,
    normally — at the K of the run it pins, so the landing view is the tree the
    paper describes. Anything unreadable (no variant, the pinned dataset not in
    this deployment) falls through to the first dataset of the first family, at
    whatever K :func:`preferred_k` likes for it.
    """
    for graph in pinned:
        k = _pinned_k(graph, rows)
        if k is None:
            continue
        for group, names in families.items():
            if graph.task in names:
                return group, graph.task, k
    for group, names in families.items():
        for name in names:
            if (k := preferred_k(name, rows, pinned, best_per_k)) is not None:
                return group, name, k
    return None
