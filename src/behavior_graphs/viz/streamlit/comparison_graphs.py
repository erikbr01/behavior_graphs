"""One behavior graph per policy, all built with the same hyperparameters.

Policy Comparison answers "is A better than B" with a number. This answers the
question that follows — *how* do they behave differently — with a picture per
policy. That only means anything when every picture came out of the same
pipeline configuration: a tree with K=6 next to a tree with K=20 differs mostly
in K, so the section drives one set of hyperparameters and looks up the run each
policy has at that setting. The selectors live in the sidebar, where the Graph
Demo keeps the same ones, leaving the page itself to the pictures.

The picture is the **trajectory tree** — the Graph Demo's default view, so a
policy's card and the page it links to show the same thing. Each card draws it as
a figure rather than a tool (no dragging, no click-to-inspect, no clip panel):
half a dozen inspectors stacked down a page is not a comparison, and the link is
one click from the page built for exploring one.

Matching a policy to its runs goes by name. A sweep lays its runs out as
``<sweep>/<dataset>/<job hash>``, so a run dir's parent is named after the
rollout dataset it was built from — which is exactly what ``data``'s run index
reports as the run's ``task``. That is a layout convention rather than a
recorded fact, so the run a policy resolves to is checked against the dataset
its ``metrics.json`` actually names before its graph is drawn
(:func:`resolve_run`); the convention holds across every run in the deployment's
sweep, but a hand-placed run dir, or the qualified ``parent/name`` keys
``rollout_outcomes`` mints to break a basename collision, would otherwise
silently draw one policy's graph under another's heading.

A tree is built from a run's window labels and window *metadata*, which is why
this reads them itself rather than calling
:func:`behavior_graphs.viz.streamlit.data.load_run`: that also loads the window
feature matrix, 37 MB a run in the deployment's sweep, and a tree never looks at
a feature. See :func:`tree_inputs`.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import streamlit as st

from behavior_graphs.core.types import EpisodeInfo, WindowInfo, WindowSet
from behavior_graphs.pipeline.artifacts import RunArtifacts, resolve_rollout_dir
from behavior_graphs.viz.compat import meta_from_windows
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import render_trajectory_tree
from behavior_graphs.viz.streamlit.data import (
    PARAM_LABELS,
    QUERY_PARAM_NAMES,
    dataset_display_name,
    param_sort_key,
    resolve_data_path,
    run_index_rows,
)

_log = logging.getLogger(__name__)

#: Canvas-height floor per "graphs per row" setting. A floor, not a height: the
#: tree grows the canvas with its own depth so each level has room for its
#: labels, and this is only what a shallow tree gets. Wider cards get more,
#: because the layout is normalised into whatever box it is given and a wide,
#: short one spreads the nodes sideways and stacks their labels.
_GRID_HEIGHTS = {1: 560, 2: 460, 3: 380}

#: Where a graph's link points, relative to the page it is drawn on.
#:
#: The app root, not ``graph_demo``: Streamlit serves an ``st.navigation`` app's
#: *default* page at the base URL and answers its own ``url_path`` with "Page not
#: found — running the app's main page". That failure is nearly invisible (the
#: main page *is* Graph Demo, and the query params still apply, so the right run
#: loads behind a dialog the reader has to dismiss), which is exactly why it is
#: worth pinning: ``tests/test_comparison_graphs.py`` checks that Graph Demo is
#: still the default page, so promoting another one fails a test rather than
#: silently pointing every link at it.
#:
#: Relative rather than ``/``, so it survives being served under a path prefix.
GRAPH_DEMO_URL = "./"

#: How many candidate runs :func:`resolve_run` will check the ``rollout_dir`` of
#: before giving up and taking the first one. The axes usually narrow a policy
#: to a handful of runs, and every check is a small cached read; the cap is here
#: so a selection that narrows to nothing in particular cannot turn into
#: thousands of stat calls.
VERIFY_LIMIT = 25


def dataset_task(dataset: str) -> str:
    """The run-index ``task`` a rollout dataset's runs are filed under.

    Datasets found outside ``$DATA_PATH/rollouts`` can carry a qualified
    ``parent/name`` key (see ``rollout_outcomes._scan_indexed_runs``); the runs
    are filed under the bare dataset directory name either way.
    """
    return dataset.rpartition("/")[2]


def runs_by_dataset(datasets: Sequence[str]) -> dict[str, list[dict]]:
    """``{dataset: its runs}``, in a stable order, for the given datasets.

    Every selected dataset gets a key, empty list included — the section says
    "no runs for this policy" rather than dropping the policy from the grid,
    because a missing column is indistinguishable from one the user forgot to
    select.
    """
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in run_index_rows():
        by_task[row["task"]].append(row)
    return {
        name: sorted(by_task.get(dataset_task(name), ()), key=lambda r: r["path"])
        for name in datasets
    }


def _values(rows: Iterable[Mapping], param: str) -> list:
    return sorted({r[param] for r in rows if r.get(param) is not None}, key=param_sort_key)


def varying_axes(runs: Mapping[str, Sequence[Mapping]]) -> list[tuple[str, str]]:
    """``[(param, label)]`` for the axes that discriminate among *these* runs.

    Scoped to the selected policies rather than to the whole sweep, unlike the
    sidebar picker's axis set: the picker keeps its selectors stable while the
    user switches tasks, whereas this section is redrawn from the policy
    selection anyway, and an axis every candidate agrees on is a dropdown with
    one entry.
    """
    rows = [row for policy_rows in runs.values() for row in policy_rows]
    return [(param, label) for param, label in PARAM_LABELS if len(_values(rows, param)) > 1]


def axis_options(runs: Mapping[str, Sequence[Mapping]], param: str) -> list:
    """The values *param* takes across every policy's remaining candidates."""
    return _values([row for policy_rows in runs.values() for row in policy_rows], param)


def widest_value(runs: Mapping[str, Sequence[Mapping]], param: str):
    """The value of *param* that the most policies have a run at.

    The default a selector opens on, so the grid starts as full as the runs on
    disk allow: picking the first value alphabetically routinely lands on a
    setting only one policy was swept over, and every other card then reads "no
    run at these settings" until the user goes hunting. Ties break toward the
    smallest value, so the choice is stable across reruns.
    """
    coverage: dict[object, int] = defaultdict(int)
    for policy_rows in runs.values():
        for value in {row[param] for row in policy_rows if row.get(param) is not None}:
            coverage[value] += 1
    if not coverage:
        return None
    return min(coverage, key=lambda v: (-coverage[v], param_sort_key(v)))


def narrow(runs: Mapping[str, Sequence[Mapping]], param: str, value) -> dict[str, list[dict]]:
    """*runs* with each policy cut to the candidates whose *param* is *value*.

    Strict equality, which drops the runs that never recorded the param — the
    same rule the sidebar picker filters on, and the one the section's promise
    needs: a run with no ``n_clusters`` is not a run that happens to agree with
    K=10. Such a run stays reachable because an axis with no value among the
    remaining candidates is not drawn at all (see :func:`axis_options`), so a
    policy swept without the sticky-smoothing arm is narrowed by the smoothing
    selector and never asked about the λ it has no value for.
    """
    return {
        name: [row for row in policy_rows if row.get(param) == value]
        for name, policy_rows in runs.items()
    }


@st.cache_data(show_spinner=False)
def _recorded_rollout_dir(run_dir: str, data_path: str | None) -> str | None:
    """Where a run's ``metrics.json`` says its rollouts came from, or None.

    Never raises: an unreadable or absent metrics.json means "this run does not
    say", which :func:`resolve_run` treats as no evidence either way rather
    than as a mismatch — a run interrupted before its evaluation stage has no
    metrics.json at all, and it is still a perfectly good graph.
    """
    try:
        raw = json.loads((Path(run_dir) / "metrics.json").read_text()).get("rollout_dir")
    except Exception as exc:  # a run that cannot say is not a run that disagrees
        _log.debug("no rollout_dir for %s (%s)", run_dir, exc)
        return None
    resolved = resolve_rollout_dir(raw, data_path)
    return None if resolved is None else str(resolved)


def resolve_run(
    candidates: Sequence[Mapping],
    dataset_dir: Path | None,
    data_path: str | None = None,
) -> dict | None:
    """The one run to draw for a policy, or None when nothing matches.

    Prefers a run whose ``metrics.json`` names *dataset_dir* — see the module
    docstring on why the name match is checked rather than trusted — and, among
    equals, one that has metrics at all: a crashed job retried under a new hash
    leaves both behind, sharing every hyperparameter, and the surviving one is
    the one with numbers in it. Falls back to the first candidate when no
    candidate says where its rollouts came from, so a sweep whose evaluation
    stage never ran still draws.
    """
    if not candidates:
        return None
    ordered = sorted(candidates, key=lambda r: (not r.get("has_metrics"), r["path"]))
    if dataset_dir is not None:
        target = str(dataset_dir)
        checked = [
            (_recorded_rollout_dir(row["path"], data_path), row) for row in ordered[:VERIFY_LIMIT]
        ]
        for recorded, row in checked:
            if recorded == target:
                return dict(row)
        if any(recorded is not None for recorded, _ in checked):
            # Runs here do say which dataset they used, and none of them is
            # this one: the name match was wrong, so draw nothing.
            return None
    return dict(ordered[0])


@st.cache_resource(show_spinner=False, max_entries=32)
def tree_inputs(run_dir: str) -> tuple[np.ndarray, list[dict]] | None:
    """``(window labels, per-window metadata)`` for a run's tree, or None.

    What the trajectory tree actually reads: one cluster label per window, and
    each window's episode, frame range and outcome. Deliberately not
    ``data.load_run``, which also loads ``windows.npz`` — the window *feature*
    matrix, 37 MB a run in the deployment's sweep and up to twelve runs on this
    page, for numbers no tree ever looks at. ``load_windows_meta`` reads the
    2.4 MB of JSON beside it instead, and the features are stood in for by an
    empty column so ``WindowSet`` still counts the rows it validates against.

    ``cache_resource``, not ``cache_data``: these are read-only, and pickling a
    50 000-window metadata list per policy per rerun is exactly the cost this
    was written to avoid.

    None when the run is missing either artifact or one of them is unreadable —
    a policy whose card cannot be drawn must not take the other policies' cards
    down with it.
    """
    artifacts = RunArtifacts(Path(run_dir))
    try:
        raw = artifacts.load_windows_meta()
        labels = artifacts.load_array("labels").astype(np.int64)
        windows = [WindowInfo.from_dict(w) for w in raw["windows"]]
        window_set = WindowSet(
            features=np.zeros((len(windows), 0), dtype=np.float32),
            windows=windows,
            episodes=[EpisodeInfo.from_dict(ep) for ep in raw["episodes"]],
        )
    except Exception as exc:  # a half-written run must not take the page down
        _log.warning("cannot read tree inputs for %s (%s)", run_dir, exc)
        return None
    if len(labels) != len(windows):
        _log.warning(
            "%s has %d labels for %d windows; skipping", run_dir, len(labels), len(windows)
        )
        return None
    return labels, meta_from_windows(window_set)


def demo_query_params(row: Mapping, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Query params that open *row*'s run on the Graph Demo page.

    Both the hyperparameters and the run dir: the run dir is what makes the
    link land on the graph that was clicked rather than on some other run
    sharing those axes, while the hyperparameters leave the demo's selectors
    sitting on the right values, so clearing its run-dir box drops the reader
    onto this run's neighbourhood instead of back at the sweep's default.

    *extra* is merged in first, for params that belong to the visit rather than
    to the run — ``pwd`` above all. ``st.link_button`` opens its URL in a new
    tab, so the destination starts a fresh Streamlit session with no memory of
    the visitor having authenticated; without the password on the link, every
    graph on a gated deployment leads to a password prompt.
    """
    params: dict[str, str] = dict(extra or {})
    for name in QUERY_PARAM_NAMES:
        value = row.get(name)
        if value is None or value == "":
            continue
        params[name] = str(value)
    params["run_dir"] = str(row["path"])
    return params


def demo_url(row: Mapping, extra: Mapping[str, str] | None = None) -> str:
    """A relative URL onto the Graph Demo page showing *row*'s run.

    See :data:`GRAPH_DEMO_URL` for why it points at the app root rather than at
    ``graph_demo``, and why it is relative.
    """
    return f"{GRAPH_DEMO_URL}?{urlencode(demo_query_params(row, extra))}"


def _axis_selectors(
    runs: Mapping[str, Sequence[Mapping]], key: str, container
) -> dict[str, list[dict]]:
    """Draw one selector per varying axis; return the runs they narrow down to.

    A cascade, like the sidebar run picker's: each selector offers the values
    still reachable given the ones above it, so no combination of settings can be
    picked that no run was ever built at. Unlike the picker, the candidates span
    every selected policy — narrowing is what makes the trees comparable, and a
    value only some policies were swept over narrows the others out, which the
    grid then reports as "no run at these settings" rather than by quietly
    drawing a tree built some other way.

    *container* is where the widgets go — the sidebar, in the one caller that
    matters, next to where the Graph Demo puts the same selectors.
    """
    candidates = {name: list(rows) for name, rows in runs.items()}
    for param, label in varying_axes(runs):
        options = axis_options(candidates, param)
        if not options:
            # Nothing left to choose: the picks above this one already landed on
            # runs that never recorded this axis (a non-sticky run has no λ).
            continue
        state_key = f"{key}_{param}"
        if state_key not in st.session_state:
            st.session_state[state_key] = widest_value(candidates, param)
        picked = container.selectbox(label, options, key=state_key)
        candidates = narrow(candidates, param, picked)
    return candidates


def render_tree_card(
    dataset: str,
    row: Mapping | None,
    *,
    fold_percent: float,
    theme: str,
    height: int,
    link_params: Mapping[str, str] | None = None,
    key: str = "pc_graph",
) -> None:
    """One policy's trajectory tree, its numbers, and a link onto the Graph Demo.

    *row* is None when the policy has no run at the chosen settings; the card
    still draws, saying so — a policy silently missing from the grid reads as one
    the user forgot to select.
    """
    st.markdown(f"**{dataset_display_name(dataset)}**")
    if row is None:
        st.caption("No run at these settings.")
        return
    run_id = Path(row["path"]).name
    loaded = tree_inputs(row["path"])
    if loaded is None:
        st.caption(f"Run `{run_id}` has no readable labels or windows.")
        return
    labels, metadata = loaded

    rollouts = len({m["rollout_idx"] for m in metadata})
    described = f"K = {row['k']} · " if row.get("k") is not None else ""
    st.caption(f"{described}{rollouts} rollouts · run `{run_id}`")
    render_trajectory_tree(
        labels,
        metadata,
        view_mode="native_svg",
        # The tree's own two knobs, at the Graph Demo's defaults
        # (configs/viz/graph_demo.yaml): fold rare branches rather than hide
        # them, and hide no transition at all — min_branch can disconnect a
        # whole rare-outcome tail, which on a page about outcomes is the one
        # simplification that must not happen behind the reader's back.
        min_branch=1,
        min_visit_fraction=fold_percent / 100.0,
        color_mode="outcome",
        height=height,
        # Keyed on the policy, not just the run: two policies can hold runs whose
        # directory names agree (a flat layout names a run after its params, and
        # this grid picks the same params for every policy), and Streamlit
        # rejects the page outright on a duplicate element key.
        key_prefix=f"{key}_fig_{dataset}_{run_id}",
        theme=theme,
        # A figure, not a tool: a dozen click-to-explore panels down a page is
        # not a comparison, and the link below is where exploring one happens.
        static_figure=True,
        show_stats=False,
        show_paths=False,
    )
    st.link_button(
        "Open in Graph Demo",
        demo_url(row, link_params),
        icon=":material/open_in_new:",
        width="stretch",
    )


def render_section(
    datasets: Mapping[str, Path],
    *,
    key: str = "pc_graph",
    link_params: Mapping[str, str] | None = None,
) -> None:
    """The whole section, for the policies in *datasets* (``{name: dir}``).

    Every control goes in the sidebar — the hyperparameters that pick the runs
    and the two that only affect the drawing — so the page below is the pictures
    and nothing else, and so the axes sit where the Graph Demo keeps them.

    Renders nothing but a note when the policies have no runs: this page is
    perfectly usable on a machine holding rollouts and no pipeline runs at all,
    and a page of empty frames would suggest something had gone wrong.
    """
    runs = runs_by_dataset(list(datasets))
    if not any(runs.values()):
        st.info(
            "No pipeline runs found for these policies. A policy's runs are the ones "
            "under a sweep folder named after its rollout dataset ("
            + ", ".join(f"`{dataset_task(name)}`" for name in datasets)
            + ") — is `$BEHAVIOR_GRAPHS_RUNS_ROOT` pointing at the sweep?"
        )
        return

    sidebar = st.sidebar
    sidebar.header("Behavior graphs")
    sidebar.caption("Every tree on the page is built at these settings.")
    candidates = _axis_selectors(runs, key=key, container=sidebar)
    per_row = int(
        sidebar.radio(
            "Trees per row",
            options=[1, 2, 3],
            # One, like the Graph Demo: a real tree runs tens of levels deep, and
            # at half the content width its node labels land on each other. Two
            # and three are for a quick side-by-side of shallow ones.
            index=0,
            horizontal=True,
            key=f"{key}_per_row",
        )
    )
    fold_percent = float(
        sidebar.slider(
            "Fold branches under N% of rollouts",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5,
            key=f"{key}_fold_pct",
            help=(
                "Drawing only, not a hyperparameter: at each branch point the "
                "children below this share of the rollouts fold into one "
                '"+N variants" node that keeps their episodes and outcomes. '
                "Nothing is hidden, and every tree folds the same way."
            ),
        )
    )

    data_path = resolve_data_path()
    resolved = {
        name: resolve_run(rows, datasets.get(name), data_path) for name, rows in candidates.items()
    }
    if missing := [name for name, row in resolved.items() if row is None]:
        st.caption(
            "No run at these settings for: "
            + ", ".join(f"`{dataset_display_name(name)}`" for name in missing)
            + ". Change an axis in the sidebar to one they were all swept over."
        )
    st.caption(
        "Each path down a tree is a way the policy behaved; a node's colour is how "
        "often that branch ended in success, and its size is how many rollouts took "
        "it. Click through to the Graph Demo to inspect one."
    )

    theme = "light" if st.session_state.get("light_mode") else "dark"
    height = _GRID_HEIGHTS.get(per_row, 380)
    cards = list(resolved.items())
    for start in range(0, len(cards), per_row):
        for column, (name, row) in zip(st.columns(per_row), cards[start : start + per_row]):
            with column, st.container(border=True):
                render_tree_card(
                    name,
                    row,
                    fold_percent=fold_percent,
                    theme=theme,
                    height=height,
                    link_params=link_params,
                    key=key,
                )
