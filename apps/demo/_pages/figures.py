"""Figures — every paper-ready export in one place.

The exports live next to the analyses that produce them: a behavior graph is
exported from Graph Demo, the violins from Policy Comparison, the study's panels
from Survey Analytics. That is the right home for each of them while you are
*deciding* what the figure should say, and the wrong one when you are assembling
a paper — which means opening three pages, re-selecting the same three runs on
one of them, and remembering which settings the last export used.

So this page collects them, pointed at the study's own three tasks. Nothing here
computes a figure differently from the page it came from; it imports the same
panels, so a figure exported here and one exported there are the same file.

Every section is lazy. Between them they want three swept runs, the rollout
tallies for eleven datasets and a run of pairwise exact tests, and a page that
paid all of that on load would take half a minute to show a title.

**The graphs are the trajectory trees the study actually showed**, drawn by the
same component, from each stage's own session config: its run, its ``min_branch``
and the rest of its ``graph:`` block merged exactly as the study merges them. The
tree is built inside the component from the window labels rather than read off
the run, so there is no way to draw it without rendering it — the alternative was
the Markov graph, which is not what any participant saw.

Rendered as a static figure: no video panel, no path list, no fold badges. The
tree is here to be exported, not explored, and the exploring version is on Graph
Demo.

**The policy comparison is here twice.** Once as the violins alone, which is
the results figure, and once with the study's data-collection answers as a row
of bars per display under each panel — the decision and its consequence in one
figure, joined by the x position rather than by a policy name a reader has to
carry between two of them. Same comparison, same letters (both sections call
:func:`_comparison_panels`); the second one only adds the bars, and exports
under its own name so a paper can carry both.

**One section exports a video rather than a still.** A behavior graph's claim is
that its decomposition lines up with what the robot does, and that pairing is
the one thing a still cannot carry: the clip and the bar have to be the same
file, playing. So "Rollout clips" writes the annotation into the MP4 — the same
strip the player draws, under the same clip (:mod:`behavior_graphs.viz.clip_figure`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from behavior_graphs.core.node_labels import load_node_labels
from behavior_graphs.stats.policy_comparison import compare_policies
from behavior_graphs.viz.labels import number_repeats
from behavior_graphs.viz.streamlit import policy_groups
from behavior_graphs.viz.streamlit.components.comparison_figure_export import (
    comparison_figure_export_panel,
)
from behavior_graphs.viz.streamlit.components.latex_export import latex_export_panel
from behavior_graphs.viz.streamlit.components.paper_export import paper_figure_downloads
from behavior_graphs.viz.streamlit.components.trajectory_tree_view import render_trajectory_tree
from behavior_graphs.viz.streamlit.data import resolve_data_path
from behavior_graphs.viz.streamlit.lazy import lazy_section
from behavior_graphs.viz.streamlit.rollout_outcomes import discover_datasets, load_outcomes
from behavior_graphs.viz.streamlit.user_study import (
    claims,
    failure_scoring,
    figure_export,
    regret,
    table_export,
)
from behavior_graphs.viz.streamlit.user_study.response_store import (
    GCSResponseStore,
    LocalResponseStore,
    results_dir,
)

st.set_page_config(page_title="Figures — Policy Graphs", layout="wide")
st.title("Figures")
st.caption(
    "Every figure the paper carries, exported at the size it will be placed at. "
    "Each section is the same panel as the page it comes from — pick the size "
    "preset that matches where the figure goes, and export the PDF."
)
st.info(
    "Export at the width the figure will **occupy on the page**. A graph "
    "exported at 7.16 in and placed in a third of a double column has its 8 pt "
    "type scaled to 2.7 pt by the LaTeX, which is what makes a three-across "
    "figure unreadable. The size presets carry a matching node and type size, "
    "since a third-page figure needs both.",
    icon="📐",
)


def _study_stages() -> list:
    """The study's stages, in the order its config lists them.

    Not cached: the artifacts each one needs are session-state-shaped and the
    component below caches its own tree build, so there is nothing here worth
    holding but the config read.
    """
    from behavior_graphs.viz.streamlit.user_studies.registry import get_variant

    variant = get_variant(failure_scoring.DEFAULT_VARIANT)
    return list(variant.stages) if variant else []


def _stage_graph(stage) -> tuple[dict, dict] | None:
    """``(artifacts, graph_params)`` for one stage, or ``None`` if it will not load.

    The graph parameters are merged the way the study merges them — the
    variant's ``graph:`` block, then the session's own on top, then anything the
    stage entry overrides — because a tree drawn at a different ``min_branch``
    from the one the participant read is a different tree.
    """
    from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
    from behavior_graphs.viz.streamlit.user_study.sessions import (
        load_session_artifacts,
        sessions_dir,
    )

    variant = get_variant(failure_scoring.DEFAULT_VARIANT)
    sess, _config, run, mp4_dir, index, errors = load_session_artifacts(
        sessions_dir() / f"{stage.session_name}.yaml", data_path=resolve_data_path()
    )
    if errors or run is None:
        for error in errors:
            st.warning(f"{stage.label}: {error}")
        return None
    params = {
        **(variant.graph_params if variant else {}),
        **(sess.get("graph") or {}),
        **(stage.graph_params or {}),
    }
    return (
        {"run": run, "mp4_dir": mp4_dir, "index": index, "run_dir": str(run.run_dir)},
        params,
    )


# ── The behavior graphs ──────────────────────────────────────────────────────

if lazy_section(
    "Behavior graphs — the trajectory tree per study task",
    key="fig_graphs",
    default_open=True,
):
    _stages = _study_stages()
    if not _stages:
        st.warning(
            "No study variant found — is `configs/viz/study_variants/` in this "
            "checkout, and its sessions under `configs/viz/study/`?"
        )
    for _stage in _stages:
        st.markdown(f"#### {_stage.label or _stage.task}")
        _loaded = _stage_graph(_stage)
        if _loaded is None:
            continue
        _artifacts, _params = _loaded
        _run = _artifacts["run"]
        _vlm = load_node_labels(Path(_artifacts["run_dir"]))
        # Numbered where a name repeats, as on Graph Demo: two clusters the
        # labeling model described in the same words are two clusters, and a
        # paper figure that draws them with the same text says otherwise.
        _names = number_repeats({nid: entry["label"] for nid, entry in _vlm.items()})
        st.caption(
            f"`{_run.graph.num_episodes}` episodes · `min_branch="
            f"{int(_params.get('min_branch', 2))}` and `color_mode="
            f"{_params.get('color_mode', 'outcome')}`, this stage's own settings"
        )

        # A fragment because the component issues st.rerun(scope="fragment")
        # internally, which raises outside one — same reason Graph Demo and the
        # study wrap their dispatches.
        @st.fragment
        def _render(stage=_stage, artifacts=_artifacts, params=_params, names=_names) -> None:
            render_trajectory_tree(
                labels=artifacts["run"].labels,
                metadata=artifacts["run"].metadata,
                view_mode=params.get("view_mode", "native_svg"),
                min_branch=int(params.get("min_branch", 2)),
                max_depth_cap=500,
                color_mode=params.get("color_mode", "outcome"),
                node_values={},
                cluster_names=names,
                node_vlm_labels=None,
                mp4_dir=artifacts["mp4_dir"],
                mp4_index=artifacts["index"],
                height=600,
                height_extra=int(params.get("graph_height_extra", 0)),
                level="rollout",
                key_prefix=f"fig_tree_{stage.task}",
                # Here to be exported, not explored: no video panel, no path
                # list, no per-node fold badges, no build statistics.
                static_figure=True,
                show_paths=False,
                show_stats=False,
                show_fold_toggles=False,
                edge_style=params.get("edge_style", "lines"),
                edge_width_slope=5.0,
                node_size_slope=16.0,
                # No `windows`/`precomputed_subclusters`: those are for
                # expanding a terminal leaf into sub-clusters, which is
                # exploration and which `static_figure` rules out anyway. The
                # study's own render omits them too, so this is the same tree.
                paper_export=True,
                paper_export_filename=f"tree_{stage.task}",
                # Open, because the export is what this page is: a collapsed
                # panel here is a click between a reader and the only thing on
                # the page.
                paper_export_expanded=True,
                # Starts at the whole page: these trees are labelled with the
                # sentences the labeling model wrote, and twenty of those fit
                # neither the 4.4 inches a Markov graph is happy in nor the wide
                # band a flat graph wants.
                paper_export_preset="page",
            )

        _render()
    st.divider()


# ── The clips the graphs are claims about ────────────────────────────────────
# The one figure on this page that is not a still: a rollout with the graph's
# decomposition of it burned in underneath, which is the only form in which
# that pairing survives leaving the app (see `viz.clip_figure`). Its own
# section rather than under the trees above, because those are drawn as
# pictures — `static_figure` means no video panel to hang an export off — and
# because what a supplementary video picks is an episode, not a branch.

if lazy_section(
    "Rollout clips — the video with its behavior bar",
    key="fig_clips",
):
    from behavior_graphs.viz.streamlit.components.camera_views import player_views
    from behavior_graphs.viz.streamlit.components.clip_export import (
        clip_export_panel,
        episode_bands,
    )
    from behavior_graphs.viz.streamlit.data import find_mp4_episode

    _stages = _study_stages()
    if not _stages:
        st.warning("No study variant found, so there are no runs here to take a clip from.")
    else:
        _by_label = {(_s.label or _s.task): _s for _s in _stages}
        _task_col, _ep_col = st.columns([2, 1])
        _stage = _by_label[
            _task_col.selectbox(
                "Task",
                list(_by_label),
                key="fig_clip_task",
                help="The same three runs as the graphs above, and the same settings.",
            )
        ]
        _loaded = _stage_graph(_stage)
        if _loaded is None:
            st.warning(f"{_stage.label}: this stage's run is not mounted here.")
        else:
            _artifacts, _ = _loaded
            _run, _index, _mp4_dir = (
                _artifacts["run"],
                _artifacts["index"],
                _artifacts["mp4_dir"],
            )
            _episodes = _index["episodes"]
            if not _episodes or _mp4_dir is None:
                st.warning("This run has no rollout videos mounted.")
            else:
                _vlm = load_node_labels(Path(_artifacts["run_dir"]))
                _names = number_repeats({nid: entry["label"] for nid, entry in _vlm.items()})

                def _episode_name(index: int) -> str:
                    _entry = find_mp4_episode(index, _index)
                    _outcome = _entry.get("success") if _entry else None
                    return (
                        f"{'✓' if _outcome is True else '✗' if _outcome is False else '•'} "
                        f"Episode {index}"
                    )

                _ep_idx = _ep_col.selectbox(
                    "Rollout",
                    [int(_e["index"]) for _e in _episodes],
                    format_func=_episode_name,
                    key=f"fig_clip_ep_{_stage.task}",
                    help="Every rollout in the run, with its outcome.",
                )
                _entry = find_mp4_episode(int(_ep_idx), _index) or {}
                # The whole episode's decomposition, not one branch of it: there
                # is no graph selection on this page, and a supplementary video
                # is watched end to end anyway.
                _bands = episode_bands(
                    _run.labels,
                    _run.metadata,
                    int(_ep_idx),
                    cluster_names=_names,
                    frame_count=_entry.get("frame_count"),
                )
                st.caption(
                    f"`{len(_bands)}` behavior segments over "
                    f"`{_entry.get('frame_count', 0)}` policy steps, coloured by "
                    "behavior as the graph is."
                )
                _views = player_views(_entry, _mp4_dir, _index)
                clip_export_panel(
                    _views["videos"],
                    _bands,
                    total_frames=_entry.get("frame_count"),
                    view_transforms=_views["view_transforms"],
                    visible_cameras=_views["visible_cameras"],
                    key=f"fig_clip_{_stage.task}",
                    filename=f"clip_{_stage.task}_ep{_ep_idx}",
                    # This page *is* its exports: a collapsed panel here is a
                    # click between a reader and the only thing in the section.
                    expanded=True,
                )
    st.divider()


# ── The violins ──────────────────────────────────────────────────────────────


@st.cache_data(ttl=60, show_spinner="Loading responses…")
def _responses() -> list[dict]:
    """Every response in the store, wherever the deployment keeps them.

    Defined up here rather than with the study sections below because the
    violins want them too: the second violin figure draws the study's
    data-collection answers under the policies they asked for.
    """
    import os

    bucket = os.environ.get("SURVEY_GCS_BUCKET", "").strip()
    if bucket:
        try:
            return GCSResponseStore(bucket).list_all()
        except Exception as error:
            st.error(f"GCS error: {error}")
            return []
    return LocalResponseStore(results_dir()).list_all()


def _comparison_panels() -> tuple[dict, list, list]:
    """``(groups, results, titles)`` — the paper's comparison, panel by panel.

    Shared by the two violin sections below, which draw the same comparison and
    must not come to disagree about a letter: the second one only adds the
    study's answers under it. Not cached, because each section is behind its own
    toggle and the tests are a few milliseconds once the tallies are read — what
    costs is :func:`load_outcomes`, which caches itself.
    """
    datasets = discover_datasets()
    groups = policy_groups.default_groups(list(datasets))
    results, titles = [], []
    for label, names in groups.items():
        arms = [
            load_outcomes(name, datasets[name]).as_arm(policy_groups.paper_label(name) or name)
            for name in names
        ]
        results.append(compare_policies(arms, pin_first=policy_groups.BASELINE_LABEL))
        titles.append(label)
    return groups, results, titles


if lazy_section(
    "Policy comparison — posterior violins",
    key="fig_violins",
):
    with st.spinner("Running pairwise exact tests…"):
        _groups, _results, _titles = _comparison_panels()
    if not _groups:
        st.warning("None of the paper's datasets are mounted here.")
    else:
        st.caption(
            "Groups and labels from "
            "`viz.streamlit.policy_groups`. Change them on Policy Comparison, "
            "which is also where the tables and p-values are. Baseline first, "
            "then by success rate — the letters are named by rate regardless, "
            "so **a** is the best arm and not the leftmost one."
        )
        comparison_figure_export_panel(_results, _titles, key="fig_violins_panel", expanded=True)
    st.divider()


# ── The ablation, as the paper prints it ─────────────────────────────────────

#: The three representations the paper compares, in table order. Proprioception
#: first as the learning-free baseline, then the policy's own tap, then a frozen
#: backbone the policy never saw. Named per dataset because the collectors
#: recorded the same policy tap under different names.
_PAPER_REPS = {
    "cooking": ["state", "cond_proj", "dinov3"],
    "_default": ["state", "rgrl_cond", "dinov3"],
}

if lazy_section(
    "Ablation — the paper's table",
    key="fig_paper_table",
    help="Three representations by three reductions, nine rows. K, window "
    "width, stride and smoothing are searched identically for every row and "
    "not reported, because the best K is a property of the task.",
    default_open=True,
):
    from behavior_graphs.ablation_study import config as _paper_config
    from behavior_graphs.ablation_study.report import per_task as _paper_per_task

    _p1, _p2, _p3 = st.columns([2, 1, 1])
    with _p1:
        # cooking only, and deliberately: it is the one dataset whose cells form
        # a complete cross product, so every row of this table is a maximum over
        # the same 96 configurations. The ABC sweeps accumulated across several
        # grid passes -- bottles' DINOv3 arm has 88 cells at PCA@10 against 146
        # at UMAP@10, chess has no identity cells at all -- so a best-per-cell
        # table there would favour whichever cell was searched hardest.
        _paper_names = [n for n in _paper_config.available() if n == "cooking"] or (
            _paper_config.available()
        )
        _paper_task = st.selectbox(
            "Task",
            _paper_names,
            key="fig_paper_task",
            help="Only tasks whose design is a complete cross product belong "
            "here. On a task whose cells came from several grid passes the rows "
            "are maxima over unequal search spaces, which favours the level "
            "that was searched hardest.",
        )
    _paper_cfg = _paper_config.load(_paper_task)
    with _p2:
        _paper_nc = st.selectbox(
            "Components",
            [10, 100],
            key="fig_paper_nc",
            help="Pinned rather than maximised over. identity takes no "
            "component count, so folding UMAP@10 and UMAP@100 into one row "
            "would search UMAP and PCA over twice as many cells as identity.",
        )
    with _p3:
        _paper_k = st.number_input(
            "K floor",
            min_value=0,
            max_value=64,
            value=int(getattr(_paper_cfg.evaluation, "k_floor", 0) or 0),
            step=1,
            key="fig_paper_k",
            help="From the task config. K is searched but not reported: the "
            "best K follows the task's own segment count, so a K row would be "
            "a row of task properties rather than of design choices.",
        )

    if not _paper_cfg.rescored.exists():
        st.warning(
            f"No `{_paper_cfg.rescored.name}` for **{_paper_task}**. "
            f"Run `study.py rescore --dataset {_paper_task}`."
        )
    else:
        _paper_reps = _PAPER_REPS.get(_paper_task, _PAPER_REPS["_default"])
        _paper_rows = _paper_per_task.rows(
            _paper_cfg,
            k_min=int(_paper_k),
            representations=_paper_reps,
            n_components=int(_paper_nc),
        )
        if not _paper_rows:
            st.warning(
                f"No cells for **{_paper_task}** with representations "
                f"{', '.join(_paper_reps)} at {_paper_nc} components and K ≥ {_paper_k}."
            )
        else:
            _counts = {r["n_cells"] for r in _paper_rows}
            if len(_counts) == 1:
                st.success(
                    f"Balanced: every row is the best of the same "
                    f"**{_counts.pop()}** configurations "
                    "(width × stride × K × smoothing)."
                )
            else:
                st.error(
                    "**Not balanced** — cells per row: "
                    + ", ".join(str(c) for c in sorted(_counts))
                    + ". Each row is a maximum over a differently sized search, "
                    "so the biggest is flattered. Do not put this in the paper."
                )
            # K is deliberately absent: searched identically for every row, then
            # left out, because the best K follows the task's segment count and
            # a column of task properties in a table of design choices reads as
            # a recommendation.
            _paper_frame = pd.DataFrame(_paper_rows)[
                ["representation", "reduction", "F1", "IoU", "density", "silhouette"]
            ]
            st.dataframe(
                _paper_frame,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                    "IoU": st.column_config.NumberColumn(format="%.3f"),
                    "density": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Predicted over ground-truth segment count. 1.0 is "
                        "the target; larger is over-segmentation, not better.",
                    ),
                    "silhouette": st.column_config.NumberColumn(
                        format="%.3f",
                        help="Unsupervised, and reported for contrast only. It "
                        "anti-correlates with F1 against hand annotation on 7 of "
                        "8 annotated tasks, and is measured in each reduction's "
                        "own space — so it is comparable neither to the other "
                        "columns nor down a block.",
                    ),
                },
            )
            st.caption(
                "**Read the three columns together.** At a fixed IoU threshold "
                "$F_1$ rises with segment density, so density says whether a row "
                "won by placing boundaries or by emitting more of them, and IoU "
                "(threshold-free) says how well the ones it placed line up. Both "
                "belong to the winning cell, not to their own column's maximum."
            )
            latex_export_panel(
                _paper_frame,
                ["F1", "IoU", "density", "silhouette"],
                group_column="representation",
                filename=f"ablation_paper_{_paper_task}.tex",
                default_caption=_paper_per_task.default_caption(
                    _paper_cfg, _paper_rows, int(_paper_k)
                ),
                default_label=f"tab:ablation-{_paper_task}",
                key="fig_paper_tex",
            )
    st.divider()


# ── The ablation, one task ───────────────────────────────────────────────────

if lazy_section(
    "Ablation — one task, representation × reduction",
    key="fig_ablation_task",
    help="The two reported axes crossed, with the IoU and density of each "
    "winning cell. Pick a task and the K floor its ground truth implies.",
    default_open=True,
):
    from behavior_graphs.ablation_study import config as _task_config
    from behavior_graphs.ablation_study.report import per_task as _per_task

    _task_names = _task_config.available()
    _c1, _c2 = st.columns([2, 1])
    with _c1:
        # cooking first when present: it is the task this table was built for,
        # and the one whose reduction effect flips between representations.
        _order = sorted(_task_names, key=lambda n: (n != "cooking", n))
        _task = st.selectbox(
            "Task", _order, key="fig_abl_task", help="One file in configs/ablation_study/."
        )
    _task_cfg = _task_config.load(_task)
    with _c2:
        _k_min = st.number_input(
            "K floor",
            min_value=0,
            max_value=64,
            value=int(getattr(_task_cfg.evaluation, "k_floor", 0) or 0),
            step=1,
            key="fig_abl_kfloor",
            help="Drop cells below this K. Not cosmetic: a partition into fewer "
            "clusters than the ground truth has segments per episode cannot "
            "represent it, and pooling those cells reversed the reduction "
            "ordering on cooking. 0 keeps every K.",
        )

    if not _task_cfg.rescored.exists():
        st.warning(
            f"No `{_task_cfg.rescored.name}` for **{_task}** yet. This table reads the "
            "re-scored artefact rather than recomputing, so that it cannot disagree "
            f"with `study.py report`. Run `study.py rescore --dataset {_task}`."
        )
    else:
        _abl_rows = _per_task.rows(_task_cfg, k_min=int(_k_min))
        if not _abl_rows:
            st.warning(f"No cells for **{_task}** at K ≥ {_k_min}.")
        else:
            st.dataframe(
                pd.DataFrame(_abl_rows)[
                    [
                        "representation",
                        "reduction",
                        "F1",
                        "IoU",
                        "density",
                        "K",
                        "median_F1",
                        "n_cells",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "F1": st.column_config.NumberColumn(format="%.3f"),
                    "IoU": st.column_config.NumberColumn(format="%.3f"),
                    "density": st.column_config.NumberColumn(
                        format="%.2f",
                        help="Predicted segments over ground-truth segments. "
                        "1.0 is the target — a larger value is over-segmentation, "
                        "not a better score.",
                    ),
                    "median_F1": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            _stale = ", ".join(
                sorted(
                    {
                        r["layer"]
                        for r in _abl_rows
                        if r["n_cells"] < max(x["n_cells"] for x in _abl_rows)
                    }
                )
            )
            if _stale:
                st.warning(
                    f"Uneven cell counts — **{_stale}** has fewer cells than the "
                    "widest arm, so its maximum is a lower bound rather than a "
                    "result. Either the sweep is still running or the rescore "
                    "predates it."
                )
            st.caption(
                "**IoU and density belong to the winning cell**, not to each "
                "column's own maximum — so the row describes one configuration "
                "that exists. `median_F1` is over the same nuisance grid the max "
                "was taken over, and a level that is good only at its peak is a "
                "different claim from one that is good throughout."
            )

            with st.expander("Export as LaTeX (booktabs)"):
                _at_caption = st.text_area(
                    "Caption",
                    value="",
                    placeholder="leave empty for the generated caption",
                    help="Raw LaTeX. The generated default states the ground-truth "
                    "source, the K floor and its justification, and what IoU and "
                    "density mean — none of which a reader can recover from the "
                    "numbers.",
                    key="fig_abl_tex_caption",
                    height=70,
                )
                _at_label = st.text_input(
                    "Label",
                    value=f"tab:ablation-{_task}",
                    key="fig_abl_tex_label",
                )
                _at_compact = st.checkbox(
                    "Compact (single column)", value=True, key="fig_abl_tex_compact"
                )
                _at_tex = _per_task.table(
                    _task_cfg,
                    k_min=int(_k_min),
                    caption=_at_caption or None,
                    label=_at_label or None,
                    compact=_at_compact,
                )
                st.code(_at_tex, language="latex")
                st.download_button(
                    "Download .tex",
                    _at_tex,
                    file_name=f"ablation_{_task}.tex",
                    mime="text/x-tex",
                    icon=":material/download:",
                    key="fig_abl_tex_dl",
                )
                st.caption(
                    "Needs `\\usepackage{booktabs}` and `\\usepackage{multirow}` "
                    "(the representation column is set as `\\multirow` blocks). "
                    "**Density is never bolded** — `latex_table.NOT_RANKABLE` — "
                    "because its target is 1.0 and neither direction is better."
                )
    st.divider()


# ── The ablation, pooled over tasks ──────────────────────────────────────────

if lazy_section(
    "Ablation — pooled over tasks",
    key="fig_ablation_pooled",
    help="Every stage option's mean held-out F1 across the study's tasks, and "
    "the LaTeX for it. The per-task tables are on Ablation Study.",
):
    from behavior_graphs.ablation_study import config as _study_config
    from behavior_graphs.ablation_study.report import aggregate as _aggregate

    # Reads caches only, so this is a file read per dataset rather than the
    # pipeline scoring run per *level* that produced them -- which is minutes a
    # dataset and has no business behind a click. `study.py cache_rows` is what
    # computes them; a dataset without one is named below rather than silently
    # dropped, because a pooled mean over an unknown subset is not a number.
    _names = _study_config.available()
    _cfgs = [_study_config.load(_n) for _n in _names]
    try:
        _agg = _aggregate.pooled(_cfgs)
    except ValueError as _exc:
        # Two criteria among the caches. Refusing is the right answer and the
        # message says which, so it goes on screen rather than into a traceback.
        st.error(str(_exc))
        _agg = None

    if _agg is not None and not _agg.rows:
        st.warning(
            "No dataset has cached ablation rows yet, so there is nothing to pool. "
            "Run `python scripts/ablation_study/study.py cache_rows --dataset <name>` "
            "for each of: " + ", ".join(_names)
        )
    elif _agg is not None:
        st.caption(
            f"Pooled over **{len(_agg.datasets)}** tasks "
            f"({', '.join(sorted(_agg.datasets))}), criterion `{_agg.criterion}`. "
            "Each cell is the mean over tasks of the best $F_1$ that option "
            "reaches, chosen on each task's selection episodes and scored on its "
            "reserved ones — so the number is held out, not the maximum it was "
            "picked by."
        )
        if _agg.missing:
            st.warning(
                "**Not** in any number here, because they have no cached rows: "
                + ", ".join(_agg.missing)
                + ". `cooking` is expected among them — it ran the `ablation` "
                "grid rather than the balanced `ablation_droid` rectangle this table is "
                "built on, so it has no core cells to select from."
            )
        st.dataframe(
            pd.DataFrame(_aggregate.dataframe_rows(_agg)),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "**`best on` is the column to read next to the mean.** An option that "
            "wins narrowly on every task and one that wins hugely on a single task "
            "can share a mean, and only the count separates them. `SD` is the "
            "spread across tasks and is descriptive, not a standard error — these "
            "are the study's particular tasks, not a sample from a population."
        )

        st.markdown("**Mean cost of choosing the wrong level**")
        st.caption(
            "Each task's own within-axis range, averaged. This is the *local* "
            "sensitivity — what a wrong choice costs once everything else is "
            "tuned — which is not the same as how much of the grid's variance the "
            "axis explains. `report/latex.py` documents where the two disagree: "
            "width and stride explain almost none of the variance yet are among "
            "the most expensive things to change at the optimum."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "stage": _aggregate.latex.AXIS_LABELS.get(_ax, _ax),
                        "mean range": (
                            None
                            if _agg.spread_of(_ax) != _agg.spread_of(_ax)
                            else round(_agg.spread_of(_ax), 3)
                        ),
                    }
                    for _ax in _agg.axes
                ]
            ),
            hide_index=True,
        )

        with st.expander("Export as LaTeX (booktabs)"):
            _abl_caption = st.text_input(
                "Caption",
                value="",
                placeholder="leave empty for the generated caption",
                help="Raw LaTeX. The generated default names the tasks, says how "
                "each cell was selected and scored, and explains the `best on` "
                "column — all three are things a reader needs and none of them "
                "are guessable from the numbers.",
                key="fig_ablation_tex_caption",
            )
            _abl_label = st.text_input(
                "Label", value="tab:ablation-pooled", key="fig_ablation_tex_label"
            )
            _abl_compact = st.checkbox(
                "Compact (single column)",
                value=True,
                help="Trims the outer column separation and shortens the headers, "
                "for a table that has to fit one column of a twocolumn layout.",
                key="fig_ablation_tex_compact",
            )
            _abl_tex = _aggregate.table(
                _agg,
                caption=_abl_caption or None,
                label=_abl_label or None,
                compact=_abl_compact,
            )
            st.code(_abl_tex, language="latex")
            st.download_button(
                "Download .tex",
                _abl_tex,
                file_name="ablation_pooled.tex",
                mime="text/x-tex",
                icon=":material/download:",
                key="fig_ablation_tex_dl",
            )
            st.caption(
                "Needs `\\usepackage{booktabs}` and, because the stage column is "
                "set as `\\multirow` blocks, `\\usepackage{multirow}`."
            )
    st.divider()


# ── The violins, with what the study asked for ───────────────────────────────

if lazy_section(
    "Policy comparison — violins over the study's choices",
    key="fig_violins_choices",
):
    # The same comparison as the section above, with one thing added: under each
    # violin, how many participants asked for that policy's data, split by the
    # display they read the baseline through. It is the figure that puts the
    # study's decision and its consequence in one place — the arm that was
    # picked and the success rate it turned out to have — and the join is the
    # x position, so a reader does it by looking down rather than by carrying a
    # policy name between two figures.
    with st.spinner("Running pairwise exact tests…"):
        _c_groups, _c_results, _c_titles = _comparison_panels()
    if not _c_groups:
        st.warning("None of the paper's datasets are mounted here.")
    else:
        # Assigned stages only, and the default for the same reason the study
        # figure restricts its boxes: the third stage's display was the
        # participant's own pick, so a share pooled over it describes the people
        # who reached for the graph rather than what the graph led people to.
        # 16 of 18 picked the graph there.
        _randomized = st.checkbox(
            "Randomized stages only",
            value=True,
            key="fig_violins_choices_randomized",
            help=(
                "Count only the stages whose display was **assigned**. The "
                "third stage let the participant choose their display, so its "
                "answers are a result of the participant and not of the "
                "display — the restriction every display comparison in the "
                "paper is made under."
            ),
        )
        _share_rows = regret.choice_share_rows(
            claims.study_responses(_responses()), randomized_only=_randomized
        )
        _c_choices = [
            figure_export.choice_series(_share_rows, _names) for _names in _c_groups.values()
        ]
        _answered = sum(series.total for panel in _c_choices for series in panel)
        if not _answered:
            st.info(
                "No data-collection answers to draw: this needs completed "
                "responses whose stages recorded a display."
            )
        st.caption(
            "Bars are **shares of that display's answers for that task**, with "
            "the count beside each — the denominators are single figures, so "
            "read the counts. The baseline carries a dash rather than a zero: "
            "nobody was offered “collect nothing”, and a policy nobody could "
            "pick is not a policy nobody picked. The heavier outline marks "
            "where each display's answers most went."
        )
        comparison_figure_export_panel(
            _c_results,
            _c_titles,
            choices=_c_choices,
            key="fig_violins_choices_panel",
            expanded=True,
        )
    st.divider()


# ── The study's own answers ──────────────────────────────────────────────────


@st.cache_data(show_spinner="Reading the retrained policies' rollouts…")
def _arm_rates() -> dict:
    """Measured success rates of every option's retrained policy.

    Cached because the study figure's regret panel wants them and they are a
    directory walk plus a tally per dataset. The regret-only figure is not on
    this page: it is a second reading of the panel that figure already carries,
    and it lives on Survey Analytics with the tables that explain it.
    """
    from behavior_graphs.viz.streamlit.user_study import regret

    return regret.measured_success_rates()


@st.cache_data(ttl=60, show_spinner="Fitting the adjusted models…")
def _adjusted_fits(rows: list[dict]) -> tuple[list[dict], dict | None, list[str]]:
    """The rating fits and the decision fit for *rows*, and what went wrong.

    Cached and shared, because three sections here draw these same two fits and
    each was sampling them again: four chains per rated outcome plus four for
    the decisions, for numbers that cannot come out different between sections
    — same model, same seed. The ``ttl`` matches :func:`_responses`, so a new
    submission reaches the fits when it reaches the store.

    ``(effects, regret_effect, notes)``. The notes are the caller's to show: a
    ``RuntimeError`` in here is the deployment saying what it does not have —
    the ``analysis`` extra, or the retrained rollouts a choice is scored
    against — and which panels that costs depends on the section.
    """
    from behavior_graphs.viz.streamlit.user_study import mixed_effects

    if not rows:
        return [], None, []
    try:
        effects = mixed_effects.continuous_effects(rows)
    except RuntimeError as error:
        return [], None, [str(error)]
    if not effects:
        return [], None, []
    try:
        return effects, mixed_effects.decision_regret_effect(rows, _arm_rates()), []
    except RuntimeError as error:
        return effects, None, [f"{error} Drawing it without the decision."]


#: Arrows for a table cell, where there is no axis to put a direction on.
_BETTER_ARROWS = {"down": "↓", "up": "↑"}


def _outcome_cell(row: dict) -> str:
    """An outcome's name with the way it improves, for the tables below.

    The figures say this after the panel name; a table has no axis to hang it
    on, so it goes after the row's name. Per row, because the outcomes
    disagree: less perceived time and less effort is the better display, more
    confidence is, and less regret is a better decision. The arrow alone, with
    what it means said once in the caption below the table rather than four
    times inside it.
    """
    arrow = _BETTER_ARROWS.get(str(row.get("better") or ""))
    return f"{row['label']} {arrow}" if arrow else str(row["label"])


if lazy_section(
    "Study answers — the raw responses",
    key="fig_study_raw",
):
    # The 18: completed participants of the reported layout, which is what
    # `claims.study_responses` narrows to and what every number in the sections
    # below is over. The panels that draw ratings apply the same filters
    # themselves; the scored data-collection choices do not, so without this the
    # figure's last panel would be drawn from four choices that appear in
    # nothing else on the page.
    _raw_rows = claims.study_responses(_responses())
    # Restricted to the stages whose display was ASSIGNED, which is what the
    # adjusted models below are fit on. The figure a paper leads with is the one
    # a reader believes before they believe a model, so it has to be the same
    # evidence: the third stage let the participant choose their display, so it
    # answers a preference question rather than a randomized one, and putting a
    # third more data in the boxes than in the interval beside them is a
    # discrepancy a careful reader finds and nobody can account for.
    # How the last panel draws the decisions. The box is the default: it is the
    # answers themselves, and it is the mark the three rating panels beside it
    # already use. The mean bar is the faster read and prints the number the
    # paper's sentences quote — see `figure_export.REGRET_MARKS`.
    _raw_mark = st.radio(
        "Regret panel",
        options=list(figure_export.REGRET_MARKS),
        format_func=lambda mark: figure_export.REGRET_MARKS[mark],
        horizontal=True,
        key="fig_study_raw_mark",
        help=(
            "Regret takes one of three values per task, so the answers pile up "
            "at 0 and at the cost of each wrong option. The box shows those "
            "piles; a mean with a standard error around it implies a spread "
            "between them that no participant could have answered."
        ),
    )
    _raw_files = figure_export.study_figure_files(
        _raw_rows,
        regret.regret_rows(_raw_rows, _arm_rates()),
        randomized_only=True,
        regret_mark=_raw_mark,
    )
    if _raw_files is None:
        st.info(
            "Nothing to draw: this needs a rated stage whose display was assigned, "
            "from a completed three-stage response."
        )
    else:
        # No `st.image` here: `paper_figure_downloads` previews the PNG itself,
        # boxing it if it would run past the height cap, and a second one above
        # it drew the figure twice.
        paper_figure_downloads(
            _raw_files,
            key="fig_study_raw_dl",
            # A stem, not a filename: the component adds each format's suffix,
            # so this was downloading `study_raw_responses.pdf.pdf`. The mark is
            # in it because the two versions are two files a paper may want to
            # compare, and one overwriting the other is how that goes wrong.
            filename=f"study_raw_responses_{_raw_mark}",
            width_in=figure_export.DOUBLE_COLUMN_WIDTH_IN,
            height_in=figure_export.FIGURE_HEIGHT_IN,
        )
        st.caption(
            "Chosen display, the three rated dimensions, and what the "
            "data-collection choices were worth — every panel a count of real "
            "answers, no model. `n` under each box is what it is drawn from, and "
            "the **arrow after a panel name is the direction that is better** — "
            "down for perceived time, effort and regret, up for confidence. "
            "The last panel is **regret — lower is better**: the success rate the "
            "chosen option left on the table against the best one available, one "
            "dot per decision. Pooling those points across tasks is the caveat to "
            "carry: the options span 33 success-rate points on set table and 8 on "
            "cooking, so the pile leans toward the widest and its upper tail is "
            "set table's alone. The per-task panels are on Survey Analytics."
        )
    st.divider()


if lazy_section(
    "Study answers — the same figure, adjusted",
    key="fig_study_marginals",
):
    # Panel for panel the figure above, with the model's estimate in place of
    # the answers: the adjusted marginal mean per display and its 95% HDI. The
    # two are meant to be swappable in a paper, which is why nothing else about
    # the figure changes — same panels, same order, same axes, same n under
    # each tick.
    #
    # Behind its own toggle because it is four Monte-Carlo chains per outcome
    # plus one for the decisions, and the figure above it costs a directory
    # walk.
    _marginal_rows = claims.study_responses(_responses())
    _marginal_effects, _marginal_regret, _marginal_notes = _adjusted_fits(_marginal_rows)
    for _note in _marginal_notes:
        st.warning(_note)
    _marginal_files = (
        figure_export.adjusted_marginals_figure_files(
            _marginal_rows, _marginal_effects, regret_effect=_marginal_regret
        )
        if _marginal_effects
        else None
    )
    if _marginal_files is None:
        st.info(
            "Nothing to draw: this needs the adjusted fits, which need completed "
            "randomized three-stage responses."
        )
    else:
        paper_figure_downloads(
            _marginal_files,
            key="fig_study_marginals_dl",
            filename="study_adjusted_marginals",
            width_in=figure_export.DOUBLE_COLUMN_WIDTH_IN,
            height_in=figure_export.FIGURE_HEIGHT_IN,
        )
        st.caption(
            "Each mark is the **estimated marginal mean (EMM)** for that display "
            "with its 95% HDI, holding task, stage and experience fixed — narrower "
            "than the answers it is a mean of, so do not read the bars as a spread "
            "of ratings. The **arrow after a panel name is the direction that is "
            "better**, as in the figure above. The preference panel is unchanged "
            "and unmodelled: it is a count of what people picked, and there is "
            "nothing to adjust it for."
        )
        st.warning(
            "**Overlapping intervals here are not “no difference”.** Each interval "
            "includes the uncertainty about where the whole scale sits — the "
            "intercept, the task and the stage — which both displays share, so it "
            "cancels in the difference. Their posteriors correlate +0.64 for "
            "perceived time, making the difference's SD 6.2 where independence "
            "would give 10.2. The two marks overlap by 0.6 points and the "
            "difference is −26.1 [−37.3, −14.0]; for effort they overlap by 6.6 and "
            "the difference is −22.9 [−37.7, −9.0]. Quote the difference — the "
            "figure right below, the EMM table further down, or panel (b) of the "
            "adjusted figure — not the gap between the marks.",
            icon="⚠️",
        )
    st.divider()

if lazy_section(
    "Study answers — the four differences, one column",
    key="fig_study_differences",
):
    # The estimates the paper's sentences quote, on their own: one row per
    # outcome, single column, four rows of height -- so it fits beside the
    # paragraph that quotes it, which neither of the double-column figures
    # above can. Same estimates as panel (b) of the adjusted figure, with the
    # decision added and everything else taken away.
    #
    # The fits are the marginals section's, cached, so opening both costs one
    # set of chains.
    _difference_rows = claims.study_responses(_responses())
    _difference_effects, _difference_regret, _difference_notes = _adjusted_fits(_difference_rows)
    for _note in _difference_notes:
        st.warning(_note)
    _difference_files = (
        figure_export.adjusted_differences_figure_files(
            _difference_effects, regret_effect=_difference_regret
        )
        if _difference_effects
        else None
    )
    if _difference_files is None:
        st.info(
            "Nothing to draw: this needs the adjusted fits, which need completed "
            "randomized three-stage responses."
        )
    else:
        paper_figure_downloads(
            _difference_files,
            key="fig_study_differences_dl",
            filename="study_adjusted_differences",
            width_in=figure_export.SINGLE_COLUMN_WIDTH_IN,
            height_in=figure_export.DIFFERENCE_FIGURE_HEIGHT_IN,
        )
        st.caption(
            "One mark per outcome: the **adjusted graph-minus-videos difference** "
            "and its 95% HDI, each the `graph` coefficient of its own model. A "
            "mark whose interval clears the zero line is a difference the model "
            "separates from none, and the **arrow after a tick is the direction "
            "that is better**, as in the figures above. **The ticks share an axis "
            "but not a scale** — three are rating points on the 0–100 scale "
            "participants answered on and the fourth, past the hairline, is "
            "points of policy success rate — so read each mark against zero "
            "rather than one mark's distance from it against another's."
        )
    st.divider()


def _rounded(value, digits: int = 1) -> object:
    """*value* rounded, or an em dash — a missing estimate is not a zero."""
    return "—" if value is None else round(float(value), digits)


if lazy_section(
    "Study answers — the claims",
    key="fig_study_claims",
):
    # Regenerated, not transcribed. Every number a sentence in the paper needs,
    # beside the denominator that sentence needs: at n=18 one participant is 5.6
    # percentage points, and a share with its count hidden invites a reader to
    # assume a study an order of magnitude larger than this one.
    #
    # No model here, so no sampler and no optional extra — these are counts and
    # within-participant differences. The adjusted estimates are in the section
    # below, and the point of having both is that they agree, which is what lets
    # a paper lead with the raw figure and cite the model as a check.
    _claim_rows = claims.study_responses(_responses())
    _preference = claims.preference_claim(_claim_rows)
    _ratings = claims.paired_rating_claims(_claim_rows)
    _decisions = claims.decision_claims(regret.regret_rows(_claim_rows, _arm_rates()))

    # Said once, at the top, and again in every column header: which of two
    # things a number is. An estimated marginal mean and a raw mean are within
    # two points of each other here, so nothing about a figure of them looks
    # wrong when they are swapped — only the interval beside them stops being
    # the interval of the thing it is printed next to.
    st.info(
        "**Everything in this section is raw data** — counts, and differences "
        "taken within each participant. No model, so no intervals. The "
        "**estimated marginal means (EMMs)** and their 95% HDIs are in the two "
        "sections below: *the same figure, adjusted*, and *adjusted rating "
        "effects*. Quote an EMM whenever the sentence carries an interval.",
        icon="🔢",
    )

    st.markdown("**Preference — raw counts**")
    if not _preference["n_chose"]:
        st.info("No completed response reached the free-choice stage.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Display": phase,
                        "Participants": count,
                        "Share": f"{count}/{_preference['n_chose']}"
                        f" = {_preference['share'][phase]:.0f}%",
                    }
                    for phase, count in _preference["counts"].items()
                ]
            ),
            hide_index=True,
        )
        if _preference["n_chose"] != _preference["n_participants"]:
            st.caption(
                f"{_preference['n_chose']} of {_preference['n_participants']} "
                "participants recorded a choice, so that is the denominator here."
            )

    st.markdown("**Ratings — raw, paired within participant**")
    if not _ratings:
        st.info("No participant rated both displays, so there is no pair to compare.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Outcome": _outcome_cell(row),
                        "Pairs": row["n_pairs"],
                        # Every column that has an EMM counterpart below says
                        # "raw" in its header, because the two are within a
                        # couple of points of each other and the only visible
                        # difference is the interval one of them can carry.
                        "Graph (raw)": round(row["mean_graph"], 1),
                        "Videos (raw)": round(row["mean_videos"], 1),
                        "Difference (raw, points)": round(row["difference"], 1),
                        "Relative (raw)": (
                            "—" if row["relative"] is None else f"{row['relative']:+.0f}%"
                        ),
                        "Graph lower": f"{row['lower']}/{row['n_pairs']}"
                        f" = {100 * row['lower'] / row['n_pairs']:.0f}%",
                        "Graph higher": f"{row['higher']}/{row['n_pairs']}"
                        f" = {100 * row['higher'] / row['n_pairs']:.0f}%",
                        "Tied": row["tied"],
                    }
                    for row in _ratings
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "The **arrow after an outcome is the direction that is better**: less "
            "perceived time and effort, more confidence. "
            "Relative is of the 0–100 scale the participant picked on, so name the "
            "construct when quoting it: “35% lower perceived effort”, “40% lower "
            "perceived time”. That is what these items measure — the clock time is a "
            "different number, and it is on Survey Analytics as step durations."
        )

    st.markdown("**Decisions — raw, per scored choice**")
    if not _decisions["pooled"]["n"]:
        st.info(
            "No choice could be scored: this needs the retrained policies' rollouts "
            "mounted, which is what the options are compared against."
        )
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Group": label,
                        "Choices": row["n"],
                        "Mean regret (raw, points)": (
                            "—" if row["mean_regret"] is None else round(row["mean_regret"], 1)
                        ),
                        "Beat the baseline": (
                            "—"
                            if row["share_positive"] is None
                            else f"{row['n_positive']}/{row['n']} = {row['share_positive']:.0f}%"
                        ),
                        "Picked the best": (
                            "—"
                            if row["share_best"] is None
                            else f"{row['n_best']}/{row['n']} = {row['share_best']:.0f}%"
                        ),
                    }
                    for label, row in (
                        [("Pooled", _decisions["pooled"])]
                        + [
                            (f"Display: {k} (randomized)", v)
                            for k, v in _decisions["by_display"].items()
                        ]
                        + [(f"Task: {k}", v) for k, v in _decisions["by_task"].items()]
                    )
                ]
            ),
            hide_index=True,
        )
        st.caption(
            f"**The display rows are the {_decisions['n_randomized']} randomized "
            "choices**, not all "
            f"{_decisions['pooled']['n']}: the third stage's display was the "
            "participant's own pick and almost all of them picked the graph, so "
            "pooling it makes the display a result of the participant. It reverses "
            "the answer, too. **And do not quote the pooled share on its own** — "
            "whether a good option existed differs by task, cooking's best arm being "
            "worth +3.3 points and two of its three negative, so pooled it is partly "
            "a statement about which tasks a participant was shown."
        )

    st.divider()


if lazy_section(
    "Study answers — preference, adjusted rating effects, and regret",
    key="fig_study",
):
    # The same rows every other section reports on -- finished, three-stage,
    # on the reported layout -- which is also what makes `_adjusted_fits` a
    # cache hit when one of those sections has already opened.
    _rows = claims.study_responses(_responses())
    _mode = st.radio(
        "Rating display",
        options=list(figure_export.ADJUSTED_FIGURE_MODES),
        format_func=lambda mode: figure_export.ADJUSTED_FIGURE_MODES[mode],
        horizontal=True,
        key="fig_study_rating_display",
    )
    # The decision panel needs every option's retrained policy on disk, so the
    # figure carries it only where those rollouts are mounted — which is what
    # `_adjusted_fits` reports back as a note.
    _effects, _regret_effect, _notes = _adjusted_fits(_rows)
    for _note in _notes:
        st.warning(_note)
    _files = (
        figure_export.adjusted_study_figure_files(
            _rows, _effects, regret_effect=_regret_effect, mode=_mode
        )
        if _effects
        else None
    )
    if _files is None:
        st.info("Nothing to draw: the figure needs completed randomized three-stage responses.")
    else:
        st.image(_files["png"], width="stretch")
        _pdf, _png = st.columns(2)
        _pdf.download_button(
            "⬇ PDF (for the paper)",
            data=_files["pdf"],
            file_name="study_answers.pdf",
            mime="application/pdf",
            width="stretch",
            key="fig_study_pdf",
        )
        _png.download_button(
            "⬇ PNG (for a slide)",
            data=_files["png"],
            file_name="study_answers.png",
            mime="image/png",
            width="stretch",
            key="fig_study_png",
        )
        # The figure draws the intervals; the paper needs the numbers behind
        # them, and one number the figure cannot carry: the percentage of the
        # model's own marginal mean. A sentence quoting this interval has to
        # quote this percentage, not the raw one in the claims section above —
        # they differ by under a point, which is why the mix-up survives review.
        # The decision last, as every figure here draws it: the ratings are
        # the subjective half of the outcome it reports. It is the row a reader
        # would otherwise have to read off a figure, and the one whose points
        # are a different kind of point -- see the caption.
        _adjusted = claims.adjusted_rating_claims(_effects)
        _decision_claim = claims.adjusted_decision_claim(_regret_effect)
        if _decision_claim is not None:
            _adjusted = [*_adjusted, _decision_claim]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Outcome": _outcome_cell(row),
                        "Graph (EMM)": _rounded(row["mean_graph"]),
                        "Videos (EMM)": _rounded(row["mean_videos"]),
                        "Difference (EMM, points)": _rounded(row["difference"]),
                        "95% HDI": (
                            "—"
                            if row["lower"] is None or row["upper"] is None
                            else f"[{row['lower']:+.1f}, {row['upper']:+.1f}]"
                        ),
                        "Relative (EMM)": (
                            "—" if row["relative"] is None else f"{row['relative']:+.1f}%"
                        ),
                        "r̂": _rounded(row["r_hat"], 3),
                    }
                    for row in _adjusted
                ]
            ),
            hide_index=True,
        )
        st.caption(
            "**EMM** = estimated marginal mean: the model's mean for that display, "
            "averaged over the observed spread of task, stage and experience, with "
            "the participant intercepts left out. “Relative (EMM)” is of the EMM "
            "for videos, so it belongs with the interval beside it — the raw "
            "percentage in the claims section does not. r̂ above 1.01 means the "
            "chains did not mix and the interval should not be quoted "
            "(Vehtari et al., 2021).\n\n"
            "The **arrow after an outcome is the direction that is better**. "
            "**The rated rows and the decision row are not in the same points.** "
            "The three ratings are on the 0–100 scale participants answered on; "
            "**success rate regret** is in points of policy success rate — the rate "
            "the chosen option left on the table against the best one available, "
            "so **lower is better** and its relative figure is a ratio of two "
            "small numbers."
        )
        # The same rows, as the paper's table: the measurements, each display's
        # EMM, and the difference with its interval. Generated from `_adjusted`
        # rather than retyped, which is the whole point -- a number in the paper
        # and the number on this page come off one fit.
        with st.expander("Export as LaTeX (booktabs)"):
            _tex_caption = st.text_input(
                "Caption",
                value=table_export.DEFAULT_CAPTION,
                # Raw, not escaped: a paper caption wants `\cite{}` and `$\hat{r}$`
                # to work. The generated default is already valid LaTeX.
                help="Raw LaTeX. A literal `%`, `&` or `_` needs a backslash.",
                key="fig_study_tex_caption",
            )
            _tex_label = st.text_input(
                "Label",
                value=table_export.DEFAULT_LABEL,
                key="fig_study_tex_label",
            )
            _tex = table_export.adjusted_table_tex(
                _adjusted,
                caption=_tex_caption or None,
                label=_tex_label or None,
            )
            st.code(_tex, language="latex")
            st.download_button(
                "Download .tex",
                _tex,
                file_name="study_adjusted_table.tex",
                mime="text/x-tex",
                icon=":material/download:",
                key="fig_study_tex_dl",
            )
            st.caption(
                "One decimal, which is what the estimate supports: the "
                "Monte-Carlo error on these means runs about 0.1 rating points, "
                "so a second decimal is the sampler's noise — seeded and "
                "reproducible, and not the study's."
            )
    st.divider()
