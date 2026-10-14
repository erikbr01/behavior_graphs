"""Policy Comparison — is policy A actually better than policy B?

Pick any N rollout datasets and the page runs the batch-testing recipe from
TRI's *Statistical thinking for robot policy evaluation*:

  - every pair compared with an unconditional exact test (Barnard's by default),
    as TRI's mirrored pair of one-sided tests rather than a two-sided one,
  - family-wise error controlled across the N(N-1)/2 pairs,
  - the surviving separations summarised as a **compact letter display** —
    policies sharing a letter are not statistically separated,
  - uncertainty drawn as a **Beta posterior violin** rather than an error bar,
    because "the intervals overlap, so there is no difference" is a fallacy and
    error bars invite it — the Chart control offers bars ± 1 SE anyway, for
    audiences that expect them, with the same letters over the top.

https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d

Below the statistics, the same policies as *behavior graphs* — one per policy,
all built at one set of pipeline hyperparameters, each linking out to the Graph
Demo page. The tests say whether the success rates differ; the graphs are where
you go to see how the behavior does (see
:mod:`behavior_graphs.viz.streamlit.comparison_graphs`).

The statistics live in :mod:`behavior_graphs.stats.policy_comparison` and the
figure in :mod:`behavior_graphs.viz.plotly.policy_comparison`; this file is the
selector, the controls and the tables.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import streamlit as st

from behavior_graphs.stats.policy_comparison import (
    CORRECTIONS,
    JEFFREYS_PRIOR,
    TESTS,
    UNIFORM_PRIOR,
    compare_policies,
)
from behavior_graphs.viz.plotly.policy_comparison import bar_se_cld_figure, violin_cld_figure
from behavior_graphs.viz.streamlit import policy_groups
from behavior_graphs.viz.streamlit.comparison_graphs import render_section
from behavior_graphs.viz.streamlit.components.comparison_figure_export import (
    comparison_figure_export_panel,
)
from behavior_graphs.viz.streamlit.data import dataset_display_name
from behavior_graphs.viz.streamlit.lazy import lazy_section
from behavior_graphs.viz.streamlit.rollout_outcomes import discover_datasets, load_outcomes

PRIORS: dict[str, tuple[float, float]] = {
    "Uniform — Beta(1, 1)": UNIFORM_PRIOR,
    "Jeffreys — Beta(½, ½)": JEFFREYS_PRIOR,
}

#: Posterior mass covered by the interval drawn on each violin and reported in
#: the summary table — the Bayesian counterpart of a confidence level, and
#: equal-tailed rather than highest-density. Fixed rather than exposed: it
#: changes no test, no p-value and no letter, so a control for it only invites
#: reading a comparison off two intervals, which is the one thing this page
#: exists to stop.
CREDIBILITY_LEVEL = 0.95

#: How to draw the result. Violins first because they are the article's
#: recommendation and the page's default; bars because that is what most
#: audiences expect to see.
VIOLIN_CHART = "Posterior violins"
CHARTS = {
    VIOLIN_CHART: violin_cld_figure,
    "Bars ± standard error": bar_se_cld_figure,
}

#: How a separated pair is marked in the p-value matrix.
_SIG_STYLE = "font-weight: 700; color: #1a4f8f;"

#: Session-state keys for the group editor. The list of ids is the page's own
#: state; each group's name and membership live in their widgets' keys, so
#: Streamlit owns them and a rerun does not need them copied back.
_GROUP_IDS = "pc_group_ids"
_GROUP_NEXT = "pc_group_next"

#: The credible-interval column's header, from the level above it.
_CRI = f"{CREDIBILITY_LEVEL:.0%} CrI"

#: How many label inputs sit side by side. Three fits the page at the widths a
#: policy name needs; more and each box is too narrow to read what is in it.
LABEL_COLUMNS = 3

#: Two arms is the smallest comparison; below that there is nothing to test.
MIN_ARMS = 2
#: The pair count is only half the cost of the unconditional tests — the grid
#: search also grows with the rollout counts. Measured per pair at 8 threads:
#: 13 ms at n=60, 0.9 s at n=1000, so 12 policies is ~0.4 s of small datasets
#: but ~25 s of LIBERO-sized ones. The cap bounds the pairs; per-pair
#: memoisation in the stats layer is what keeps a rerun from paying twice.
MAX_ARMS = 12


st.header("Policy Comparison")

datasets = discover_datasets()
if len(datasets) < MIN_ARMS:
    st.warning(
        f"Found {len(datasets)} rollout dataset(s); at least {MIN_ARMS} are needed to "
        "compare. Datasets are read from `$DATA_PATH/rollouts` and from the "
        "`rollout_dir` of every indexed run — is `$DATA_PATH` set?"
    )
    st.stop()


def _label(name: str) -> str:
    display = dataset_display_name(name)
    return display if display == name else f"{display}  ·  {name}"


def _seed_groups(available: Sequence[str]) -> None:
    """Put the paper's groups in session state, once per session.

    Groups whose patterns match nothing here are skipped
    (:func:`policy_groups.default_groups`); if that leaves nothing — any machine
    without these datasets — one group of whatever the first few datasets are,
    so the page still has something to compare.
    """
    if _GROUP_IDS in st.session_state:
        return
    resolved = policy_groups.default_groups(available)
    if not resolved:
        resolved = {"Policies": list(available)[:3]}
    for position, (label, names) in enumerate(resolved.items()):
        st.session_state[f"pc_group_label::g{position}"] = label
        st.session_state[f"pc_group_sets::g{position}"] = names[:MAX_ARMS]
    st.session_state[_GROUP_IDS] = [f"g{i}" for i in range(len(resolved))]
    st.session_state[_GROUP_NEXT] = len(resolved)


def _add_group() -> None:
    position = st.session_state[_GROUP_NEXT]
    st.session_state[f"pc_group_label::g{position}"] = f"Group {position + 1}"
    st.session_state[f"pc_group_sets::g{position}"] = []
    st.session_state[_GROUP_IDS] = [*st.session_state[_GROUP_IDS], f"g{position}"]
    st.session_state[_GROUP_NEXT] = position + 1


def _remove_group(group_id: str) -> None:
    st.session_state[_GROUP_IDS] = [g for g in st.session_state[_GROUP_IDS] if g != group_id]


_seed_groups(list(datasets))

st.caption(
    "One **group** is one family of tests: every pair inside it is compared and "
    "the correction is applied over those pairs. Keep a group to one task — "
    "comparing datasets from different tasks compares task difficulty, not "
    "policies — and use several groups to put the tasks side by side."
)
for _group_id in list(st.session_state[_GROUP_IDS]):
    _name_key, _sets_key = f"pc_group_label::{_group_id}", f"pc_group_sets::{_group_id}"
    with st.container(border=True):
        _name_col, _remove_col = st.columns([5, 1], vertical_alignment="bottom")
        _name_col.text_input("Group", key=_name_key)
        _remove_col.button(
            "Remove",
            key=f"pc_group_drop::{_group_id}",
            on_click=_remove_group,
            args=(_group_id,),
            width="stretch",
            disabled=len(st.session_state[_GROUP_IDS]) < 2,
            help="A page with no groups has nothing to compare.",
        )
        st.multiselect(
            "Policies",
            options=list(datasets),
            format_func=_label,
            max_selections=MAX_ARMS,
            key=_sets_key,
            label_visibility="collapsed",
        )
st.button("＋ Add a group", on_click=_add_group)

groups = [
    (
        st.session_state[f"pc_group_label::{gid}"] or f"Group {i + 1}",
        st.session_state[f"pc_group_sets::{gid}"],
    )
    for i, gid in enumerate(st.session_state[_GROUP_IDS])
]
# Every dataset any group asked for, for the sections that are about the
# policies rather than about one comparison.
selected = [name for _, names in groups for name in names]

with st.expander("Test settings"):
    left, mid, right = st.columns(3)
    test = left.selectbox(
        "Pairwise test",
        options=list(TESTS),
        format_func=lambda k: TESTS[k],
        help=(
            "Neither margin of the 2x2 table is fixed by the experiment — we chose the "
            "rollout counts, not the total number of successes — so the unconditional "
            "tests are the correct and more powerful choice. Fisher's is here for "
            "comparison against the conditional convention. Each is run as TRI's "
            "mirrored pair of one-sided tests, so the p-values below are one-sided."
        ),
    )
    correction = mid.selectbox(
        "Multiple-comparison correction",
        options=list(CORRECTIONS),
        format_func=lambda k: CORRECTIONS[k],
        help=(
            "Every extra pair is another chance to see a difference that is not there. "
            "Holm controls the same family-wise error as Bonferroni with more power."
        ),
    )
    alpha = right.select_slider(
        "Family-wise α",
        options=[0.001, 0.01, 0.05, 0.1],
        value=0.05,
        format_func=lambda v: f"{v:g}",
    )
    prior_label = left.selectbox("Posterior prior", options=list(PRIORS))

usable = [(label, names) for label, names in groups if len(names) >= MIN_ARMS]
if not usable:
    st.info(f"Select at least {MIN_ARMS} policies in a group.")
    st.stop()
# Only worth saying once some group *did* work: with nothing to compare at all,
# the message above is the whole story.
if short := [label for label, names in groups if 0 < len(names) < MIN_ARMS]:
    st.caption(
        f"Not compared, fewer than {MIN_ARMS} policies: "
        + ", ".join(f"**{label}**" for label in short)
        + "."
    )

with st.expander("Display labels"):
    st.caption(
        "What each policy is called — on the chart, in the tables and in the "
        "exported figure. A dataset name is what the disk calls a run; a figure "
        "in a paper wants something a reader can hold in their head. Renaming "
        "changes no count and no test: the arms are the same rollouts either way."
    )
    # Keyed on the dataset, so a label survives changing the groups, the test
    # settings or the chart — and comes back if the policy does.
    _label_columns = st.columns(min(LABEL_COLUMNS, len(selected)))
    display_labels = {}
    for _i, _name in enumerate(dict.fromkeys(selected)):
        # The paper's own name for it where there is one, else the dataset's
        # display name. The box is labelled with the dataset, not the default —
        # every baseline defaults to "baseline", and three boxes labelled
        # "baseline" would not say which run each one renames.
        _default = policy_groups.paper_label(_name) or dataset_display_name(_name)
        _typed = _label_columns[_i % len(_label_columns)].text_input(
            _name,
            value=_default,
            key=f"policy_label::{_name}",
            help=f"Shown instead of `{_name}`. Blank falls back to `{_default}`.",
        )
        # Two policies given one label are numbered apart downstream
        # (`unique_labels`), so a duplicate is untidy rather than broken.
        display_labels[_name] = _typed.strip() or _default

outcomes_by_group = {
    label: [load_outcomes(name, datasets[name]) for name in names] for label, names in usable
}
_all_outcomes = [o for group in outcomes_by_group.values() for o in group]
if unlabelled := [o for o in _all_outcomes if not o.scored]:
    st.warning(
        "No scored rollouts in "
        + ", ".join(f"`{o.name}`" for o in unlabelled)
        + ". These datasets record no per-episode `success`, so their posterior is "
        "just the prior and every comparison against them is inconclusive."
    )
if partly := [o for o in _all_outcomes if o.unscored and o.scored]:
    st.caption(
        "Excluded from the tallies as unscored: "
        + ", ".join(f"{o.unscored} episode(s) in `{o.name}`" for o in partly)
        + "."
    )

with st.spinner("Running pairwise exact tests…"):
    results = {
        label: compare_policies(
            [o.as_arm(display_labels[o.name]) for o in group],
            alpha=alpha,
            test=test,
            correction=correction,
            prior=PRIORS[prior_label],
            credible_level=CREDIBILITY_LEVEL,
        )
        for label, group in outcomes_by_group.items()
    }

chart = st.radio(
    "Chart",
    options=list(CHARTS),
    horizontal=True,
    help=(
        "The letters are the same either way — they come from the tests, not the "
        "drawing. Standard error bars are drawn at ±1 SE of the observed rate; two "
        "of them overlapping is not a test, which is why the violins are the default."
    ),
)


def _render_group(label: str, outcomes, result, key: str) -> None:
    """One group's chart and tables.

    Its own family of tests: the letters and p-values here are corrected over
    this group's pairs and say nothing about a policy in another group.
    """
    st.plotly_chart(CHARTS[chart](result), use_container_width=True, theme=None, key=f"{key}_chart")

    # Only worth a column when a display_name renames something; otherwise it is
    # a verbatim copy of Policy, and it pushes the numbers off the visible width.
    show_dataset = any(o.name != name for o, name in zip(outcomes, result.labels))
    summary = pd.DataFrame(
        [
            {
                "Group": result.letters[i] or "—",
                "Policy": result.labels[i],
                "Successes": result.arms[i].successes,
                "Rollouts": result.arms[i].trials,
                "Success rate": 100 * result.arms[i].rate,
                "Posterior median": 100 * result.medians[i],
                _CRI: f"{result.intervals[i][0]:.1%} – {result.intervals[i][1]:.1%}",
                **({"Dataset": outcomes[i].name} if show_dataset else {}),
            }
            for i in result.order
        ]
    )
    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Success rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Posterior median": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    if lazy_section(
        "Pairwise p-values",
        key=f"{key}_pvalues",
        default_open=len(results) == 1,
        help=(
            "One-sided, in the direction the observed rates point, and adjusted for the "
            "family of N(N-1)/2 comparisons. Bold marks a pair separated at the α above."
        ),
    ):
        names = [result.labels[i] for i in result.order]
        grid = result.adjusted[np.ix_(result.order, result.order)]
        # Pre-formatted strings rather than a numeric frame with a Styler format:
        # st.dataframe renders NaN with its own null placeholder ("None"), so the
        # Styler's na_rep never reaches the diagonal.
        shown = pd.DataFrame(
            [["—" if np.isnan(v) else f"{v:.4f}" for v in row] for row in grid],
            index=names,
            columns=names,
        )
        emphasis = np.where(np.nan_to_num(grid, nan=1.0) <= result.alpha, _SIG_STYLE, "")
        st.dataframe(shown.style.apply(lambda _: emphasis, axis=None), use_container_width=True)


# ── One tab per group ────────────────────────────────────────────────────────

if len(results) == 1:
    _only = next(iter(results))
    _render_group(_only, outcomes_by_group[_only], results[_only], "pc_g0")
else:
    for _position, (_tab, _label) in enumerate(zip(st.tabs(list(results)), results)):
        with _tab:
            _render_group(_label, outcomes_by_group[_label], results[_label], f"pc_g{_position}")

# Under the violins only — see the panel's own docstring on why the bar chart
# gets no paper-ready export.
if chart == VIOLIN_CHART:
    comparison_figure_export_panel(list(results.values()), list(results))


# ── Behavior graphs ──────────────────────────────────────────────────────────
# Last, and lazy. The first thing this section needs is the run index, which is
# tens of seconds cold over a full sweep — and nothing above it needs a single
# pipeline run, so a reader who came for the success rates must not pay for it.
#
# `pwd` rides along into the links because the deployment is password-gated and
# a link is a fresh page load: without it, following one from an authorized
# session lands on the password prompt (see viz.streamlit.auth.password_gate).
if lazy_section(
    "Behavior graphs",
    key="pc_graphs",
    help=(
        "One behavior graph per selected policy, all built with the same pipeline "
        "hyperparameters, each linking out to the Graph Demo page."
    ),
):
    _pwd = st.query_params.get("pwd")
    render_section(
        {name: datasets[name] for name in selected},
        link_params={"pwd": _pwd} if _pwd else None,
    )
