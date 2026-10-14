"""Ablation Study — results of the ABC subtask-boundary sweep, and paper figures.

Reads the same artefacts the paper does: ``rescored.json`` per dataset, through
:mod:`behavior_graphs.ablation_study.cells`. Nothing here recomputes a score, so what
this page shows and what ``study.py report`` prints cannot disagree.

Two things it deliberately does differently from the other pages.

The figures are matplotlib, rendered to images, rather than interactive Plotly.
Plotly cannot export a PDF here (no ``kaleido``), and a figure bound for LaTeX
needs vector output at a fixed column width -- so previewing the very bytes that
get exported is worth more than being able to hover. See
:mod:`behavior_graphs.viz.paper_figures`.

And a dataset whose sweep has not been re-scored is named as such rather than
skipped, because "no rows" and "not scored yet" are different states and the
sweep takes hours.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from behavior_graphs.ablation_study import cells as C
from behavior_graphs.ablation_study import config as study_config
from behavior_graphs.ablation_study import splits
from behavior_graphs.ablation_study.report import REPORTS
from behavior_graphs.viz import paper_figures as pf

st.set_page_config(page_title="Ablation Study — Policy Graphs", layout="wide")
st.title("Ablation Study")
st.caption(
    "Which representation and graph-creation choices recover ground-truth "
    "subtasks, measured by segmentation $F_1$."
)


@st.cache_data(ttl=120, show_spinner="Loading sweep…")
def _load(dataset: str) -> tuple[list[dict], str]:
    """Cells as plain dicts, plus a status line.

    Dicts rather than ``Cell`` objects because Streamlit's cache pickles the
    return value, and a dataclass carrying a metrics dict per cell round-trips
    far more cheaply as records.
    """
    cfg = study_config.load(dataset)
    if not cfg.rescored.exists():
        return [], f"no {cfg.rescored.name} yet — run `study.py rescore --dataset {dataset}`"
    rows = C.load(cfg)
    return [r.__dict__ for r in rows], f"{len(rows)} cells"


def _cells(records: list[dict]) -> list[C.Cell]:
    return [C.Cell(**r) for r in records]


available = study_config.available()
if not available:
    st.error("No dataset configs found under configs/ablation_study/.")
    st.stop()

col_ds, col_refresh = st.columns([4, 1])
with col_ds:
    dataset = st.selectbox("Dataset", available, help="One file in configs/ablation_study/.")
with col_refresh:
    st.write("")
    if st.button("Refresh", help="Re-read rescored.json"):
        _load.clear()

cfg = study_config.load(dataset)
records, status = _load(dataset)

# ── What this dataset is ──────────────────────────────────────────────────────

named = splits.resolve(cfg) if cfg.splits_path.exists() else None
m1, m2, m3, m4 = st.columns(4)
m1.metric("Task", cfg.task.replace("_", " ")[:28])
m2.metric("Cells", len(records) or "—")
if named:
    m3.metric("Episodes", sum(len(v) for v in named.values()))
    m4.metric(
        "Split (train/val/test)",
        f"{len(named['train'])}/{len(named['val'])}/{len(named['test'])}",
    )

if not records:
    st.warning(status)
    st.info(
        "The sweep writes one job directory per configuration; re-scoring turns "
        "those into `rescored.json`, which is what every report and figure here "
        "reads. Until then there is nothing to show for this dataset."
    )
    st.stop()

rows = _cells(records)
present = [r for r in cfg.representations if any(c.layer == r for c in rows)]
missing = [r for r in cfg.representations if r not in present]
st.caption(
    f"{status} · criterion {cfg.evaluation.criterion} · "
    f"arms present: {', '.join(present)}"
    + (f" · **not yet swept: {', '.join(missing)}**" if missing else "")
)
if missing:
    st.warning(
        f"{len(missing)} of {len(cfg.representations)} representations have no "
        "cells yet, so any comparison below is partial."
    )

st.divider()

# ── Figures ───────────────────────────────────────────────────────────────────

st.subheader("Paper figures")
st.caption(
    "Rendered by the same code that exports them, at single-column width with "
    "the paper's font sizes — so the preview is the artefact, not an "
    "approximation of it."
)

fig_names = list(pf.FIGURES)
chosen = st.multiselect(
    "Figures",
    fig_names,
    default=fig_names[:2],
    format_func=lambda n: f"{n} — {pf.FIGURES[n][1].split('.')[0]}",
)
fmt = st.radio(
    "Export format",
    ["pdf", "svg", "png"],
    horizontal=True,
    help=(
        "pdf and svg are vector, for the paper; png is for slides. pdf embeds "
        "Type 42 fonts so the text stays selectable."
    ),
)

for name in chosen:
    builder, blurb = pf.FIGURES[name]
    st.markdown(f"**{name}** — {blurb}")
    try:
        fig = builder(rows, cfg)
    except Exception as e:  # noqa: BLE001
        # One broken figure must not take the page with it: the others are
        # still worth seeing, and the traceback is what says why.
        st.error(f"{name} failed to build: {type(e).__name__}: {e}")
        st.exception(e)
        continue
    left, right = st.columns([3, 1])
    with left:
        # Rendered at the exported size rather than stretched to the column, so
        # the preview shows the real proportions and text size.
        st.pyplot(fig, use_container_width=False)
    with right:
        st.download_button(
            f"Download .{fmt}",
            data=pf.export(fig, fmt),
            file_name=f"{cfg.name}_{name}.{fmt}",
            mime={"pdf": "application/pdf", "svg": "image/svg+xml", "png": "image/png"}[fmt],
            key=f"dl_{name}_{fmt}",
        )
        st.caption(f"{len(pf.export(fig, fmt)) / 1024:.0f} kB")
    fig.clf()

st.divider()

# ── Best per representation ───────────────────────────────────────────────────

st.subheader("Best per representation")
best_rows = []
for layer in present:
    b = C.best(rows, layer=layer)
    if b is None:
        continue
    best_rows.append(
        {
            "representation": pf.rep_label(layer),
            "F1": round(b.score, 4),
            "uniform": round(pf._uniform(b), 4),
            "headroom": round(b.score - pf._uniform(b), 4),
            "density": round(b.density, 2),
            "reduction": f"{b.reduction}-{b.n_components}",
            "clusterer": b.clusterer,
            "K": b.k_label(),
            "window": f"w{b.width}/s{b.stride}",
            "smoothing": b.smoothing,
            "min_seg": b.min_segment,
            "cells": sum(1 for c in rows if c.layer == layer),
        }
    )
best = pd.DataFrame(best_rows).sort_values("F1", ascending=False)
st.dataframe(best, hide_index=True)
st.caption(
    f"`headroom` is F1 minus the count-matched uniform baseline. Differences "
    f"below the across-fold standard error of {cfg.evaluation.noise_floor:.3f} "
    "are not resolvable. `cells` differ between arms because a wide feature "
    "array runs a reduced grid, which is why the `matched` report exists."
)

st.divider()

# ── Text reports ──────────────────────────────────────────────────────────────

st.subheader("Reports")
st.caption("The same text `study.py report` writes, so the two cannot drift.")
which = st.selectbox("Report", sorted(REPORTS), index=sorted(REPORTS).index("matched"))
if st.button(f"Run `{which}`", help="Some reports re-score on held-out episodes and take a minute"):
    with st.spinner(f"Running {which}…"):
        try:
            st.code(REPORTS[which](cfg), language="text")
        except Exception as e:  # noqa: BLE001
            st.error(f"{which} failed: {type(e).__name__}: {e}")
            st.exception(e)

with st.expander("Every cell (table)"):
    st.dataframe(
        pd.DataFrame(
            [
                {a: c.axis(a) for a in C.AXES}
                | {"F1": round(c.score, 4), "density": round(c.density, 2)}
                for c in rows
            ]
        ).sort_values("F1", ascending=False),
        hide_index=True,
    )
