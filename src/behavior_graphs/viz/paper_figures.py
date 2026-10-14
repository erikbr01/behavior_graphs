"""Paper-ready figures for the ABC subtask-boundary ablation.

Matplotlib rather than Plotly, which is what the rest of the app draws with.
Two reasons, and the first is decisive: Plotly's static export needs ``kaleido``,
which is not installed, so it cannot write a PDF at all. And a figure going into
LaTeX wants vector output at a known column width with the paper's own font
sizes, which is what matplotlib's PDF backend gives directly.

So the app shows these as rendered images rather than interactive charts. That is
the point -- what you preview is byte-identical to what you export, so a figure
cannot look right on screen and wrong in the paper.

Each builder takes the loaded cells and returns a ``Figure``. Nothing here reads
from disk or touches Streamlit, so every figure is testable on synthetic cells.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Iterable, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on this box, and none needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavior_graphs.ablation_study import cells as C  # noqa: E402
from behavior_graphs.ablation_study.config import DatasetConfig  # noqa: E402

#: Single-column width for a two-column paper, in inches. 3.35in is the IEEE and
#: ACM column; a figure drawn at this size and dropped in at 100% keeps the font
#: sizes below, which is the whole point of setting them.
COL_W = 3.35
#: Full text width, for a figure spanning both columns.
FULL_W = 6.9

#: Deliberately conservative. A paper figure is read at 9pt or smaller, so the
#: defaults (10pt labels, heavy spines, coloured background) are all wrong.
PAPER_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # Keep text as text in the PDF so it can be searched, copied, and restyled,
    # rather than baked into outlines at whatever size it was drawn.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

#: One colour per representation, held across every figure so a reader who
#: learns it once keeps it. Colour-blind safe (Wong).
REP_COLORS: dict[str, str] = {
    "state": "#0072B2",
    "rgrl_enc": "#009E73",
    "rgrl_cond": "#D55E00",
    "rgrl_dit": "#CC79A7",
    "dinov3": "#E69F00",
}
_FALLBACK = "#7f7f7f"

#: Shown instead of the raw layer name. `rgrl_*` means nothing to a reader.
REP_LABELS: dict[str, str] = {
    "state": "Proprioception",
    "rgrl_enc": "Policy encoder",
    "rgrl_cond": "Policy conditioning",
    "rgrl_dit": "Policy DiT block",
    "dinov3": "DINOv3 (frozen)",
    # The DROID cooking collection's own taps. `cond_proj` is the same thing
    # `rgrl_cond` is on the ABC tasks under the name that collector recorded,
    # so it gets the same label -- a table showing both spellings would read as
    # two representations.
    "cond_proj": "Policy conditioning",
    # Single-view DINOv3, named by camera. The unqualified `dinov3` above is
    # every view the policy trained on, concatenated, which is the arm that
    # compares fairly against a policy tap; these two say whether the second
    # view earns its width.
    "dinov3_ext": "DINOv3, exterior view",
    "dinov3_wrist": "DINOv3, wrist view",
}


def rep_label(layer: str) -> str:
    return REP_LABELS.get(layer, layer)


def rep_color(layer: str) -> str:
    return REP_COLORS.get(layer, _FALLBACK)


@contextlib.contextmanager
def paper_style():
    """Draw with the paper's rc settings, leaving the global state as it was."""
    with plt.rc_context(PAPER_RC):
        yield


def export(fig, fmt: str = "pdf") -> bytes:
    """Serialise a figure. ``pdf`` and ``svg`` are vector; ``png`` is for slides.

    No PGF: it writes fine but matplotlib needs a working LaTeX install to
    measure the text, which is a dependency this box does not have and a paper
    does not need -- a PDF with embedded Type 42 fonts drops into LaTeX just as
    cleanly.
    """
    if fmt not in ("pdf", "svg", "png"):
        raise ValueError(f"unsupported format {fmt!r}; want pdf, svg or png")
    buf = io.BytesIO()
    # 600 dpi only matters for png; vector formats ignore it.
    fig.savefig(buf, format=fmt, dpi=600 if fmt == "png" else None)
    return buf.getvalue()


# --------------------------------------------------------------------- helpers


def _present(rows: Sequence[C.Cell], cfg: DatasetConfig) -> list[str]:
    """Representations that actually have cells, in the config's order."""
    have = {c.layer for c in rows}
    return [r for r in cfg.representations if r in have]


def _uniform(cell: C.Cell, iou: str = "0.5") -> float:
    """The count-matched uniform baseline recorded alongside a cell's score."""
    return float(cell.metrics.get(f"segmental_uniform_f1@{iou}", float("nan")))


# --------------------------------------------------------------------- figures


def representation_bars(
    rows: Sequence[C.Cell], cfg: DatasetConfig, matched: bool = True
) -> plt.Figure:
    """Best segmentation F1 per representation, against the uniform baseline.

    The headline figure. Two bars per representation when ``matched``: the best
    over everything that arm searched, and the best over the configurations
    *every* arm searched. They differ because the arms got unequal budgets, and
    showing only the first would let search effort read as representation
    quality.
    """
    from behavior_graphs.ablation_study.report.matched import MATCH_AXES, common_configs

    present = _present(rows, cfg)
    shared = common_configs(list(rows), present) if matched and len(present) > 1 else set()

    labels, own, mat, unif = [], [], [], []
    for layer in present:
        mine = [c for c in rows if c.layer == layer]
        b = max(mine, key=lambda c: c.score)
        labels.append(rep_label(layer))
        own.append(b.score)
        unif.append(_uniform(b))
        inter = [c for c in mine if tuple(str(c.axis(a)) for a in MATCH_AXES) in shared]
        mat.append(max(inter, key=lambda c: c.score).score if inter else np.nan)

    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_W, 2.1))
        x = np.arange(len(labels))
        show_matched = matched and not all(np.isnan(mat))
        w = 0.38 if show_matched else 0.6
        ax.bar(
            x - (w / 2 if show_matched else 0),
            own,
            w,
            label="Own best",
            color=[rep_color(layer) for layer in present],
        )
        if show_matched:
            ax.bar(
                x + w / 2,
                mat,
                w,
                label="Matched subgrid",
                color=[rep_color(layer) for layer in present],
                alpha=0.45,
                hatch="//",
            )
        # One line, not a bar: the baseline is a property of the segment count,
        # so it barely moves between arms and a bar per arm would imply it does.
        base = float(np.nanmean(unif))
        ax.axhline(base, color="0.35", linestyle="--", linewidth=0.9)
        ax.annotate(
            f"count-matched uniform ({base:.3f})",
            xy=(0.99, base),
            xycoords=("axes fraction", "data"),
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.35",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=18, ha="right")
        ax.set_ylabel("Segmentation $F_1$ (IoU 0.5)")
        ax.set_ylim(0, max(1.0, max(own) * 1.15))
        if show_matched:
            ax.legend(loc="upper left", ncols=2)
        ax.set_title(f"{cfg.name}: best per representation")
    return fig


def axis_variance(rows: Sequence[C.Cell], cfg: DatasetConfig) -> plt.Figure:
    """Share of score variance explained by each design axis.

    Answers "which dials matter", and it is the justification for fixing width
    and stride in the reduced grids -- so it is the figure that has to be shown
    if that restriction is mentioned.
    """
    per_job = C.per_job(rows)
    axes_eta = [
        (a, C.eta_squared(list(rows if a == "min_segment" else per_job), a)) for a in C.AXES
    ]
    axes_eta.sort(key=lambda kv: kv[1])

    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_W, 2.3))
        names = [a.replace("_", " ") for a, _ in axes_eta]
        vals = [v for _, v in axes_eta]
        ax.barh(np.arange(len(names)), vals, color="#0072B2", height=0.68)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel(r"variance explained, $\eta^2$")
        ax.set_xlim(0, max(vals) * 1.18 if vals else 1)
        for i, v in enumerate(vals):
            ax.annotate(
                f"{v:.3f}",
                (v, i),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=6,
            )
        ax.set_title(f"{cfg.name}: which dials decide the score")
        # `min_segment` is measured over cells and the rest over jobs, because
        # collapsing min_segment to its best would give that axis zero variance
        # by construction.
    return fig


def k_curve(rows: Sequence[C.Cell], cfg: DatasetConfig) -> plt.Figure:
    """Best score against cluster count, one line per representation.

    Shows where each representation peaks. K is a user-facing choice rather
    than something the method fixes, so the shape of this curve is the useful
    output, not its argmax.
    """
    present = _present(rows, cfg)
    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_W, 2.1))
        drew = False
        for layer in present:
            # k-means/GMM/spectral counts only. HDBSCAN's dial is a minimum
            # cluster size, which is not a K and would plot as a huge one.
            pts: dict[float, float] = {}
            for c in rows:
                if c.layer != layer or c.clusterer == "hdbscan" or c.k is None:
                    continue
                try:
                    k = float(c.k)
                except (TypeError, ValueError):
                    continue
                pts[k] = max(pts.get(k, -np.inf), c.score)
            if len(pts) < 2:
                continue
            ks = sorted(pts)
            ax.plot(
                ks, [pts[k] for k in ks], marker="o", color=rep_color(layer), label=rep_label(layer)
            )
            drew = True
        if not drew:
            ax.text(
                0.5,
                0.5,
                "no clusterer with a cluster count",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
                color="0.4",
            )
        ax.set_xlabel("clusters $K$")
        ax.set_ylabel("Segmentation $F_1$ (IoU 0.5)")
        ax.set_xscale("log", base=2)
        if drew:
            ax.legend(loc="best", ncols=1)
        ax.set_title(f"{cfg.name}: score against $K$")
    return fig


def density_curve(rows: Sequence[C.Cell], cfg: DatasetConfig) -> plt.Figure:
    """Score against predicted-to-true segment density.

    The metric's own safety argument: it is single-peaked in density, so a
    configuration cannot win by over-segmenting. Worth a figure because the
    alternative criterion (gain over a count-matched reference) is *not*
    single-peaked, and that is why it was dropped.
    """
    present = _present(rows, cfg)
    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_W, 2.1))
        for layer in present:
            mine = [c for c in rows if c.layer == layer and np.isfinite(c.density)]
            if not mine:
                continue
            ax.scatter(
                [c.density for c in mine],
                [c.score for c in mine],
                s=3,
                alpha=0.35,
                color=rep_color(layer),
                label=rep_label(layer),
                linewidths=0,
            )
        ax.axvline(1.0, color="0.35", linestyle="--", linewidth=0.9)
        ax.annotate(
            "one predicted segment per true segment",
            xy=(1.0, 0.02),
            xycoords=("data", "axes fraction"),
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=6,
            color="0.35",
        )
        ax.set_xlabel("segment density (predicted / true)")
        ax.set_ylabel("Segmentation $F_1$ (IoU 0.5)")
        ax.set_xlim(0, 3)
        leg = ax.legend(loc="upper right", ncols=1, markerscale=2.5)
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
        ax.set_title(f"{cfg.name}: single-peaked in density")
    return fig


def iou_sweep(rows: Sequence[C.Cell], cfg: DatasetConfig) -> plt.Figure:
    """Each representation's best, as the IoU threshold for a hit is tightened.

    A single threshold is a choice; this shows whether an ordering survives it.
    """
    present = _present(rows, cfg)
    thresholds = [f"{t:g}" for t in cfg.evaluation.iou_thresholds]
    with paper_style():
        fig, ax = plt.subplots(figsize=(COL_W, 2.1))
        for layer in present:
            mine = [c for c in rows if c.layer == layer]
            if not mine:
                continue
            ys = []
            for t in thresholds:
                vals = [
                    float(c.metrics[f"segmental_f1@{t}"])
                    for c in mine
                    if f"segmental_f1@{t}" in c.metrics
                ]
                ys.append(max(vals) if vals else np.nan)
            if all(np.isnan(ys)):
                continue
            ax.plot(
                [float(t) for t in thresholds],
                ys,
                marker="o",
                color=rep_color(layer),
                label=rep_label(layer),
            )
        ax.set_xlabel("IoU threshold for a matched segment")
        ax.set_ylabel("Segmentation $F_1$")
        ax.legend(loc="best")
        ax.set_title(f"{cfg.name}: sensitivity to the IoU threshold")
    return fig


#: Figure name -> (builder, one-line description shown in the app).
FIGURES: dict[str, tuple[Any, str]] = {
    "representation": (
        representation_bars,
        "Best segmentation F1 per representation, own best against the matched "
        "subgrid, with the count-matched uniform baseline.",
    ),
    "axis_variance": (
        axis_variance,
        "Variance explained by each design axis. The justification for fixing "
        "width and stride in the reduced grids.",
    ),
    "k_curve": (k_curve, "Score against cluster count, per representation."),
    "density": (
        density_curve,
        "Score against segment density, showing the criterion is single-peaked.",
    ),
    "iou_sweep": (iou_sweep, "Whether the ordering survives a tighter IoU threshold."),
}


def build(name: str, rows: Iterable[C.Cell], cfg: DatasetConfig):
    """Build one figure by name."""
    if name not in FIGURES:
        raise KeyError(f"unknown figure {name!r}; have {', '.join(FIGURES)}")
    return FIGURES[name][0](list(rows), cfg)
