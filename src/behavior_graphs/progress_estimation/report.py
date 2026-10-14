"""Shared table builders for progress-estimation results.

``sweep_table_md`` is used by both ``scripts/sweep_progress_allds.py`` (which
produces ``sweep_results.json``) and ``scripts/merge_progress_sweep.py`` (which
merges two such files and adds an embedding column), so the best-K-per-dataset
table is defined in exactly one place.

``booktabs_table`` renders an already-display-formatted metrics frame as LaTeX.
It lives here rather than in the Streamlit page that first needed it so that the
paper-facing scripts can call it too, and so it is reachable from a test -- a
page module cannot be imported without executing its whole Streamlit body.
"""

from __future__ import annotations

import pandas as pd

# Scheme keys stored in sweep_results.json (kept stable for backward compatibility
# with already-saved sweeps and the consolidate/merge consumers).
STAGE = "graph_direct:stage"
ABSORB = "graph_direct:absorb"

_DEFAULT_TITLE = "Cross-dataset progress estimation (graph, identity, best-K by stage-VOC, 3-fold)"
_DEFAULT_SUBTITLE = (
    "graph:stage = SARM stage progress; graph:absorb = P(absorb@SUCCESS); "
    "frame_index = trivial linear baseline (perfect VOC/MAE, no success detection)."
)

#: Printed under every table, because the bestK column is easy to read as a
#: held-out result and is not one: K is chosen by maximising the very stage VOC
#: on the same row, over every K swept, with no inner split.
_SELECTION_NOTE = (
    "bestK maximises the stage VOC reported beside it, over every K swept and with no "
    "inner split, so that row is selected on the metric rather than held out. "
    "`stage_voc_median_over_k` in sweep_results.json is selected on nothing."
)


def sweep_table_md(
    results: dict,
    title: str = _DEFAULT_TITLE,
    subtitle: str = _DEFAULT_SUBTITLE,
    show_emb: bool = False,
) -> str:
    """Best-K row per dataset from a ``{name: sweep_result}`` mapping.

    With ``show_emb`` an embedding-layer column is inserted (used by the merged
    policy-embedding table).
    """
    emb_h = "emb | " if show_emb else ""
    lines = [
        f"# {title}",
        "",
        subtitle,
        "",
        _SELECTION_NOTE,
        "",
        f"| dataset | family | {emb_h}eps (s/f) | meanlen | bestK | stage VOC | stage Pear | "
        "stage MAE | stage AUROC | absorb AUROC |",
        "|---|---|---|---|---|---|---|---|---|" + ("---|" if show_emb else "") + "--|",
    ]
    for name, r in results.items():
        if "per_k" not in r:
            note = r.get("skipped") or ("ERROR" if "error" in r else "?")
            emb_c = " |" if show_emb else ""
            lines.append(f"| {name} | {r.get('family', '')} |{emb_c} — | — | — | {note} | | | | |")
            continue
        # per_k keys are ints in-memory, strings once round-tripped through JSON
        pk = r["per_k"]
        # `best_k` is None when no K produced a finite stage VOC, and the sweep
        # still writes `per_k` -- so the "per_k not in r" guard above lets that
        # row through, `pk.get(None, pk.get("None"))` yields None, and `kd[STAGE]`
        # raised TypeError while writing the table, AFTER every dataset had been
        # fitted. A dataset nothing could score belongs in the same row shape as
        # a skipped one.
        kd = pk.get(r["best_k"], pk.get(str(r["best_k"]))) if r.get("best_k") is not None else None
        if kd is None:
            emb_c = " |" if show_emb else ""
            lines.append(
                f"| {name} | {r.get('family', '')} |{emb_c} — | — | — | "
                f"no finite stage VOC at any K | | | | |"
            )
            continue
        st, ab = kd[STAGE], kd[ABSORB]
        emb_c = f"{r.get('layer', '').replace('.npz', '')} | " if show_emb else ""
        lines.append(
            f"| {name} | {r['family']} | {emb_c}{r['n_succ']}/{r['n_fail']} | "
            f"{r['mean_len']:.0f} | {r['best_k']} | {st['voc']:.3f} | {st['pearson']:.3f} | "
            f"{st['mae']:.3f} | {st['auroc']:.3f} | {ab['auroc']:.3f} |"
        )
    return "\n".join(lines) + "\n"


# Backslash first is not enough on its own -- str.translate maps each ORIGINAL
# character exactly once, so an escape that itself contains specials is safe.
_LATEX_ESCAPES = str.maketrans(
    {
        "\\": r"\textbackslash{}",
        "~": r"\textasciitilde{}",  # bare ~ is a non-breaking space
        "^": r"\textasciicircum{}",  # bare ^ is "Missing $ inserted"
        **{c: f"\\{c}" for c in "&%$#_{}"},
    }
)


def latex_escape(text: object) -> str:
    """Escape LaTeX's special characters in a plain string."""
    return str(text).translate(_LATEX_ESCAPES)


def _latex_cell(value: object) -> str:
    """One display cell -> LaTeX, preserving the markdown bold the table uses."""
    text = str(value)
    bold = text.startswith("**") and text.endswith("**")
    if bold:
        text = text[2:-2]
    text = latex_escape(text).replace("±", r"$\pm$").replace("—", "--")
    return rf"\textbf{{{text}}}" if bold else text


def chose_k_off_eval(entry: dict) -> bool:
    """Whether a sweep entry's K was chosen WITHOUT looking at the eval fold.

    The one definition of "reportable" for these runs, imported by the app page
    and by both document generators -- a second copy is how one of them starts
    publishing a number the others refuse to.

    Two spellings, because the nested path grew after the flat one: it writes
    ``selection: "nested"`` and, having no single winning K to name, the
    sentinel ``best_k: "selected"``. Anything else is a K that maximised the
    very metric being reported, across the folds being reported -- an
    optimistic bias measured at 0.004-0.048 VOC per family on the 54-dataset
    sweep, and not a number to put in a paper.
    """
    return entry.get("selection") == "nested" or entry.get("best_k") == "selected"


def booktabs_table(
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    higher_better: dict[str, bool],
    index_name: str = "Method",
    baseline_suffix: str = "",
) -> str:
    r"""Render a DISPLAY-formatted metrics frame as a booktabs table.

    Takes the frame that is already on screen rather than re-deriving cells from
    the metrics dict, so the export cannot drift from what the reader saw -- same
    rounding, same bolding, same rows, same aggregation. Only the markup is
    translated.

    ``caption`` is inserted verbatim, so the caller owns any LaTeX in it (and any
    escaping it needs). ``baseline_suffix``, if given, is stripped off row labels
    that carry it and replaced with an ``\emph{(baseline)}`` tag -- the display
    marker is a U+00B7 middot, which plain pdflatex will not typeset.
    """
    if frame.empty:
        # frame.apply on a fully empty frame returns a Series, after which setting
        # .columns is an attribute assignment rather than a rename and the emitted
        # tabular does not match its column_format. Nothing to typeset anyway.
        return "% no rows to render\n"
    body = frame.apply(lambda col: col.map(_latex_cell))
    up, down = r"$\uparrow$", r"$\downarrow$"
    body.columns = [f"{latex_escape(c)} {up if higher_better[c] else down}" for c in frame.columns]
    labels = []
    for name in map(str, frame.index):
        if baseline_suffix and name.endswith(baseline_suffix):
            labels.append(latex_escape(name.removesuffix(baseline_suffix)) + r" \emph{(baseline)}")
        else:
            labels.append(latex_escape(name))
    body.index = labels
    # Without this pandas emits a spurious blank second header row for the index.
    body.index.name = None
    tex = (
        body.reset_index()
        .rename(columns={"index": index_name})
        .to_latex(
            index=False,
            escape=False,
            column_format="l" + "r" * len(frame.columns),
            caption=caption,
            label=label,
            position="t",
        )
    )
    return "% requires \\usepackage{booktabs}\n" + tex.replace(
        "\\begin{table}[t]\n", "\\begin{table}[t]\n\\centering\n", 1
    )
