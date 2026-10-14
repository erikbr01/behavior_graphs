"""The LaTeX-export expander, shared by the pages that offer one.

Written once because two pages offering "the same" export is how they stop
being the same: the Failure Detection panel grew a decimals dial, a mean±std
merge, real bolding and raw-LaTeX captions, while Progress Estimation still had
a simpler local version. Both now render through
:func:`behavior_graphs.viz.latex_table.booktabs_table`, which is the copy with the
bolding rules (a denominator, a count and a spread are never "the best").
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from behavior_graphs.viz.latex_table import booktabs_table


def latex_export_panel(
    display: pd.DataFrame,
    score_columns: list[str],
    *,
    filename: str,
    default_caption: str,
    default_label: str,
    rank_rows: bool = True,
    group_column: str | None = None,
    key: str = "tex",
) -> str:
    """An expander offering exactly the table above, as pasteable LaTeX.

    The frame handed in must be the *unrounded* one, so the precision dial is
    the only thing deciding how many decimals come out -- rounding for the
    screen first would cap it.

    ``group_column`` is handed straight to
    :func:`~behavior_graphs.viz.latex_table.booktabs_table`, which then emits that
    column as ``\\multirow`` blocks -- so a table drawn on screen as grouped
    rows exports as grouped rows rather than as a repeated label.

    ``rank_rows`` says whether there is one row per method, which is what makes
    "bold the best" mean anything. When something else survives into the rows --
    a seed, a layer, a cluster count -- the best row names that setting rather
    than a method, so the option is withheld instead of being offered and
    quietly misleading.
    """
    with st.expander("Export as LaTeX (booktabs)"):
        left, right = st.columns(2)
        precision = left.number_input(
            "Decimals", min_value=0, max_value=6, value=3, step=1, key=f"{key}_prec"
        )
        # Nothing to merge unless the frame carries mean_/std_ pairs.
        mergeable = any(c.startswith("mean_") for c in display.columns)
        merge_std = right.checkbox(
            "Combine mean ± std",
            value=mergeable,
            disabled=not mergeable,
            help="One `$0.812 \\pm 0.041$` cell per metric instead of two columns.",
            key=f"{key}_merge",
        )
        rank_note = (
            "Several rows per method here, so the best of them would name a swept "
            "setting rather than a method."
        )
        compact = left.checkbox(
            "Compact (fits one column)",
            value=False,
            help="Trims the outer padding, narrows `\\tabcolsep`, drops the "
            "leading zero from `0.945`, and sets Kendall/AUROC as `$\\tau$`/`AUC`. "
            "Measured on the rubric table under IEEEtran: 326 pt as-is, 247 pt "
            "compact, against a 252 pt column.",
            key=f"{key}_compact",
        )
        size = right.selectbox(
            "Table font size",
            ["(document default)", "small", "footnotesize"],
            index=0,
            help="Applied inside the float. Most IEEE tables are set `\\small`; "
            "on a wide table it is worth more width than any other setting here "
            "and costs no information.",
            key=f"{key}_size",
        )
        abbreviate = right.checkbox(
            "Abbreviate headers",
            value=True,
            help="`$\\tau$` for Kendall and `AUC` for AUROC, under Compact. Worth "
            "about 20 pt of an IEEE column — the header is what sets those "
            "columns' width, since `Kendall` is wider than the `0.930` under it. "
            "In full they fit at `footnotesize` but not at `small`.",
            key=f"{key}_abbrev",
        )
        nozeros = left.checkbox(
            "Drop leading zeros",
            value=False,
            help="`.945` for `0.945` — right for a metric bounded in [-1, 1]. "
            "Worth about 4 pt of an IEEE column with three metric columns, 9 pt "
            "with four; reach for it only if the table is that close.",
            key=f"{key}_nozeros",
        )
        bold_best = right.checkbox(
            "Bold the best method per metric",
            value=rank_rows,
            disabled=not rank_rows,
            help=None if rank_rows else rank_note,
            key=f"{key}_bold",
        )
        caption = left.text_input(
            "Caption",
            value=default_caption,
            # Raw rather than escaped: a paper caption wants \cite{} and $k$ to
            # work. Only the generated default is escaped, because a dataset
            # name is not LaTeX to begin with.
            help="Raw LaTeX: `\\cite{}` and `$k$` work, and a literal `%`, `&` "
            "or `_` needs a backslash.",
            key=f"{key}_caption",
        )
        label = left.text_input("Label", value=default_label, key=f"{key}_label")

        latex = booktabs_table(
            display,
            score_columns=score_columns,
            caption=caption or None,
            label=label or None,
            precision=int(precision),
            merge_std=merge_std,
            bold_best=bold_best,
            group_column=group_column,
            compact=compact,
            strip_leading_zero=nozeros,
            font_size=None if size.startswith("(") else size,
            abbreviate_headers=abbreviate,
        )
        st.code(latex, language="latex")
        st.download_button(
            "Download .tex",
            latex,
            file_name=filename,
            mime="text/x-tex",
            icon=":material/download:",
            key=f"{key}_dl",
        )
    return latex


__all__ = ["latex_export_panel"]
