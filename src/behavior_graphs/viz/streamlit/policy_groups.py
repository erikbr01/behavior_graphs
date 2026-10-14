"""Which policies the paper compares, and what to call them.

Configuration for the Policy Comparison page, kept out of the page script so it
can be read and tested without a Streamlit context — a typo in a dataset name
here is silent otherwise, showing up as one policy keeping its raw name in a
figure nobody re-checked.

Two things live here. :data:`DEFAULT_GROUPS` is which datasets are compared
against which: a group is a family of tests, every pair inside it compared and
the correction applied over those pairs only, so a group has to be one task —
comparing across tasks compares task difficulty rather than policies.
:data:`PAPER_LABELS` is what each dataset is called on the figure.

Both are defaults. The page lets either be edited, and nothing here is baked
into an exported figure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fnmatch import fnmatch

#: The groups the paper compares, in the order its figure shows them: the two
#: tasks the policies do well, then the one they do not.
#:
#: Patterns are matched with :mod:`fnmatch` against the dataset names on disk,
#: so a run collected later joins its group without an edit here. A plain name
#: with no wildcard matches only itself, which is how ``prepare_tray`` picks its
#: three and leaves ``_dp`` and ``_rgrl`` out.
DEFAULT_GROUPS: dict[str, tuple[str, ...]] = {
    "Set table": ("set_table_*_pretrained_dit",),
    "Prepare tray": (
        "prepare_tray_baseline",
        "prepare_tray_plusbottom",
        "prepare_tray_plustop",
    ),
    "Cooking": ("cooking_baseline_*",),
}

#: The label every group's baseline carries, and the reason it is worth a
#: constant: it is how a page finds the baseline among a group's arms, having
#: already thrown the dataset names away. Not derivable from the names —
#: ``cooking_baseline_carrot_dit`` is a *variant* of the cooking baseline.
BASELINE_LABEL = "baseline"

#: What a dataset is called on the figure and in the tables.
#:
#: Every group's baseline is just "baseline": a group is its own comparison, so
#: the name only has to be unambiguous within one panel, and three panels each
#: with a "baseline" is easier to read across than three baselines with
#: different names. The rest say what was *added* to that baseline, which is the
#: comparison the paper is making — the dataset names say where the extra
#: demonstrations came from, and the labels say it in words a caption can use —
#: the object that moved and where it started, without the noun the task
#: description already gives ("milk left", not "milk carton left").
PAPER_LABELS: dict[str, str] = {
    "set_table_baseline_pretrained_dit": "baseline",
    "set_table_plusleft_pretrained_dit": "plus milk left",
    "set_table_plusmiddle_pretrained_dit": "plus milk middle",
    "set_table_plusright_pretrained_dit": "plus milk right",
    "prepare_tray_baseline": "baseline",
    "prepare_tray_plusbottom": "plus bowl bottom",
    "prepare_tray_plustop": "plus bowl top",
    "cooking_baseline_dit_200": "baseline",
    "cooking_baseline_carrot_dit": "plus carrot",
    "cooking_baseline_corn_dit": "plus corn",
    "cooking_baseline_lid_dit": "plus lid",
}


def resolve(patterns: Sequence[str], available: Sequence[str]) -> list[str]:
    """The datasets matching *patterns*, in the order they were asked for.

    De-duplicated, because two patterns of one group can name the same dataset
    and an arm drawn twice is a policy compared against itself.
    """
    found: list[str] = []
    for pattern in patterns:
        for name in available:
            if name not in found and fnmatch(name, pattern):
                found.append(name)
    return found


def default_groups(available: Sequence[str]) -> dict[str, list[str]]:
    """:data:`DEFAULT_GROUPS` against the datasets that actually exist.

    A group whose patterns match nothing is left out — this page runs on
    machines that have none of these datasets, and an empty group there would
    be three quarters of the screen explaining itself.
    """
    return {
        label: names
        for label, patterns in DEFAULT_GROUPS.items()
        if (names := resolve(patterns, available))
    }


def paper_label(name: str, labels: Mapping[str, str] | None = None) -> str | None:
    """What to call *name* on the figure, or ``None`` if nothing is known."""
    return (PAPER_LABELS if labels is None else labels).get(name)
