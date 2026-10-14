"""Reports over a loaded sweep. Each answers one question and prints a table.

Every module here takes a :class:`~behavior_graphs.ablation_study.config.DatasetConfig`
and returns a string, so the CLI is a dispatch table and nothing needs a
dataset path of its own.
"""

from behavior_graphs.ablation_study.report import (
    ablation,
    aggregate,
    criterion,
    crosstask,
    dials,
    heldout,
    latex,
    leakage,
    matched,
    per_task,
    proxies,
)

#: Report name -> callable(cfg, **kwargs) -> str
REPORTS = {
    "dials": dials.run,
    "ablation": ablation.run,
    "aggregate": aggregate.run,
    "plateau": dials.plateau_report,
    "proxies": proxies.run,
    "heldout": heldout.run,
    "matched": matched.run,
    "per_task": per_task.run,
    "latex": latex.run,
    "leakage": leakage.run,
    "criterion": criterion.run,
    "crosstask": crosstask.run,
}

__all__ = [
    "REPORTS",
    "ablation",
    "aggregate",
    "criterion",
    "crosstask",
    "dials",
    "heldout",
    "latex",
    "leakage",
    "matched",
    "per_task",
    "proxies",
]
