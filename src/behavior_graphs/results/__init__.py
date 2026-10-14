"""The paper's reported results, as data.

``paper_results.json`` beside this module holds the numbers the public reviewer
page draws: the retrained policies' success rates, the user study's answers, the
mixed-effects estimates, and the progress benchmark's per-dataset metrics. It is
written by ``scripts/export_paper_results.py`` from the artifacts those results
come off, and nothing in it is typed in by hand.

**Why a file rather than a computation.** Three reasons, and each one on its
own would be enough:

* The page that reads it is served by the ``reviewer`` container, which mounts
  the rollout runs and nothing else -- no response store, no benchmark results.
  A page computing these would have nothing to compute them from.
* One of the four sections is four MCMC chains.
* A website and a paper disagreeing about a number is worse than either being
  wrong. Exported once, they cannot.

The json is the unit of review: a diff against it is a diff of every number the
site shows, which is the property that makes an update to the study's data
something a human can check.

That property is why the predicted progress curves live in a second file,
``progress_curves.json``, rather than in the first. They are 42,000 numbers
summarising 7,682 rollouts, and pretty-printed beside the rest they turned a
88 kB document a person could read into a 762 kB one nobody would. The curves
file is written compact and on one line per series, because it is bulk a chart
consumes and not a table anybody checks by eye; everything a sentence in the
paper rests on stays in ``paper_results.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

#: Bumped when a section's shape changes in a way a reader of the old one would
#: get wrong. :func:`load` refuses a document from the future, since a page
#: drawing a field that has moved draws it blank rather than loudly.
SCHEMA = 1

#: Where the export lands and where the page looks, as one constant so the two
#: cannot drift.
DEFAULT_PATH = Path(__file__).with_name("paper_results.json")

#: The predicted progress curves, kept out of the document above. Written by
#: the same ``--sections progress`` run, so the two cannot describe different
#: sets of datasets.
CURVES_PATH = Path(__file__).with_name("progress_curves.json")

#: Sections, and the section a page asks for by name. Listed so a missing one
#: names itself in the error rather than arriving as a ``KeyError`` on a key the
#: reader has to go and look up.
SECTIONS = ("policies", "study", "adjusted", "progress")

_EXPORT_COMMAND = "python scripts/export_paper_results.py"


class ResultsUnavailable(RuntimeError):
    """The export is missing, or missing the section a caller asked for."""


def load(path: Path | str | None = None) -> dict:
    """The exported document, parsed.

    Raises :class:`ResultsUnavailable` rather than returning an empty dict: a
    page that renders its chrome around no data is a page that looks like the
    study found nothing.
    """
    target = Path(path) if path is not None else DEFAULT_PATH
    try:
        document = json.loads(target.read_text())
    except FileNotFoundError as error:
        raise ResultsUnavailable(
            f"No exported results at {target}. Write them with `{_EXPORT_COMMAND}`."
        ) from error
    except json.JSONDecodeError as error:
        raise ResultsUnavailable(f"{target} is not readable JSON: {error}") from error
    schema = document.get("schema")
    if schema != SCHEMA:
        raise ResultsUnavailable(
            f"{target} is schema {schema!r}, and this code reads {SCHEMA}. "
            "Re-export it, or read it with the version of the page it was written for."
        )
    return document


def load_curves(path: Path | str | None = None) -> dict:
    """The predicted progress curves, parsed.

    Its own file and its own loader, but the same ``--sections progress`` run
    writes both, so a document and a curves file in one checkout describe the
    same 43 collections. The schema is checked the same way.
    """
    target = Path(path) if path is not None else CURVES_PATH
    try:
        document = json.loads(target.read_text())
    except FileNotFoundError as error:
        raise ResultsUnavailable(
            f"No progress curves at {target}. Write them with "
            f"`{_EXPORT_COMMAND} --sections progress`."
        ) from error
    except json.JSONDecodeError as error:
        raise ResultsUnavailable(f"{target} is not readable JSON: {error}") from error
    if document.get("schema") != SCHEMA:
        raise ResultsUnavailable(
            f"{target} is schema {document.get('schema')!r}, and this code reads {SCHEMA}."
        )
    return document


def section(document: Mapping, name: str) -> dict:
    """One section, or :class:`ResultsUnavailable` naming how to make it.

    Sections are exported independently -- they come off three different
    machines -- so a document holding three of the four is a normal state, and
    the page asking for the fourth should say which command produces it.
    """
    if name not in SECTIONS:
        raise KeyError(f"unknown section {name!r}; pick from {', '.join(SECTIONS)}")
    value = document.get(name)
    if not value:
        raise ResultsUnavailable(
            f"The exported results carry no {name!r} section. Write it with "
            f"`{_EXPORT_COMMAND} --sections {name}`."
        )
    return dict(value)


def policy_arms(task: Mapping) -> list:
    """One task's arms as :class:`~behavior_graphs.stats.policy_comparison.PolicyOutcomes`.

    The violins are a Beta posterior over the counts, so the page rebuilds the
    arms and asks the same functions the paper figure asks. The exported letters
    and p-values are *not* recomputed: Barnard's exact test is a grid search per
    pair, and the website must not be able to come out differently.
    """
    from behavior_graphs.stats.policy_comparison import PolicyOutcomes

    return [
        PolicyOutcomes(name=arm["label"], successes=arm["successes"], trials=arm["trials"])
        for arm in task["arms"]
    ]


def rows(document: Mapping, name: str, key: str = "rows") -> list[dict]:
    """``document[name][key]`` as a list of dicts, with the section's error."""
    return [dict(row) for row in section(document, name).get(key, ())]


def pick(table: Sequence[Mapping], **fields: object) -> list[dict]:
    """The rows of *table* whose fields all match — the one filter every chart does."""
    return [
        dict(row) for row in table if all(row.get(key) == value for key, value in fields.items())
    ]
