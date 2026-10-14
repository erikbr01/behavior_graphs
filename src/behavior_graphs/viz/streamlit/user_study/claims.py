"""The study's numbers in the form a paper sentence needs them.

Every panel elsewhere answers "what do these responses look like". This answers
"what may we write", which is a different question with two extra obligations:

* **A denominator with every share.** "89% preferred the graph" is 16 of 18 at
  these sample sizes, where one participant is 5.6 percentage points, and a
  percentage with the count hidden invites a reader to assume a study an order
  of magnitude larger than this one. The denominator is :func:`study_responses`
  -- the completed participants of the reported layout, and nobody else.
* **A unit that names what was measured.** The participant picked a value on a
  0--100 scale shown in steps of 20 (:mod:`...user_study.nasa_tlx`), so a
  percentage change is of the instrument's own scale and is fair to report --
  as long as the sentence says which construct: **perceived** effort and
  **perceived** time, which is what these items ask. It matters most for
  ``temporal_demand``, which this study words as how long the work took rather
  than as the canonical time-*pressure* item: "40% lower perceived time" is the
  claim, and clock time is a different number the store holds properly in
  :func:`analytics.task_duration_rows`. Differences in POINTS and the shares
  whose denominator is people need no such care, which is why they come first.

The rating claims are **paired within participant**: the two randomized stages
give every participant exactly one graph rating and one videos rating, so the
difference can be read per person rather than between groups. The pair is across
two different tasks, which is why the adjusted model
(:mod:`~behavior_graphs.viz.streamlit.user_study.mixed_effects`) is what conditions
on task -- but the paired direction counts need no model at all, and on the live
store the two land within about two points of each other (-25.6 raw against
-26.1 adjusted for temporal demand, -21.1 against -22.9 for effort, +2.2 against
+2.5 for confidence), which is the fact that lets a paper show the raw figure as
its evidence and cite the model as a check rather than as the argument.

Nothing here fits anything, so none of it needs the optional analysis extra.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence

from behavior_graphs.viz.streamlit.user_study import (
    analytics,
    figure_export,
    mixed_effects,
    regret,
)

#: Which way each reported outcome improves, for a renderer that has no axis
#: to put an arrow on — a table cell, a sentence. The rated dimensions'
#: directions are the instrument's (:data:`analytics.BETTER_DIRECTION`); regret
#: is a distance to the best option available, so less of it is a better
#: decision.
#:
#: Carried on every claim row rather than looked up by the caller, because a
#: row and its direction going through different code paths is how a table ends
#: up saying that more perceived effort is the better display.
BETTER_DIRECTION = {**analytics.BETTER_DIRECTION, regret.DECISION_METRIC: "down"}

#: The outcomes the paper reports, in the order it reports them.
CLAIM_OUTCOMES = ("temporal_demand", "effort", "confidence")


def _share(part: int, whole: int) -> float | None:
    """``part`` as a percentage of ``whole``, or ``None`` when nothing was asked."""
    return 100 * part / whole if whole else None


def study_responses(responses: Iterable[dict]) -> list[dict]:
    """The participants the study reports: finished, and on the reported layout.

    Three filters, and the third is the one that is easy to forget. The store
    holds the lab app's previews, the sessions that were abandoned part-way, and
    the *earlier layouts* -- and the earlier layouts are the dangerous ones,
    because a pilot session that ran to the end is complete by any test the
    store can apply while having answered different questions of a different
    person. Two such sessions sat in the denominator here and made "16 of 18"
    read as 80% of 20.

    Idempotent, so a caller that has already narrowed its responses can pass
    them straight through.
    """
    return analytics.for_version(
        analytics.completed(analytics.from_participants(responses)),
        analytics.THREE_STAGE_VERSION,
    )


def preference_claim(responses: Iterable[dict]) -> dict:
    """Who picked which display when the choice was theirs.

    ``n_participants`` is the study's own denominator and ``n_chose`` is how
    many of them recorded a choice; the shares are of the second, and the two
    being equal is worth checking rather than assuming.
    """
    responses = study_responses(responses)
    # `figure_export.preference_counts`, not a second reading of the response:
    # it resolves the chosen stage through `analytics.stage_rows` exactly as the
    # panels do, and a claim that disagreed with the figure beside it about who
    # picked what would be the worst possible bug to ship in a paper.
    counts = figure_export.preference_counts(responses)
    chose = sum(counts.values())
    return {
        "counts": counts,
        "n_chose": chose,
        "n_participants": len(responses),
        "share": {phase: _share(n, chose) for phase, n in counts.items()},
    }


def paired_rating_claims(responses: Iterable[dict]) -> list[dict]:
    """Per outcome: the within-participant graph-versus-videos comparison.

    Only participants with exactly one rating of each display for that outcome
    -- which the randomized stages give -- so every row of the pair is one
    person deciding about the same thing twice. A participant missing one side
    is left out rather than compared against the group mean.

    ``lower``/``higher``/``tied`` count participants, not ratings, and are the
    claim to reach for first: they need no scale assumption, and "higher for
    none of them" is a stronger sentence than any mean.
    """
    by_participant: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in mixed_effects.assigned_rating_rows(responses):
        key = (row["participant"], row["outcome"])
        side = "graph" if row["graph"] else "videos"
        by_participant.setdefault(key, {"graph": [], "videos": []})[side].append(row["score"])

    claims = []
    for outcome in CLAIM_OUTCOMES:
        pairs = [
            (sides["graph"][0], sides["videos"][0])
            for (_participant, name), sides in by_participant.items()
            if name == outcome and len(sides["graph"]) == 1 and len(sides["videos"]) == 1
        ]
        if not pairs:
            continue
        graph = [value for value, _ in pairs]
        videos = [value for _, value in pairs]
        mean_graph, mean_videos = statistics.mean(graph), statistics.mean(videos)
        claims.append(
            {
                "outcome": outcome,
                "label": analytics.TLX_LABELS.get(outcome, outcome),
                "better": BETTER_DIRECTION.get(outcome),
                "n_pairs": len(pairs),
                "mean_graph": mean_graph,
                "mean_videos": mean_videos,
                "difference": mean_graph - mean_videos,
                # Of the RAW videos mean. If a sentence carries an interval it
                # has to carry the fit's percentage instead -- see
                # `adjusted_rating_claims`, whose divisor is the model's own
                # marginal mean. The two are within a point of each other here,
                # which is exactly why quoting one with the other's interval
                # would go unnoticed.
                "relative": (
                    100 * (mean_graph - mean_videos) / mean_videos if mean_videos else None
                ),
                "lower": sum(1 for g, v in pairs if g < v),
                "higher": sum(1 for g, v in pairs if g > v),
                "tied": sum(1 for g, v in pairs if g == v),
            }
        )
    return claims


def adjusted_rating_claims(effects: Iterable[Mapping]) -> list[dict]:
    """The fitted estimates in the shape :func:`paired_rating_claims` returns.

    *effects* are
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.continuous_effects`
    -- passed in rather than fitted here, so this module still needs no sampler
    and no optional extra.

    The reason to have it at all is the **percentage**. A sentence that carries
    the model's interval has to carry the model's point estimate, and if it also
    carries a relative figure then that has to be of the model's own marginal
    mean rather than of the raw one. The two divisors differ by under a point on
    the live store (-40.3% adjusted against -39.7% raw for temporal demand),
    which is small enough that mixing them would never be noticed and is no
    reason to do it.
    """
    claims = []
    for effect in effects:
        outcome = str(effect.get("outcome", ""))
        marginal = effect.get("marginal_means") or {}
        videos = (marginal.get("videos") or {}).get("mean")
        graph = (marginal.get("graph") or {}).get("mean")
        claims.append(
            {
                "outcome": outcome,
                "label": analytics.TLX_LABELS.get(outcome, outcome),
                "better": BETTER_DIRECTION.get(outcome),
                "mean_graph": graph,
                "mean_videos": videos,
                "difference": effect.get("mean"),
                "lower": effect.get("lower"),
                "upper": effect.get("upper"),
                # Of the adjusted videos mean, so this and the interval above
                # come off one fit.
                "relative": (
                    100 * float(effect["mean"]) / videos
                    if videos and effect.get("mean") is not None
                    else None
                ),
                "r_hat": effect.get("r_hat"),
            }
        )
    return claims


def adjusted_decision_claim(regret_effect: Mapping | None) -> dict | None:
    """The fitted decision estimate, in :func:`adjusted_rating_claims`' shape.

    So the outcome the study is actually about is a row of the same table as the
    three self-reported ones, rather than a number a reader has to go and find
    in a figure. Same model, same estimand: the ``graph`` coefficient with its
    95% HDI, and the adjusted mean each display led to.

    *regret_effect* is
    :func:`~behavior_graphs.viz.streamlit.user_study.mixed_effects.decision_regret_effect`
    — passed in rather than fitted here, so this module still needs no sampler.
    ``None`` when there is nothing fitted to report, which is what a deployment
    without the retrained rollouts mounted has: scoring a choice needs every
    option's policy on disk.

    **Its points are not the other rows' points.** Regret is in points of policy
    success rate and the ratings are on the 0--100 scale participants answered
    on, so the table has to say which row is which — and this one improves
    *downward*, like two of the three above it: less regret is a better
    decision. The relative figure is of the adjusted videos mean, as it is
    there, and it is a ratio of two small numbers here: read it with the
    interval beside it, not on its own.
    """
    if not regret_effect or regret_effect.get("mean") is None:
        return None
    marginal = regret_effect.get("marginal_means") or {}
    videos = (marginal.get("videos") or {}).get("mean")
    graph = (marginal.get("graph") or {}).get("mean")
    return {
        "outcome": regret.DECISION_METRIC,
        # The name the figures give this panel, read off them rather than
        # spelled again: a table row and the panel it refers to calling the
        # same quantity two things is a reader's worst five minutes.
        "label": figure_export.STUDY_PANEL_TITLES["regret"],
        "better": BETTER_DIRECTION[regret.DECISION_METRIC],
        "mean_graph": graph,
        "mean_videos": videos,
        "difference": regret_effect.get("mean"),
        "lower": regret_effect.get("lower"),
        "upper": regret_effect.get("upper"),
        "relative": (100 * float(regret_effect["mean"]) / videos if videos else None),
        "r_hat": regret_effect.get("r_hat"),
    }


def decision_claims(rows: Sequence[Mapping]) -> dict:
    """What the data-collection choices were worth, per display and per task.

    *rows* are :func:`~behavior_graphs.viz.streamlit.user_study.regret.regret_rows`.

    Two different questions, over two different sets, which is why this returns
    both:

    * **How well did they choose?** ``pooled`` and ``by_task``: mean regret, and
      how often the best option was picked. The per-task rows are not
      decoration -- the options span 33 success-rate points on set table and 8
      on cooking, so a pooled mean leans toward the widest -- and
      ``share_positive`` answers the separate question of whether the choice
      helped at all, which regret cannot: cooking's best arm is worth +3.3
      points and two of its three are negative.
    * **Did the display change them?** ``by_display``, over the **randomized**
      stages only (``n_randomized``). The three-stage layout's last stage is the
      participant's own pick, and 16 of 18 picked the graph, so pooling it makes
      the display a result of the participant rather than something drawn for
      them. The adjusted fit uses the randomized stages, so this is also what
      makes the two comparable.

    A row with no ``assignment`` -- the layouts that never assigned a display --
    counts as randomized, since for them there was no choice to contaminate it.
    """

    def _values(subset: Sequence[Mapping], key: str) -> list[float]:
        return [float(row[key]) for row in subset if row.get(key) is not None]

    def _summary(subset: Sequence[Mapping]) -> dict:
        # Regret is the decision measure (`regret.DECISION_METRIC`) and is
        # defined for every scored choice. Gain over the baseline is kept for
        # the one question regret cannot answer -- did the choice help at all,
        # against reading the same policy and collecting nothing -- and it is
        # `None` wherever the task's baseline has no rollouts, which is why the
        # two have separate denominators.
        regrets = _values(subset, regret.DECISION_METRIC)
        gains = _values(subset, "gain_over_baseline")
        best = sum(1 for row in subset if row.get("chose_best"))
        return {
            "n": len(subset),
            "mean_regret": statistics.mean(regrets) if regrets else None,
            "n_positive": sum(1 for gain in gains if gain > 0),
            "share_positive": _share(sum(1 for gain in gains if gain > 0), len(gains)),
            "n_best": best,
            "share_best": _share(best, len(subset)),
        }

    rows = list(rows)
    randomized = regret.randomized_rows(rows)
    tasks = [task for task in regret.TASK_ORDER if any(row["task"] == task for row in rows)]
    return {
        "pooled": _summary(rows),
        "n_randomized": len(randomized),
        "by_display": {
            phase: _summary([row for row in randomized if row["phase"] == phase])
            for phase in regret.PHASE_ORDER
            if any(row["phase"] == phase for row in randomized)
        },
        "by_task": {task: _summary([row for row in rows if row["task"] == task]) for task in tasks},
    }
