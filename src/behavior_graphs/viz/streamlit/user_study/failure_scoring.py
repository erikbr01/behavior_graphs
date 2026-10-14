"""Scoring a participant's failure-mode ranking against what actually failed.

The study asks each participant to rank the failure modes of a policy they have
just watched or read a graph of. Until now the analytics page could only show
how those rankings compared *to each other* — which mode got ranked highly, by
whom. There was no correct answer to compare them to.

There is now: every failed episode of the three study rollouts carries a
hand-written failure mode
(:mod:`behavior_graphs.annotations.failure_modes`), and those modes are the
study's own option wordings, so the two join on the string. This turns the
ranking from a preference into an answer that can be right or wrong, which is
what the paper's claim about the graph rests on.

**Four numbers, because no one of them is the answer.**

``top1_hit``
    Did rank 1 name the most common real failure? The blunt reading, and the
    only one a reader needs no explanation for. Ties count for either mode.

``coverage``
    What share of the failed episodes the participant's picks account for. The
    one that matches what the ranking is *for*: somebody who names the two modes
    behind 80% of the failures has understood the policy, whatever order they
    put them in.

``achievable``
    The coverage of the best possible picks of that many modes — the ceiling.
    Reported beside coverage rather than folded into it, because the ceiling is
    a fact about the task and not about the participant: where one mode is 96%
    of failures, everybody's coverage is high and the number says little.

``chance``
    The coverage expected from picking that many modes at random, which is
    exactly ``picks / bank size`` — every mode is equally likely to be included,
    and the shares sum to one. The floor, and it differs sharply by task:
    cooking offers ten modes and set table four, so the same coverage means
    different things in each.

``order_tau``
    Kendall's tau-b between the order the participant put their picks in and the
    order their true frequencies imply. Only over the modes they picked, and
    undefined (``None``) when those are all equally common — there is no order
    to get right then.

Pure: annotations and responses in, rows out, no Streamlit. The page reads
:func:`score_rows` and :func:`summarise`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from behavior_graphs.annotations import failure_modes as annotations

#: The study variant these rollouts and question banks belong to. Every response
#: records the variant it was collected under, so a session from another one is
#: skipped rather than scored against the wrong ground truth.
DEFAULT_VARIANT = "three_stage_v1"


@dataclass(frozen=True)
class Score:
    """One participant's ranking for one task, scored. See the module docstring."""

    picks: int
    top1_hit: bool
    coverage: float
    achievable: float
    chance: float
    order_tau: float | None

    @property
    def coverage_ratio(self) -> float:
        """Coverage as a share of what was achievable with this many picks."""
        return self.coverage / self.achievable if self.achievable else 0.0


def ground_truth(rollout_dir: Path | str) -> dict[str, int]:
    """``{failure mode: episodes}`` for one rollout dataset.

    Straight from the annotation store, so a mode annotated after this page was
    last opened is counted the next time it is.
    """
    counts: dict[str, int] = {}
    for annotation in annotations.load_all(rollout_dir).values():
        counts[annotation.mode] = counts.get(annotation.mode, 0) + 1
    return counts


def task_rollout_dirs(variant_id: str = DEFAULT_VARIANT) -> dict[str, Path]:
    """``{task: rollout dir}`` for a variant, via its stages' session configs.

    Through the configs rather than a table here, so a variant that swaps a
    stage's dataset is scored against the dataset it actually showed.
    """
    import yaml

    from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
    from behavior_graphs.viz.streamlit.user_study.sessions import resolve_path, sessions_dir

    variant = get_variant(variant_id)
    if variant is None:
        return {}
    directory = sessions_dir()
    dirs: dict[str, Path] = {}
    for stage in variant.stages:
        config = directory / f"{stage.session_name}.yaml"
        if not config.is_file():
            continue
        raw = (yaml.safe_load(config.read_text()) or {}).get("rollout_dir")
        if raw:
            dirs[stage.task] = resolve_path(str(raw))
    return dirs


def question_banks(variant_id: str = DEFAULT_VARIANT) -> dict[str, list[str]]:
    """``{task: the failure modes it offered}``, by the stage's question set.

    The bank size is what :data:`Score.chance` is computed from, so it has to be
    the list the participant actually chose from rather than the modes that
    happened to occur.
    """
    from behavior_graphs.viz.streamlit.user_studies.registry import get_variant
    from behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions import (
        QUESTION_SETS,
    )

    variant = get_variant(variant_id)
    if variant is None:
        return {}
    banks = {}
    for stage in variant.stages:
        spec = QUESTION_SETS.get(stage.question_set or stage.task)
        if spec:
            banks[stage.task] = list(spec["failure_modes"])
    return banks


def load_ground_truth(variant_id: str = DEFAULT_VARIANT) -> dict[str, dict[str, int]]:
    """``{task: {mode: episodes}}`` for every stage that has annotations."""
    truth = {}
    for task, rollout_dir in task_rollout_dirs(variant_id).items():
        counts = ground_truth(rollout_dir)
        if counts:
            truth[task] = counts
    return truth


def score(
    ranked: Sequence[str],
    truth: Mapping[str, int],
    bank: Sequence[str] = (),
) -> Score | None:
    """Score one ranking, or ``None`` when there is nothing to score it against.

    ``None`` for an empty ranking or a dataset with no annotated failures —
    rather than a zero, which would read as a participant who got everything
    wrong.
    """
    total = sum(truth.values())
    # De-duplicated: the study will not let a participant pick one mode twice,
    # but a mode counted twice would put coverage above 100% and there is no
    # reading of this figure in which that is not a bug.
    picks = list(dict.fromkeys(str(mode) for mode in ranked if mode))
    if not picks or total <= 0:
        return None

    # Ties count for either mode: two modes at 40% each are both "the" answer.
    hit = truth.get(picks[0], 0) == max(truth.values())
    covered = sum(truth.get(mode, 0) for mode in picks)
    best = sum(sorted(truth.values(), reverse=True)[: len(picks)])
    return Score(
        picks=len(picks),
        top1_hit=hit,
        coverage=covered / total,
        achievable=best / total,
        chance=len(picks) / len(bank) if bank else 0.0,
        order_tau=_order_tau(picks, truth),
    )


def _order_tau(picks: Sequence[str], truth: Mapping[str, int]) -> float | None:
    """Kendall tau-b between the picked order and the order the counts imply.

    Over the picks only. A participant ranks two or three of a bank of ten, so
    correlating against the whole bank would mostly measure the modes they were
    never asked about.
    """
    if len(picks) < 2:
        return None
    counts = [truth.get(mode, 0) for mode in picks]
    if len(set(counts)) < 2:
        # Every pick equally common: no order to get right, so no score.
        return None
    from scipy.stats import kendalltau

    # Negated, so agreement with descending frequency is +1.
    tau = kendalltau(list(range(len(picks))), [-count for count in counts]).statistic
    return None if tau != tau else float(tau)  # NaN-safe


def score_rows(
    responses: Iterable[Mapping],
    truth_by_task: Mapping[str, Mapping[str, int]],
    banks: Mapping[str, Sequence[str]] | None = None,
) -> list[dict]:
    """One scored row per (response, task).

    From the responses, *not* from :func:`analytics.ranking_rows`. That reuse
    was a trap while the two disagreed about who a response came from: those
    rows identified a participant by ``participant_id`` alone, and the study
    collects anonymously, so every session in the store was ``"anon"`` and
    grouping rankings back together by participant concatenated all of them
    into one per task. It did not fail loudly either — it reported coverage
    above 100%, which is how it was caught. :func:`analytics.participant` is
    the one reading now and this reads through it too, so the identities agree;
    scoring straight off the responses is kept because a scored row needs the
    raw ranking and the bank, which those rows do not carry.

    The display each task was read in still comes from
    :func:`analytics.stage_modalities`, the same resolution every other panel
    on the page uses.
    """
    from behavior_graphs.viz.streamlit.user_study import analytics

    banks = banks or {}
    rows = []
    for response in responses:
        understanding = response.get("understanding_responses")
        if not isinstance(understanding, dict):
            continue
        by_task = analytics.stage_modalities(response)
        for key, answers in understanding.items():
            # The layout's first build wrote this block flat, so a value here
            # can be the answer rather than a task's answers.
            if not isinstance(answers, dict):
                continue
            ranked = answers.get("failure_mode_ranking")
            task = str(key)
            if not isinstance(ranked, list) or task not in truth_by_task:
                continue
            scored = score(ranked, truth_by_task[task], banks.get(task, ()))
            if scored is None:
                continue
            rows.append(
                {
                    "response": str(response.get("_response_id", "") or ""),
                    "participant": analytics.participant(response),
                    "task": task,
                    "phase": by_task.get(task, ""),
                    "group_label": analytics.group_label(response),
                    "picks": scored.picks,
                    "top1_hit": scored.top1_hit,
                    "coverage": scored.coverage,
                    "achievable": scored.achievable,
                    "coverage_ratio": scored.coverage_ratio,
                    "chance": scored.chance,
                    "order_tau": scored.order_tau,
                    "first_pick": str(ranked[0]) if ranked else "",
                }
            )
    return rows


def summarise(rows: Iterable[Mapping], by: Sequence[str] = ("task", "phase")) -> list[dict]:
    """Mean scores per group of *rows*, grouped by *by*.

    Plain Python rather than a dataframe so this is testable without pandas and
    reads the same as the numbers it produces: ``order_tau`` is averaged over
    the rankings that *have* one, since a task where every pick is equally
    common contributes no order to get right rather than a zero.
    """
    grouped: dict[tuple, list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key, "") for key in by), []).append(row)

    summary = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        taus = [row["order_tau"] for row in group if row.get("order_tau") is not None]
        summary.append(
            {
                **dict(zip(by, key, strict=False)),
                "n": len(group),
                "top1_hit_rate": sum(bool(row["top1_hit"]) for row in group) / len(group),
                "coverage": sum(row["coverage"] for row in group) / len(group),
                "achievable": sum(row["achievable"] for row in group) / len(group),
                "coverage_ratio": sum(row["coverage_ratio"] for row in group) / len(group),
                "chance": sum(row["chance"] for row in group) / len(group),
                "order_tau": (sum(taus) / len(taus)) if taus else None,
                "order_tau_n": len(taus),
            }
        )
    return summary


def dominant_share(truth: Mapping[str, int]) -> float:
    """The share of failures taken by the single most common mode.

    Near 1 means the ranking had almost nothing to discriminate: everybody's
    rank 1 is the same mode and every rank below it is noise. Worth saying on
    the page next to a task's scores rather than leaving a reader to infer that
    a 96% hit rate was a 96% giveaway.
    """
    total = sum(truth.values())
    return max(truth.values()) / total if total else 0.0
