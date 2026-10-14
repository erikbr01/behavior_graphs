"""What a participant's data-collection choice cost, in policy success rate.

The study's second question asks a participant to pick the additional data they
would collect to improve the policy they have just been reading about. For every
option of every task that question offers, we went and collected that data,
retrained the policy on it, and rolled the retrained policy out 60 times — so
each option has a measured success rate, and the question has a right answer.

**Regret** here is the distance to that right answer: the success rate of the
best option for the task minus the success rate of the option the participant
picked, in success-rate points. Picking the best option is 0; picking an option
that leaves 30 points of success rate on the table is 30. It is what makes two
displays comparable on the thing the study is actually about — not whether the
graph was liked or felt quicker, but whether it led to a better decision.

Two properties of that definition are load-bearing:

- **Best among the options offered, not the best imaginable.** The baseline
  policy is not one of the options — nobody was offered "collect nothing" — so
  it is not in the max. It is reported beside the arms as context, and it earns
  that place: on the rates measured so far the best option beats it everywhere,
  while *most* options do not — for two of the three tasks, two of the three
  options leave the policy worse than collecting nothing at all. That is a fact
  about the decision which the distance to the best option cannot carry, and it
  is what ``regret_of_gain`` below exists to say.
- **Only a task whose every option was retrained can be scored at all.** With
  one arm missing, the max is a max over what happens to be on disk, and every
  regret computed against it is understated by an unknown amount. Such a task
  produces no rows (see :func:`unscored_datasets`, which is what the page says
  out loud instead of quietly scoring two of three options).

The options are matched by the exact string the response stored, which is how
the answer is recorded — see
:mod:`behavior_graphs.viz.streamlit.user_studies.variants.contrastive_same_page.questions`.
Wordings that the question bank has since dropped are kept here as aliases: what
identifies the retrained policy for ``set_table`` is where the chocolate milk
starts, and the six-option version of that question crossed that position with
the arrangement of the other two objects, so six strings name three policies.

Reading a response is :mod:`behavior_graphs.viz.streamlit.user_study.analytics`'s
job and stays there: this module takes the tidy choice rows that module returns
and scores them, so every layout that recorded a data-collection answer is
scored, and a layout that recorded none produces no rows.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from behavior_graphs.viz.streamlit.user_study import analytics


@dataclass(frozen=True)
class CollectionArm:
    """One data-collection option, and the policy retrained on that data.

    ``options`` is every string a response may have stored for this arm: the
    wording the question bank asks today first, then the superseded ones. They
    are all the same arm because they all name the same collected data — the
    label is what a participant read, the dataset is what we measured.
    """

    label: str
    dataset: str
    options: tuple[str, ...]


@dataclass(frozen=True)
class TaskArms:
    """One task's options, and the policy the study showed before any of them.

    ``baseline`` is the policy whose rollouts the participant read — the videos
    they watched and the graph they explored are of *that* policy — so it is the
    reference for "did this choice help at all". It is deliberately not an arm:
    see the module docstring.
    """

    task: str
    label: str
    baseline: str
    arms: tuple[CollectionArm, ...]

    @property
    def datasets(self) -> tuple[str, ...]:
        """Every rollout dataset this task needs a success rate for."""
        return tuple(arm.dataset for arm in self.arms)


#: The retrained policies, one arm set per task. The datasets are the canonical
#: DROID rollout collections registered in ``configs/data/droid.yaml``; the
#: option strings are the ones ``QUESTION_SETS`` asks (plus dropped wordings).
#:
#: ``set_table`` appears twice because two policies have been studied under it:
#: the flow-based ``pretrained_dit`` policy the three-stage and contrastive
#: layouts show, and the earlier ``v2`` diffusion policy the order-effect layout
#: showed, whose options were recorded as ids rather than as prose. Each is
#: scored against the arms retrained from *its own* baseline — mixing them would
#: compare a choice against a policy the participant never saw.
TASK_ARMS: tuple[TaskArms, ...] = (
    TaskArms(
        task="set_table",
        label="Set the table",
        baseline="set_table_baseline_pretrained_dit",
        arms=(
            CollectionArm(
                label="Milk left",
                dataset="set_table_plusleft_pretrained_dit",
                options=(
                    "Chocolate milk left, cola middle, cupcake right",
                    # The six-option wording, in both of its arrangements of
                    # the cupcake and the cola.
                    "Milk left, cola middle, cupcake right",
                    "Milk left, cupcake middle, cola right",
                ),
            ),
            CollectionArm(
                label="Milk middle",
                dataset="set_table_plusmiddle_pretrained_dit",
                options=(
                    "Cola left, chocolate milk middle, cupcake right",
                    "Cola left, milk middle, cupcake right",
                    "Cupcake left, milk middle, cola right",
                ),
            ),
            CollectionArm(
                label="Milk right",
                dataset="set_table_plusright_pretrained_dit",
                options=(
                    "Cupcake left, cola middle, chocolate milk right",
                    "Cola left, cupcake middle, milk right",
                    "Cupcake left, cola middle, milk right",
                ),
            ),
        ),
    ),
    TaskArms(
        task="prepare_tray",
        label="Prepare the tray",
        baseline="prepare_tray_baseline",
        arms=(
            CollectionArm(
                label="Bowl on top",
                dataset="prepare_tray_plustop",
                options=("Bowl and fork on top of the shelf",),
            ),
            CollectionArm(
                label="Bowl in middle",
                dataset="prepare_tray_plusbottom",
                options=("Bowl in the middle of the shelf, fork on top of the shelf",),
            ),
        ),
    ),
    TaskArms(
        task="cooking",
        label="Cook the meal",
        baseline="cooking_baseline_dit_200",
        arms=(
            CollectionArm(
                label="Carrot",
                dataset="cooking_baseline_carrot_dit",
                options=("Pick up the carrot and place it in the pot.",),
            ),
            CollectionArm(
                label="Corn",
                dataset="cooking_baseline_corn_dit",
                options=("Pick up the corn and place it in the pot.",),
            ),
            CollectionArm(
                label="Lid",
                dataset="cooking_baseline_lid_dit",
                options=("Pick up the lid and place it on the pot.",),
            ),
        ),
    ),
    TaskArms(
        # The order-effect layout's ranking of the same three milk positions,
        # over the v2 diffusion policy it showed. It stores option *ids*, and
        # withholds the position from the participant on purpose (they rank
        # "Option 1/2/3" beside three clips), so the ids are all there is to
        # match on.
        task="set_table_v2",
        label="Set the table (v2 policy)",
        baseline="set_table_baseline_v2_sync",
        arms=(
            CollectionArm(
                label="Milk left",
                dataset="set_table_baseline_v2_plusleft_sync",
                options=("milk_left",),
            ),
            CollectionArm(
                label="Milk middle",
                dataset="set_table_baseline_v2_plusmiddle_sync",
                options=("milk_middle",),
            ),
            CollectionArm(
                label="Milk right",
                dataset="set_table_baseline_v2_plusright_sync",
                options=("milk_right",),
            ),
        ),
    ),
)

#: The order the tasks are drawn in, which is the order ``three_stage_v1``
#: lists its stages — not the order a participant met them, which is drawn per
#: participant, and not alphabetical.
TASK_ORDER: tuple[str, ...] = tuple(task.task for task in TASK_ARMS)

#: Every rollout dataset a full scoring needs a success rate for: the arms, and
#: the baselines that are reported beside them.
REFERENCE_DATASETS: tuple[str, ...] = tuple(
    dict.fromkeys(dataset for task in TASK_ARMS for dataset in (*task.datasets, task.baseline))
)

_ARM_BY_OPTION: dict[str, tuple[TaskArms, CollectionArm]] = {
    option: (task, arm) for task in TASK_ARMS for arm in task.arms for option in arm.options
}


def arm_for_option(option: str) -> tuple[TaskArms, CollectionArm] | None:
    """``(task, arm)`` for a stored answer, or ``None`` if it names no arm.

    The answer identifies the task as well as the option: every option string
    is unique across the tasks, which is what lets a layout that keys its
    answers by modality rather than by task (the contrastive one) be scored at
    all — there is no task recorded on those rows to group them by.
    """
    return _ARM_BY_OPTION.get(option)


def task_by_key(task: str) -> TaskArms | None:
    return next((entry for entry in TASK_ARMS if entry.task == task), None)


@dataclass(frozen=True)
class SuccessRate:
    """A retrained policy's measured outcome: *successes* out of *trials*.

    Only scored rollouts count as trials — an episode nobody labelled is not a
    failure, and folding it in would depress exactly the arms whose labelling is
    incomplete. See
    :class:`behavior_graphs.viz.streamlit.rollout_outcomes.DatasetOutcomes`, which
    is where this comes from.
    """

    dataset: str
    successes: int
    trials: int

    def __post_init__(self) -> None:
        if self.trials <= 0:
            raise ValueError(f"{self.dataset}: a success rate needs a scored rollout")  # noqa: TRY003
        if not 0 <= self.successes <= self.trials:
            raise ValueError(f"{self.dataset}: {self.successes} of {self.trials}")  # noqa: TRY003

    @property
    def rate(self) -> float:
        return self.successes / self.trials


def measured_success_rates(
    datasets: Sequence[str] = REFERENCE_DATASETS,
) -> dict[str, SuccessRate]:
    """``{dataset: rate}`` read off the rollout collections on disk.

    Read rather than recorded: the outcomes of a real-robot rollout set are
    annotated by hand and get corrected, and a number frozen into this file
    would keep scoring choices against a label that has since been fixed. A
    dataset that is not mounted, or that has no scored episode, is simply
    absent — :func:`unscored_datasets` is what turns that into something the
    page can say.

    Imported lazily because the readers pull in Streamlit and the filesystem,
    and everything else in this module is pure.
    """
    from pathlib import Path

    from behavior_graphs.core.io import resolve_data_path
    from behavior_graphs.viz.streamlit.rollout_outcomes import (
        ROLLOUTS_SUBDIR,
        discover_datasets,
        is_rollout_dataset,
        load_outcomes,
    )

    data_path = resolve_data_path()
    dirs: dict[str, Path] = {}
    if data_path:
        root = Path(data_path) / ROLLOUTS_SUBDIR
        dirs = {name: root / name for name in datasets if is_rollout_dataset(root / name)}
    if not dirs:
        # A deployment whose rollouts are not under ``$DATA_PATH/rollouts``, or
        # one with no data root set at all. Only when the standard location came
        # up completely empty, as on the policy-comparison page: with no
        # ``$DATA_PATH`` this route walks every indexed run to find out where the
        # rollouts live, which is tens of seconds this panel must not spend
        # because one arm of one task has not been collected yet.
        #
        # Its ``BEHAVIOR_GRAPHS_DISPLAY_DATA`` allowlist applies here, which is the
        # other reason it is the fallback rather than the route: a retrained
        # policy is reference data for this panel whether or not the deployment
        # offers it as a dataset to browse.
        found = discover_datasets()
        dirs = {name: found[name] for name in datasets if name in found}

    rates: dict[str, SuccessRate] = {}
    for name, dataset_dir in dirs.items():
        outcomes = load_outcomes(name, dataset_dir)
        if outcomes.scored:
            rates[name] = SuccessRate(
                dataset=name, successes=outcomes.successes, trials=outcomes.scored
            )
    return rates


def unscored_datasets(rates: Mapping[str, SuccessRate]) -> dict[str, tuple[str, ...]]:
    """``{task: datasets with no measured rate}`` — the tasks that cannot be scored.

    Baselines are not in it: a missing baseline costs the "did it help at all"
    column and nothing else, whereas a missing arm makes the best option, and
    therefore every regret for the task, wrong.
    """
    missing = {
        task.task: tuple(name for name in task.datasets if name not in rates) for task in TASK_ARMS
    }
    return {task: names for task, names in missing.items() if names}


def scoreable(rates: Mapping[str, SuccessRate]) -> tuple[TaskArms, ...]:
    """The tasks every one of whose arms has a measured success rate."""
    return tuple(task for task in TASK_ARMS if all(name in rates for name in task.datasets))


def best_arm(
    task: TaskArms, rates: Mapping[str, SuccessRate]
) -> tuple[CollectionArm, SuccessRate] | None:
    """The task's best-performing option, or ``None`` if one is unmeasured.

    Ties go to the arm the question bank lists first, which is the order the
    options were shown in — arbitrary, but fixed, so an exported figure does not
    change its "best option" between two runs over the same data.
    """
    if any(name not in rates for name in task.datasets):
        return None
    scored = [(arm, rates[arm.dataset]) for arm in task.arms]
    return max(scored, key=lambda pair: pair[1].rate)


def arm_span(task: TaskArms, rates: Mapping[str, SuccessRate]) -> float | None:
    """Best minus worst option, in success-rate points, or ``None`` unscored.

    The denominator of the min-max normalization below. ``None`` rather than
    zero when every arm measured the same: there is no scale to normalize onto,
    and dividing by it would turn a task nobody could have got wrong into one
    where everybody did.
    """
    if any(arm.dataset not in rates for arm in task.arms):
        return None
    values = [rates[arm.dataset].rate for arm in task.arms]
    span = 100 * (max(values) - min(values))
    return span or None


def gain_over_baseline(task: TaskArms, rates: Mapping[str, SuccessRate]) -> float | None:
    """What choosing well was *worth*: best option minus the baseline, in points.

    The denominator of the other normalization, and the more direct reading of
    the decision — the question was which data to collect, and this is how much
    success rate the right answer bought over collecting none.

    ``None`` when it is not positive, which is the case that has to be guarded
    rather than divided by: if no option beat collecting nothing then there was
    no gain to forgo a share of, and a negative denominator would silently
    report the worst choice as the best one. It does not happen on the measured
    rates — every task's best option beats its baseline — but the rates are read
    off disk precisely because they get re-annotated, so it is not a property
    this function may assume.
    """
    best = best_arm(task, rates)
    baseline = rates.get(task.baseline)
    if best is None or baseline is None:
        return None
    gain = 100 * (best[1].rate - baseline.rate)
    return gain if gain > 0 else None


#: How separated the best option has to be from the baseline before the
#: share-of-gain score is reported at all. The same α the comparison page tests
#: at, and the same exact test — a gain has to be a gain before a ratio can be
#: taken against it.
GAIN_ALPHA = 0.05


def gain_p_value(task: TaskArms, rates: Mapping[str, SuccessRate]) -> float | None:
    """One-sided p for "the best option beats the baseline", or ``None``.

    Barnard's exact test, as on the Policy Comparison page: neither margin of
    the table is fixed by the experiment — we chose the rollout counts, not the
    number of successes — so the unconditional test is the correct one. Memoised
    per pair of counts in the stats layer, so calling this once per scored
    answer costs one test per task.
    """
    from behavior_graphs.stats.policy_comparison import PolicyOutcomes, pairwise_pvalues

    best = best_arm(task, rates)
    baseline = rates.get(task.baseline)
    if best is None or baseline is None:
        return None
    arms = (
        PolicyOutcomes("best", best[1].successes, best[1].trials),
        PolicyOutcomes("baseline", baseline.successes, baseline.trials),
    )
    return float(pairwise_pvalues(arms)[0][1])


def measurable_gain(
    task: TaskArms, rates: Mapping[str, SuccessRate], alpha: float = GAIN_ALPHA
) -> float | None:
    """:func:`gain_over_baseline`, but only when the gain is separated from zero.

    ``None`` otherwise, which suppresses the share-of-gain score for that task
    rather than dividing by a number the data does not support. This is not
    fastidiousness: cooking's best option is 16 successes in 60 against the
    baseline's 14, a gain of 3.3 points at p≈0.38 — and dividing a 5-to-8 point
    regret by it produced a mean "165% of the gain forgone", which reads as a
    strong finding and is an artifact of a near-zero denominator. A blank is the
    honest output, and the bounded normalization still scores the task.
    """
    gain = gain_over_baseline(task, rates)
    if gain is None:
        return None
    p = gain_p_value(task, rates)
    return None if p is not None and p > alpha else gain


def chance_normalized(task: TaskArms, rates: Mapping[str, SuccessRate]) -> float | None:
    """Mean min-max regret of picking one of the options uniformly at random.

    The reference the normalized score is unreadable without, and it is *not*
    0.5: it depends on how the arms are spread, and only lands on a half for a
    two-option task or three evenly spaced ones. Reported per task and pooled
    with the same weights as the scores it sits beside.
    """
    span = arm_span(task, rates)
    if span is None:
        return None
    values = [100 * rates[arm.dataset].rate for arm in task.arms]
    best = max(values)
    return sum((best - value) / span for value in values) / len(values)


def arm_rows(rates: Mapping[str, SuccessRate]) -> list[dict]:
    """The reference table: every option, its policy, and how it did.

    One row per arm plus one per baseline, so the panel can show what the
    regrets are computed against rather than asking the reader to take them on
    faith. Rows for unmeasured datasets carry ``None`` rates instead of being
    dropped — a missing arm is the reason a task is not scored, and it has to be
    visible for that to be readable.
    """
    rows = []
    for task in TASK_ARMS:
        best = best_arm(task, rates)
        for arm in (None, *task.arms):
            dataset = task.baseline if arm is None else arm.dataset
            rate = rates.get(dataset)
            rows.append(
                {
                    "task": task.task,
                    "task_label": task.label,
                    "kind": "baseline" if arm is None else "arm",
                    "label": "Collect nothing (baseline)" if arm is None else arm.label,
                    "option": "" if arm is None else arm.options[0],
                    "dataset": dataset,
                    "successes": None if rate is None else rate.successes,
                    "trials": None if rate is None else rate.trials,
                    "success_rate": None if rate is None else rate.rate,
                    # Only ever true of an arm: the baseline is not an option, so
                    # it cannot be the best one.
                    "is_best": arm is not None and best is not None and arm is best[0],
                }
            )
    return rows


def regret_rows(responses: Iterable[dict], rates: Mapping[str, SuccessRate]) -> list[dict]:
    """One row per scored data-collection answer, with its regret.

    ``regret`` is in success-rate points (0–100), which is the unit the arms are
    reported in and the one a reader of "12 points of success rate left on the
    table" can hold. ``chose_best`` is the same answer as ``regret == 0`` said as
    a rate, since a chart of "how often was the best option picked" reads better
    at these sample sizes than a mean of a mostly-zero quantity.
    """
    rows = []
    for row in analytics.choice_rows(responses):
        found = arm_for_option(row["choice"])
        if found is None:
            continue
        task, arm = found
        best = best_arm(task, rates)
        if best is None:
            continue
        best_option, best_rate = best
        chosen = rates[arm.dataset]
        baseline = rates.get(task.baseline)
        span, gain = arm_span(task, rates), measurable_gain(task, rates)
        rows.append(
            {
                **{key: row[key] for key in ("participant", "version", "group_label", "status")},
                # The task the option belongs to, not the one the response
                # recorded: only the three-stage layout records one, and this is
                # the same answer for the layouts that do.
                "task": task.task,
                "task_label": task.label,
                "phase": row["phase"],
                # Whether the display was drawn for them or picked by them, so a
                # by-display comparison can be held to the randomized stages.
                "assignment": row.get("assignment", ""),
                "choice": row["choice"],
                "choice_label": arm.label,
                "dataset": arm.dataset,
                "success_rate": 100 * chosen.rate,
                "best_label": best_option.label,
                "best_dataset": best_option.dataset,
                "best_rate": 100 * best_rate.rate,
                "regret": 100 * (best_rate.rate - chosen.rate),
                "chose_best": math.isclose(chosen.rate, best_rate.rate),
                # What the choice was worth against reading the same policy and
                # collecting nothing. Negative for an option that made the
                # policy worse, which two of the three cooking arms did.
                "baseline_rate": None if baseline is None else 100 * baseline.rate,
                "gain_over_baseline": (
                    None if baseline is None else 100 * (chosen.rate - baseline.rate)
                ),
                # Two normalizations, because they disagree in both directions
                # and each is wrong on its own:
                #
                # `regret_normalized` is the distance to the best option as a
                # share of the spread between best and worst. Bounded 0-1 and so
                # poolable across tasks, which raw points are not — set table's
                # options span 33 points and cooking's 8, so a pooled mean of
                # points is mostly set table. Read against `chance_normalized`.
                #
                # `regret_of_gain` is the same distance as a share of what
                # choosing well was worth: the best option minus collecting
                # nothing. Above 1 means the pick left the policy worse than not
                # collecting at all — which happens, and which the bounded score
                # cannot say. It also stops a task whose options are all good
                # from reading as badly as one whose options are all bad:
                # prepare tray's second option is the worst of two (1.0
                # normalized) and still gets 59% of the available gain.
                #
                # Blank where the gain is not separated from zero: see
                # `measurable_gain`, and cooking, whose 3.3-point "gain" turned
                # a 5-point regret into 165% of it.
                "regret_normalized": (
                    None if span is None else 100 * (best_rate.rate - chosen.rate) / span
                ),
                "regret_of_gain": (
                    None if gain is None else 100 * (best_rate.rate - chosen.rate) / gain
                ),
                "chance_normalized": chance_normalized(task, rates),
                # What choosing well was worth, and whether that is a gain at
                # all: reported on the row so the page can say why a task has
                # no share-of-gain score rather than leaving a gap.
                "gain": gain_over_baseline(task, rates),
                "gain_p": gain_p_value(task, rates),
                "worse_than_baseline": (None if baseline is None else chosen.rate < baseline.rate),
            }
        )
    return rows


#: The two displays, graph first: it is the one the study is about, and the one
#: a reader should meet on the left of a panel. Defined here rather than in
#: :mod:`behavior_graphs.viz.streamlit.user_study.figure_export`, which takes its
#: order from this — the pooled rows below and the figure drawn from them have
#: to agree, and two tuples spelling the same order is how they stop agreeing.
PHASE_ORDER = ("graph", "videos")


def randomized_rows(rows: Iterable[dict]) -> list[dict]:
    """The answers from stages whose display was **assigned**, not chosen.

    The restriction every display comparison on these responses is made under,
    in one place because it was spelled out in four: the three-stage layout's
    last stage let the participant pick their own display, so an answer from it
    is an outcome of the participant rather than of the display. On the live
    store that stage is 18 of the 54 scored answers and 16 of 18 participants
    took the graph for it, which is enough to reverse which display led to the
    better decisions.

    **A row with no ``assignment`` recorded counts as randomized.** The layouts
    that never assigned a display had no free choice to contaminate the
    comparison with, so dropping their answers would lose evidence rather than
    protect anything.
    """
    return [row for row in rows if row.get("assignment", "") != "chosen"]


def choice_share_rows(
    responses: Iterable[dict],
    *,
    randomized_only: bool = True,
) -> list[dict]:
    """How often each option was picked off each display, as a share of that display.

    One row per (arm set, display, option), and **every option of the arm set
    gets a row** — including the ones nobody picked off that display. That is
    what the function is for: the interesting thing a display can do is leave
    the best option unchosen, and rows only for the options that were picked
    cannot say it. An option nobody picked is ``picks=0``, which is a different
    statement from an option that was never offered (no row at all), and a
    figure drawing these has to keep the two apart.

    The denominator is the answers *that display drew for that arm set*, not the
    participants and not the answers overall: a share of picks is only a share
    among the options that were on the same screen. An (arm set, display) pair
    with no answers contributes no rows, so a share is never taken over zero.

    ``randomized_only`` drops the stages whose display the participant *chose*
    (:func:`randomized_rows`), and is the default here rather than in
    :func:`regret_rows` because a *share* pooled over that stage says "the
    people who reached for the graph also picked this data" — a sentence about
    people, not about displays.

    The rows carry the arm set's ``baseline`` because ``task`` does not identify
    it — two policies have been studied under ``set_table`` — and because the
    baseline is the one policy in the comparison that **was not an option**. A
    figure drawing a bar per policy needs to know not to draw a zero there.
    """
    answers = analytics.choice_rows(responses)
    picked: dict[TaskArms, dict[str, Counter]] = {}
    for row in randomized_rows(answers) if randomized_only else answers:
        # Only answers that name a display, as :func:`by_display` does: a layout
        # that showed one condition and never recorded which (the production
        # one) has nothing to say about graph against videos, and pooling it
        # into either column puts answers in it that were not made off it.
        phase = str(row.get("phase") or "")
        if not phase:
            continue
        found = arm_for_option(row["choice"])
        if found is None:
            continue
        task, arm = found
        picked.setdefault(task, {}).setdefault(phase, Counter())[arm.dataset] += 1

    rows = []
    for task in TASK_ARMS:
        by_phase = picked.get(task)
        if not by_phase:
            continue
        phases = [phase for phase in PHASE_ORDER if phase in by_phase]
        phases += [phase for phase in by_phase if phase not in PHASE_ORDER]
        for phase in phases:
            counts = by_phase[phase]
            answers = sum(counts.values())
            for arm in task.arms:
                rows.append(
                    {
                        "task": task.task,
                        "task_label": task.label,
                        "baseline": task.baseline,
                        "phase": phase,
                        "dataset": arm.dataset,
                        "choice_label": arm.label,
                        "picks": counts.get(arm.dataset, 0),
                        "answers": answers,
                        # A fraction, where the regret columns are in points:
                        # this one is drawn on an axis, and the figure that
                        # draws it is the one place it becomes a percentage.
                        "share": counts.get(arm.dataset, 0) / answers,
                    }
                )
    return rows


#: The quantity the panels and the paper report per decision: regret, the
#: success rate the chosen option left on the table against the best one
#: available. **Lower is better**, which every panel that draws it has to say.
#:
#: Gain over the baseline was tried as the headline and dropped. It reads more
#: easily on its own -- higher is better, and it can say an option made the
#: policy worse, which regret can only score as far from best -- but it is a
#: worse measure of the *decision*, because it carries the task's headroom:
#: prepare tray had +28.3 points available and cooking +3.3, so a mean of gains
#: is mostly a statement about which tasks a display was shown with. Regret
#: asks only how well they chose among what they were offered.
#:
#: Gain is still computed on every row, and is the right quantity for "what was
#: this choice worth"; it is simply not the decision measure.
DECISION_METRIC = "regret"


def by_display(rows: Iterable[dict], metric: str = DECISION_METRIC) -> dict[str, list[float]]:
    """``{display: [metric]}`` pooled over the tasks.

    Pooling is over answers, not participants: the three-stage layout has one
    participant answering up to three tasks, and each answer is one decision
    made off one display, which is the unit the comparison is about. Only rows
    that name a display — a layout that showed one condition and never recorded
    which (the production one) has nothing to contribute to a graph-versus-
    videos comparison.
    """
    pooled: dict[str, list[float]] = {phase: [] for phase in PHASE_ORDER}
    for row in rows:
        if row["phase"] in pooled and row.get(metric) is not None:
            pooled[row["phase"]].append(float(row[metric]))
    return pooled


def regret_by_display(rows: Iterable[dict]) -> dict[str, list[float]]:
    """``{display: [regret]}`` — :func:`by_display` on the regret column."""
    return by_display(rows, "regret")


def by_task(
    rows: Iterable[dict], metric: str = DECISION_METRIC
) -> dict[str, dict[str, list[float]]]:
    """``{task: {display: [metric]}}``, in :data:`TASK_ORDER`.

    Only tasks that were answered: an empty panel per task the selected version
    never showed would read as missing data rather than as a task this study did
    not run.
    """
    rows = list(rows)
    answered = [task for task in TASK_ORDER if any(row["task"] == task for row in rows)]
    grouped: dict[str, dict[str, list[float]]] = {
        task: {phase: [] for phase in PHASE_ORDER} for task in answered
    }
    for row in rows:
        per_phase = grouped.get(row["task"])
        if per_phase is not None and row["phase"] in per_phase and row.get(metric) is not None:
            per_phase[row["phase"]].append(float(row[metric]))
    return grouped


def regret_by_task(rows: Iterable[dict]) -> dict[str, dict[str, list[float]]]:
    """``{task: {display: [regret]}}`` — :func:`by_task` on the regret column."""
    return by_task(rows, "regret")


def summary_rows(rows: Iterable[dict]) -> list[dict]:
    """Per task and display: how many answers, mean regret, how often the best.

    Plus a pooled ``All tasks`` row per display. The pooled *normalized* means
    are the headline; the pooled mean of raw points is not, and is reported only
    because it is the unit the arms are given in. The tasks' options span very
    different ranges — 33 success-rate points for one arm set against 8 for
    another — so a pooled mean of points is very largely a mean over the widest
    task, whichever display happened to draw it.
    """
    rows = list(rows)
    out = []
    groups: list[tuple[str, str, list[dict]]] = []
    for task in TASK_ORDER:
        in_task = [row for row in rows if row["task"] == task]
        if in_task:
            groups.append((task, in_task[0]["task_label"], in_task))
    groups.append(("", "All tasks", rows))
    for task, label, in_group in groups:
        for phase in PHASE_ORDER:
            answers = [row for row in in_group if row["phase"] == phase]
            if not answers:
                continue
            regrets = [float(row["regret"]) for row in answers]
            out.append(
                {
                    "task": task,
                    "task_label": label,
                    "phase": phase,
                    "answers": len(answers),
                    "mean_gain": _mean_of(answers, "gain_over_baseline"),
                    # `None` where no answer in the group has a baseline to
                    # measure against, the same convention `_mean_of` uses: a
                    # median of nothing is not zero.
                    "median_gain": _median_of(answers, "gain_over_baseline"),
                    "mean_regret": sum(regrets) / len(regrets),
                    "median_regret": _median(regrets),
                    "mean_normalized": _mean_of(answers, "regret_normalized"),
                    "mean_of_gain": _mean_of(answers, "regret_of_gain"),
                    # Averaged over the same answers, so the reference matches
                    # the mix of tasks the score beside it was taken over.
                    "chance_normalized": _mean_of(answers, "chance_normalized"),
                    "worse_than_baseline_rate": _rate_of(answers, "worse_than_baseline"),
                    "chose_best": sum(1 for row in answers if row["chose_best"]),
                    "chose_best_rate": (
                        100 * sum(1 for row in answers if row["chose_best"]) / len(answers)
                    ),
                }
            )
    return out


def has_score(value: object) -> bool:
    """Is this an answer at all — not missing, and not a NaN standing in for one.

    Both, because a caller may hand these rows over as a dataframe's records and
    pandas turns a ``None`` in a float column into ``NaN`` on the way. An
    ``is None`` test alone lets that through, and a single NaN takes a mean with
    it: the page's own regret table read ``NaN`` for exactly this reason.
    """
    return value is not None and (not isinstance(value, float) or value == value)


def _mean_of(rows: Sequence[Mapping], key: str) -> float | None:
    """Mean of *key* over the rows that have one.

    ``None`` where none does, rather than zero: a task whose options all measure
    the same has no normalized regret at all, and zero would read as everybody
    having chosen perfectly.
    """
    values = [float(row[key]) for row in rows if has_score(row.get(key))]
    return sum(values) / len(values) if values else None


def _rate_of(rows: Sequence[Mapping], key: str) -> float | None:
    """Percentage of the rows where *key* is true, over those that know."""
    known = [row for row in rows if has_score(row.get(key))]
    return 100 * sum(1 for row in known if row[key]) / len(known) if known else None


def _median_of(rows: Sequence[Mapping], key: str) -> float | None:
    """:func:`_median` over the rows that have *key*, or ``None`` if none do."""
    values = [float(row[key]) for row in rows if has_score(row.get(key))]
    return _median(values) if values else None


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
