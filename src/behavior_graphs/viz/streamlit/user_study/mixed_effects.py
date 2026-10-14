"""Continuous mixed-effects summaries for the randomized user-study stages.

This is intentionally separate from the descriptive analytics helpers.  The
resulting estimates condition on task, study stage, experience, and repeated
responses from a participant; raw boxplots do not.
"""

from __future__ import annotations

from collections.abc import Iterable

_EXPERIENCE = {
    "None at all": 0,
    "A little": 1,
    "Some": 2,
    "A good deal": 3,
    "A great deal": 4,
}
_OUTCOMES = ("temporal_demand", "effort", "confidence")
#: Fixed so two renders of the same responses produce the same intervals.
_SEED = 20260913
_MISSING_EXTRA = "Install the `analysis` extra to fit adjusted study effects."
#: Above this the four chains have not agreed on the same posterior, so the
#: summary is not a summary of anything. 1.01 is the threshold Vehtari et al.
#: (2021) argue for over the older 1.1, which passes chains that clearly have
#: not mixed. A sampler that fails outright already raises; this is the one
#: that finishes and hands back numbers looking exactly as trustworthy as
#: good ones, which is worse.
_MAX_R_HAT = 1.01


class ModelFitError(RuntimeError):
    """A model these responses cannot support, or a sampler that gave up.

    Bambi raises out of its own internals — an ``IndexError`` where a design
    matrix comes out degenerate, a ``ValueError`` where a term is constant —
    and the pages that fit these models render the rest of the study beneath
    them. Every such failure arrives here instead, as one type carrying a
    message worth showing, so a store that cannot support the model costs its
    caller a warning rather than the page. Subclasses ``RuntimeError`` because
    the missing-dependency guard below already raises that and callers already
    catch it.
    """


def _fit(formula: str, data, *, subject: str, draws: int, tune: int):
    """Fit one Gaussian mixed model, or raise :class:`ModelFitError`."""
    try:
        import bambi as bmb
    except ModuleNotFoundError as error:  # pragma: no cover - deployment guard
        raise RuntimeError(_MISSING_EXTRA) from error

    try:
        model = bmb.Model(formula, data, family="gaussian")
        posterior = model.fit(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.95,
            random_seed=_SEED,
        )
    except Exception as error:  # noqa: BLE001 - bambi/pymc raise from their internals
        raise ModelFitError(
            f"Could not fit the adjusted model for {subject} on these "
            f"{len(data)} scored responses ({type(error).__name__}: {error}). "
            "It needs several participants who saw both displays across the "
            "randomized stages, and more than one value of every term."
        ) from error
    return model, posterior


def _posterior_summary(draws_array) -> dict[str, float]:
    """Posterior mean and two-sided 95% HDI of a flattened draws array."""
    import arviz as az

    values = draws_array.reshape(-1)
    try:
        lower, upper = az.hdi(values, prob=0.95)
    except TypeError:  # ArviZ before the 1.0 argument rename.
        lower, upper = az.hdi(values, hdi_prob=0.95)
    return {"mean": float(values.mean()), "lower": float(lower), "upper": float(upper)}


def _marginal_means(model, posterior, data) -> dict[str, dict[str, float]]:
    """``{display: {mean, lower, upper}}`` — the adjusted mean per display.

    Each posterior draw is averaged over the *observed* distribution of task,
    stage and experience, with the participant intercepts left out: a
    population-level adjusted marginal mean, not a raw condition average and
    not a prediction for one participant. That is what makes the two displays
    comparable when the tasks they were read in differ, which they do -- every
    participant rated one task off the graph and a different one off the videos.

    Shared by the rating models and the decision-regret one so the figure's
    panels cannot come to mean two different things by "adjusted mean".
    """
    means = {}
    for modality, graph in (("videos", 0), ("graph", 1)):
        scenario = data.copy()
        scenario["graph"] = graph
        predicted = model.predict(
            posterior,
            kind="response_params",
            data=scenario,
            include_group_specific=False,
            inplace=False,
        )
        means[modality] = _posterior_summary(predicted.posterior["mu"].values.mean(axis=-1))
    return means


def _graph_effect(posterior) -> dict[str, float]:
    """The ``graph`` coefficient's posterior mean and two-sided 95% HDI.

    Written against both ArviZ APIs on purpose. ``arviz>=0.17`` is what the
    `analysis` extra asks for, and the two sides of the 1.0 rename disagree
    on the keyword *and* on what the columns come back called — which is a
    difference between a developer's environment and the deployed image, so a
    version-specific call here is a page that works everywhere it is tested
    and raises a TypeError everywhere it is used.
    """
    import arviz as az

    try:
        summary = az.summary(posterior, var_names=["graph"], ci_prob=0.95, ci_kind="hdi")
    except TypeError:  # ArviZ before the 1.0 argument rename.
        summary = az.summary(posterior, var_names=["graph"], hdi_prob=0.95)
    row = summary.loc["graph"]
    # `hdi_2.5%`/`hdi_97.5%` before 1.0 and `hdi_95%_lb`/`..._ub` after. The
    # one thing both spellings agree on is which of the two is the lower bound.
    columns = [column for column in summary.columns if "hdi" in str(column).lower()]
    if len(columns) != 2:  # pragma: no cover - defensive API guard
        raise RuntimeError(
            f"ArviZ did not return a two-sided 95% HDI summary: {list(summary.columns)!r}"
        )
    lower, upper = sorted(float(row[column]) for column in columns)
    effect = {"mean": float(row["mean"]), "lower": lower, "upper": upper}

    # Carried, not discarded: `mcse_mean` is the Monte-Carlo error on the
    # estimate, and it is what says how many digits of it are the data rather
    # than the sampler. On this study it runs about 0.1 rating points, so the
    # second decimal of an effect is noise -- reproducible noise, because the
    # fit is seeded, which is how an over-precise interval gets into a paper.
    # Read where this ArviZ reports them: they are not part of the summary API
    # that moved at 1.0, but neither was `hdi_prob`.
    for name in ("r_hat", "ess_bulk", "mcse_mean"):
        if name in summary.columns:
            effect[name] = float(row[name])

    r_hat = effect.get("r_hat")
    if r_hat is not None and r_hat > _MAX_R_HAT:
        raise ModelFitError(
            f"The chains did not converge on one posterior (r_hat {r_hat:.3f} > "
            f"{_MAX_R_HAT}), so this estimate is not one. More tuning draws, or "
            "a model these responses can support -- see the terms in the formula."
        )
    return effect


def _participant(response: dict) -> str:
    """A grouping key for ``(1|participant)``, or "" for a response without one.

    Deliberately stricter than :func:`analytics.participant`, which is a label
    for a table and ends at "anon" so that two unidentifiable responses read as
    one row rather than as a lost one. A model cannot afford that: "anon" as a
    group would pool strangers under one random intercept and shrink them
    toward each other. A response this cannot name is left out of the fit
    instead, and ``_response_id`` -- which every stored record has -- is tried
    before giving up.
    """
    return str(
        response.get("participant_id")
        or response.get("session_id")
        or response.get("_response_id")
        or ""
    )


def assigned_rating_rows(responses: Iterable[dict]) -> list[dict]:
    """Return ratings from randomized, completed three-stage study records."""
    rows: list[dict] = []
    for response in responses:
        if (
            response.get("study_variant") != "three_stage_v1"
            or response.get("status", "submitted") != "submitted"
            or response.get("origin") == "lab"
        ):
            continue
        participant = _participant(response)
        tasks = response.get("task_order")
        modalities = response.get("stage_modalities")
        assignments = response.get("modality_assignment")
        ratings = response.get("nasa_tlx")
        experience = _EXPERIENCE.get(str(response.get("robot_data_collection_experience") or ""))
        if not participant or experience is None or not isinstance(tasks, list):
            continue
        if not all(isinstance(value, dict) for value in (modalities, assignments, ratings)):
            continue
        for stage, task in enumerate(tasks, start=1):
            if assignments.get(task) != "assigned" or modalities.get(task) not in {
                "graph",
                "videos",
            }:
                continue
            task_ratings = ratings.get(task)
            if not isinstance(task_ratings, dict):
                continue
            for outcome in _OUTCOMES:
                score = task_ratings.get(outcome)
                if score in {0, 20, 40, 60, 80, 100}:
                    rows.append(
                        {
                            "participant": participant,
                            "task": str(task),
                            "stage": stage,
                            "experience_level": experience,
                            "graph": int(modalities[task] == "graph"),
                            "outcome": outcome,
                            "score": float(score),
                        }
                    )
    return rows


def continuous_effects(
    responses: Iterable[dict], *, draws: int = 1_000, tune: int = 1_000
) -> list[dict]:
    """Fit the paper's Gaussian mixed model and return adjusted summaries.

    Imports are local so collecting and browsing study responses does not
    require the optional Bayesian-analysis dependencies.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - deployment guard
        raise RuntimeError(_MISSING_EXTRA) from error

    data = pd.DataFrame(assigned_rating_rows(responses))
    if data.empty:
        return []

    results = []
    for outcome in _OUTCOMES:
        subset = data[data.outcome == outcome].copy()
        if subset.empty:
            continue
        subset["task"] = pd.Categorical(subset.task)
        subset["stage"] = pd.Categorical(subset.stage)
        model, posterior = _fit(
            "score ~ graph + experience_level + task + stage + (1|participant)",
            subset,
            subject=outcome.replace("_", " "),
            draws=draws,
            tune=tune,
        )
        results.append(
            {
                "outcome": outcome,
                **_graph_effect(posterior),
                "marginal_means": _marginal_means(model, posterior, subset),
                # Per display, so a figure of the marginal means can say what
                # each estimate was fit on the way the raw boxes say what they
                # are drawn from.
                "n": {
                    "graph": int((subset.graph == 1).sum()),
                    "videos": int((subset.graph == 0).sum()),
                },
            }
        )
    return results


def decision_regret_effect(
    responses: Iterable[dict], rates: dict, *, draws: int = 1_000, tune: int = 1_000
) -> dict | None:
    """Estimate the graph--video difference in data-collection regret.

    Regret is :data:`regret.DECISION_METRIC` from
    :mod:`behavior_graphs.viz.streamlit.user_study.regret`: the success rate of
    the best option available minus that of the one the participant picked, in
    success-rate points, so **lower is better** and a NEGATIVE ``graph``
    coefficient means the display led to better decisions.

    Fit on regret rather than on gain over the baseline, which this model
    reported for a while. Gain reads more easily on its own, but it carries the
    task's headroom -- +28.3 success-rate points were available on prepare tray
    against +3.3 on cooking -- so its per-display mean is partly a statement
    about which tasks a display was shown with. Regret asks only how well they
    chose among the options they were given, which is the question the display
    could have helped with.

    The same adjusted model as :func:`continuous_effects`: only the two
    randomized stages enter it, and task, stage, and experience are held fixed
    so the ``graph`` coefficient is the display's effect on the decision
    itself.
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as error:  # pragma: no cover - deployment guard
        raise RuntimeError(_MISSING_EXTRA) from error
    from behavior_graphs.viz.streamlit.user_study import regret

    rows = []
    for response in responses:
        if (
            response.get("study_variant") != "three_stage_v1"
            or response.get("status", "submitted") != "submitted"
            or response.get("origin") == "lab"
        ):
            continue
        participant = _participant(response)
        experience = _EXPERIENCE.get(str(response.get("robot_data_collection_experience") or ""))
        tasks = response.get("task_order")
        modalities = response.get("stage_modalities")
        assignments = response.get("modality_assignment")
        if not participant or experience is None or not isinstance(tasks, list):
            continue
        if not isinstance(modalities, dict) or not isinstance(assignments, dict):
            continue
        stages = {
            str(task): stage
            for stage, task in enumerate(tasks, start=1)
            if assignments.get(task) == "assigned" and modalities.get(task) in {"graph", "videos"}
        }
        if not stages:
            continue
        # Scored one response at a time, rather than joining the whole store's
        # scored rows back onto their participants: one response in means every
        # row out is this participant's, with no identity to agree about. The
        # join it replaces was matching nothing at all -- both sides read a
        # participant differently, and `analytics.participant` is the fix for
        # that -- but a stage still has to be matched per task, and doing it
        # here needs neither side to keep spelling a participant the same way.
        for scored in regret.regret_rows([response], rates):
            stage = stages.get(str(scored["task"]))
            # Regret is computed for every scored choice, unlike gain, which
            # is `None` wherever the task's baseline has no rollouts. Guarded
            # anyway: `float(None)` is a TypeError one dataset away, and this
            # reads the metric by name.
            if stage is None or scored.get(regret.DECISION_METRIC) is None:
                continue
            rows.append(
                {
                    "participant": participant,
                    "task": str(scored["task"]),
                    "stage": stage,
                    "experience_level": experience,
                    "graph": int(modalities[scored["task"]] == "graph"),
                    "regret": float(scored[regret.DECISION_METRIC]),
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        return None
    data["task"] = pd.Categorical(data.task)
    data["stage"] = pd.Categorical(data.stage)
    model, posterior = _fit(
        "regret ~ graph + experience_level + task + stage + (1|participant)",
        data,
        subject="data-collection regret",
        draws=draws,
        tune=tune,
    )
    return {
        **_graph_effect(posterior),
        "n": len(data),
        # The same two quantities the rating models return, so the figure that
        # draws marginal means can draw this panel the same way: the adjusted
        # gain each display led to, rather than only the difference between
        # them. Holding task fixed matters more here than anywhere else —
        # cooking's best option is worth +3.3 points and prepare tray's +28.3,
        # so a raw per-display mean is mostly a statement about which tasks
        # each display happened to be shown with.
        "marginal_means": _marginal_means(model, posterior, data),
        "n_by_display": {
            "graph": int((data.graph == 1).sum()),
            "videos": int((data.graph == 0).sum()),
        },
    }
