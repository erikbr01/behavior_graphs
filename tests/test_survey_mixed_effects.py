"""The adjusted mixed models behind the paper figure's intervals.

Fitting is not what is tested here — that is bambi's job, and a four-chain fit
per assertion would make this file the slowest in the suite. What is tested is
the boundary: which responses reach a model at all, and what a store too small
to support one costs its caller. The pages that fit these models render the
whole study beneath them, so a response set that cannot be modelled has to come
back as a message rather than as an exception out of a sampler's internals.
"""

from __future__ import annotations

import pytest

pytest.importorskip("bambi")
pytest.importorskip("pandas")

from behavior_graphs.viz.streamlit.user_study import mixed_effects  # noqa: E402

# pytest puts this file's directory on sys.path (tests/ is not a package).
from test_survey_analytics import CONTRASTIVE, PRODUCTION, THREE_STAGE  # noqa: E402


def test_a_store_too_small_to_support_the_model_is_a_message_and_not_a_crash():
    """One participant leaves the design matrix degenerate, and bambi says so
    with an IndexError out of its term-construction internals. Reaching the
    page, that takes down every panel below the figure; as a ModelFitError it
    costs the figure and nothing else."""
    with pytest.raises(mixed_effects.ModelFitError) as caught:
        mixed_effects.continuous_effects([THREE_STAGE], draws=20, tune=20)

    # Worth showing: it names which outcome could not be fit, how many
    # responses it had, and what would fix it.
    message = str(caught.value)
    assert any(outcome in message for outcome in ("temporal demand", "effort", "confidence"))
    assert "randomized stages" in message


def test_the_fit_failure_is_catchable_as_the_runtime_error_the_pages_already_catch():
    assert issubclass(mixed_effects.ModelFitError, RuntimeError)


def test_layouts_that_never_randomized_a_display_reach_no_model():
    """Only the three-stage layout assigns a display at random, which is what
    makes the graph coefficient an effect rather than a preference. The other
    layouts contribute no rows, so there is nothing to fit and no error."""
    assert mixed_effects.assigned_rating_rows([PRODUCTION, CONTRASTIVE]) == []
    assert mixed_effects.continuous_effects([PRODUCTION, CONTRASTIVE]) == []


def test_regret_needs_a_scored_choice_and_returns_nothing_without_one():
    """No retrained-policy rates, so no choice can be scored: the caller gets
    None and drops the panel, rather than a model fit to zero rows."""
    assert mixed_effects.decision_regret_effect([THREE_STAGE], {}) is None


@pytest.mark.parametrize(
    ("columns", "raises_on_new_api"),
    [
        # ArviZ 1.x: the keyword the code reaches for first, and `_lb`/`_ub`.
        ({"mean": [-8.0], "hdi_95%_lb": [-16.0], "hdi_95%_ub": [1.0]}, False),
        # ArviZ 0.x: `ci_prob` is not a keyword, and the bounds are percentiles.
        ({"mean": [-8.0], "hdi_2.5%": [-16.0], "hdi_97.5%": [1.0]}, True),
    ],
    ids=["arviz_1x", "arviz_0x"],
)
def test_the_effect_is_read_off_either_arviz_summary_api(monkeypatch, columns, raises_on_new_api):
    """``arviz>=0.17`` is what the `analysis` extra asks for, and the 1.0 rename
    moved both the keyword and the column names. A developer's environment and
    the deployed image sit on opposite sides of it, which is exactly how a
    version-specific call passes every test here and then puts a TypeError on
    the page of the running container.
    """
    import arviz as az
    import pandas as pd

    def summary(posterior, var_names=None, hdi_prob=None, **kwargs):
        if raises_on_new_api and ("ci_prob" in kwargs or "ci_kind" in kwargs):
            raise TypeError("summary() got an unexpected keyword argument 'ci_prob'")
        return pd.DataFrame(columns, index=["graph"])

    monkeypatch.setattr(az, "summary", summary)

    assert mixed_effects._graph_effect(object()) == {
        "mean": -8.0,
        "lower": -16.0,
        "upper": 1.0,
    }


def _summary(monkeypatch, **extra):
    """Stub `az.summary` with a one-row frame carrying *extra* columns."""
    import arviz as az
    import pandas as pd

    columns = {"mean": [-8.0], "hdi_95%_lb": [-16.0], "hdi_95%_ub": [1.0], **extra}
    monkeypatch.setattr(
        az, "summary", lambda posterior, **kwargs: pd.DataFrame(columns, index=["graph"])
    )


def test_the_sampler_diagnostics_come_back_with_the_estimate(monkeypatch):
    """`mcse_mean` is the one that says how many digits of an effect are the
    study rather than the sampler, and the fit is seeded — so the error it
    measures is reproducible, which is how it ends up quoted as a result."""
    _summary(monkeypatch, r_hat=[1.001], ess_bulk=[3142.0], mcse_mean=[0.11])

    effect = mixed_effects._graph_effect(object())

    assert effect["r_hat"] == pytest.approx(1.001)
    assert effect["ess_bulk"] == pytest.approx(3142.0)
    assert effect["mcse_mean"] == pytest.approx(0.11)


def test_chains_that_did_not_converge_are_refused_rather_than_summarised(monkeypatch):
    """A sampler that fails raises already. This is the one that finishes and
    hands back numbers indistinguishable from good ones."""
    _summary(monkeypatch, r_hat=[1.4], ess_bulk=[31.0], mcse_mean=[4.2])

    with pytest.raises(mixed_effects.ModelFitError, match="did not converge"):
        mixed_effects._graph_effect(object())


def test_an_arviz_that_reports_no_diagnostics_still_gives_an_estimate():
    """They are not part of the summary API that moved at 1.0, but neither was
    `hdi_prob` — so their absence is a missing check, not a failure."""
    import arviz as az
    import pandas as pd

    frame = pd.DataFrame(
        {"mean": [-8.0], "hdi_95%_lb": [-16.0], "hdi_95%_ub": [1.0]}, index=["graph"]
    )
    original = az.summary
    az.summary = lambda posterior, **kwargs: frame
    try:
        effect = mixed_effects._graph_effect(object())
    finally:
        az.summary = original

    assert set(effect) == {"mean", "lower", "upper"}


def test_the_marginal_means_are_population_level_and_per_display():
    """Both halves of "estimated marginal mean" are choices that can be made
    wrong silently.

    ``include_group_specific=False`` is what makes it a population-level mean
    rather than one for the 18 participants in hand: with the intercepts left
    in, the estimate answers "what did these people rate" and the interval
    shrinks to the sample. And the scenario has to set ``graph`` for every row,
    not filter to the rows that had it — filtering would average each display
    over the tasks *it* was shown with, which is the confound the adjustment
    exists to remove.
    """
    import types

    import numpy as np
    import pandas as pd

    data = pd.DataFrame(
        {
            "score": [20.0, 40.0, 60.0, 80.0],
            "graph": [1, 0, 1, 0],
            "task": ["cooking", "prepare_tray", "prepare_tray", "cooking"],
        }
    )
    seen = []

    class _Model:
        def predict(self, posterior, *, kind, data, include_group_specific, inplace):
            seen.append(
                {
                    "kind": kind,
                    "include_group_specific": include_group_specific,
                    "inplace": inplace,
                    "graph": sorted(data.graph.unique()),
                    "rows": len(data),
                    "tasks": sorted(data.task.unique()),
                }
            )
            # Two chains of three draws over four rows; the mean over rows is
            # the row value, so the summary below is arithmetic we can check.
            mu = np.full((2, 3, len(data)), 10.0 * (1 + data.graph.iloc[0]))
            return types.SimpleNamespace(posterior={"mu": types.SimpleNamespace(values=mu)})

    means = mixed_effects._marginal_means(_Model(), object(), data)

    assert [call["include_group_specific"] for call in seen] == [False, False]
    assert [call["inplace"] for call in seen] == [False, False]
    # Videos first, then graph, each over *every* row and both tasks.
    assert [call["graph"] for call in seen] == [[0], [1]]
    assert [call["rows"] for call in seen] == [4, 4]
    assert [call["tasks"] for call in seen] == [["cooking", "prepare_tray"]] * 2
    assert means["videos"]["mean"] == pytest.approx(10.0)
    assert means["graph"]["mean"] == pytest.approx(20.0)
    # A point posterior has a zero-width interval rather than no interval.
    assert means["graph"]["lower"] == pytest.approx(20.0)
    assert means["graph"]["upper"] == pytest.approx(20.0)
