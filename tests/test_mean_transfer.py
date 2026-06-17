"""Tests for the ``MeanTransferSurrogate``."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.campaign import Campaign
from baybe.exceptions import IncompatibleSurrogateError
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace.core import SearchSpace
from baybe.surrogates import MeanTransferSurrogate
from baybe.targets.numerical import NumericalTarget

# The valid mode combinations to test (the "new" anchor with "discard" init is not
# part of the relevant configurations).
_MODES = [
    param("pretrained", "freeze", id="pretrained-freeze"),
    param("pretrained", "warmstart", id="pretrained-warmstart"),
    param("combined", "freeze", id="combined-freeze"),
    param("combined", "discard", id="combined-discard"),
]


def _objective() -> SingleTargetObjective:
    return SingleTargetObjective(target=NumericalTarget(name="y", minimize=False))


@pytest.fixture(name="searchspace")
def fixture_searchspace() -> SearchSpace:
    """A search space with a single active-value task parameter."""
    x = NumericalDiscreteParameter(name="x", values=np.linspace(-2, 2, 15))
    task = TaskParameter(
        name="Function", values=["target", "source"], active_values=["target"]
    )
    return SearchSpace.from_product(parameters=[x, task])


@pytest.fixture(name="measurements")
def fixture_measurements() -> pd.DataFrame:
    """Source and target measurements of a shifted quadratic function."""
    rng = np.random.default_rng(1337)
    xs = np.linspace(-2, 2, 15)

    def f(x: float, shift: float = 0.0) -> float:
        return -((x - 0.3) ** 2) + shift

    rows = [
        {"x": x, "Function": "source", "y": f(x, 0.1) + rng.normal(0, 0.01)} for x in xs
    ]
    rows += [
        {"x": x, "Function": "target", "y": f(x) + rng.normal(0, 0.01)} for x in xs[::3]
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize(("anchors", "mean_kernel_init"), _MODES)
def test_fit_and_posterior(searchspace, measurements, anchors, mean_kernel_init):
    """The surrogate fits and produces a posterior for each mode."""
    surrogate = MeanTransferSurrogate(
        anchors=anchors, mean_kernel_init=mean_kernel_init
    )
    surrogate.fit(searchspace, _objective(), measurements)

    candidates = measurements[measurements["Function"] == "target"][["x", "Function"]]
    stats = surrogate.posterior_stats(candidates)
    assert len(stats) == len(candidates)


@pytest.mark.parametrize(("anchors", "mean_kernel_init"), _MODES)
def test_recommendation(searchspace, measurements, anchors, mean_kernel_init):
    """The surrogate works as a drop-in for a Bayesian recommender."""
    surrogate = MeanTransferSurrogate(
        anchors=anchors, mean_kernel_init=mean_kernel_init
    )
    recommender = TwoPhaseMetaRecommender(
        recommender=BotorchRecommender(surrogate_model=surrogate)
    )
    campaign = Campaign(
        searchspace=searchspace, objective=_objective(), recommender=recommender
    )
    campaign.add_measurements(measurements)
    recommendation = campaign.recommend(batch_size=3)

    assert len(recommendation) == 3
    # The task column is fixed to the single active value.
    assert (recommendation["Function"] == "target").all()


def test_no_task_parameter(measurements):
    """Fitting without a task parameter raises."""
    searchspace = NumericalDiscreteParameter(
        name="x", values=np.linspace(-2, 2, 15)
    ).to_searchspace()
    surrogate = MeanTransferSurrogate()
    with pytest.raises(IncompatibleSurrogateError, match="exactly one"):
        surrogate.fit(
            searchspace,
            _objective(),
            measurements[measurements["Function"] == "target"].drop(
                columns=["Function"]
            ),
        )


def test_multiple_active_values(measurements):
    """A task parameter with more than one active value raises."""
    x = NumericalDiscreteParameter(name="x", values=np.linspace(-2, 2, 15))
    task = TaskParameter(
        name="Function",
        values=["target", "source"],
        active_values=["target", "source"],
    )
    searchspace = SearchSpace.from_product(parameters=[x, task])
    surrogate = MeanTransferSurrogate()
    with pytest.raises(IncompatibleSurrogateError, match="exactly one active value"):
        surrogate.fit(searchspace, _objective(), measurements)


def test_multiple_source_tasks(searchspace, measurements):
    """More than one source task raises."""
    extra = measurements[measurements["Function"] == "source"].copy()
    extra["Function"] = "source2"
    augmented = pd.concat([measurements, extra], ignore_index=True)
    surrogate = MeanTransferSurrogate()
    with pytest.raises(IncompatibleSurrogateError, match="exactly one source task"):
        surrogate.fit(searchspace, _objective(), augmented)


@pytest.mark.parametrize(
    "missing", ["source", "target"], ids=["no-source", "no-target"]
)
def test_missing_data(searchspace, measurements, missing):
    """Missing source or target data degrades gracefully instead of raising."""
    data = measurements[measurements["Function"] != missing]
    surrogate = MeanTransferSurrogate()
    surrogate.fit(searchspace, _objective(), data)

    candidates = measurements[measurements["Function"] == "target"][["x", "Function"]]
    stats = surrogate.posterior_stats(candidates)
    assert len(stats) == len(candidates)
