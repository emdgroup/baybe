"""Surrogate tests."""

from unittest.mock import patch

import pandas as pd
import pytest

from baybe.objectives.pareto import ParetoObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.surrogates.composite import BroadcastingSurrogate, CompositeSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget


@patch.object(GaussianProcessSurrogate, "_fit")
def test_caching(patched, searchspace, objective, fake_measurements):
    """A second fit call with the same context does not trigger retraining."""
    # Prepare the setting
    surrogate = GaussianProcessSurrogate()

    # First call
    surrogate.fit(searchspace, objective, fake_measurements)
    patched.assert_called()

    patched.reset_mock()

    # Second call
    surrogate.fit(searchspace, objective, fake_measurements)
    patched.assert_not_called()


@pytest.mark.parametrize(
    "meta_surrogate_cls", [BroadcastingSurrogate, CompositeSurrogate]
)
def test_composite_surrogates(meta_surrogate_cls):
    """Composition yields a valid surrogate."""
    t1 = NumericalTarget("t1", "MAX")
    t2 = NumericalTarget("t1", "MIN")
    searchspace = NumericalContinuousParameter("p", [0, 1]).to_searchspace()
    objective = ParetoObjective([t1, t2])
    measurements = pd.DataFrame({"p": [0], "t1": [0], "t2": [0]})

    if issubclass(meta_surrogate_cls, BroadcastingSurrogate):
        surrogate = BroadcastingSurrogate(GaussianProcessSurrogate())
    else:
        surrogate = CompositeSurrogate(
            {"t1": GaussianProcessSurrogate(), "t2": RandomRecommender()}
        )
    BotorchRecommender(surrogate_model=surrogate).recommend(
        2, searchspace, objective, measurements
    )
