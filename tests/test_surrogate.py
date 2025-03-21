"""Surrogate tests."""

from unittest.mock import patch

import pandas as pd
import pytest

from baybe.objectives.pareto import ParetoObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.surrogates.composite import CompositeSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.random_forest import RandomForestSurrogate
from baybe.targets._deprecated import NumericalTarget


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
    "surrogate",
    [
        CompositeSurrogate(
            {"t1": GaussianProcessSurrogate(), "t2": RandomForestSurrogate()},
        ),
        CompositeSurrogate.from_replication(GaussianProcessSurrogate()),
    ],
    ids=["via_init", "via_template"],
)
def test_composite_surrogates(surrogate):
    """Composition yields a valid surrogate."""
    t1 = NumericalTarget("t1", "MAX")
    t2 = NumericalTarget("t2", "MIN")
    searchspace = NumericalDiscreteParameter("p", [0, 1]).to_searchspace()
    objective = ParetoObjective([t1, t2])
    measurements = pd.DataFrame({"p": [0], "t1": [0], "t2": [0]})
    BotorchRecommender(surrogate_model=surrogate).recommend(
        2, searchspace, objective, measurements
    )
