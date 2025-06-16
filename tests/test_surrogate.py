"""Surrogate tests."""

from unittest.mock import patch

import pandas as pd
import pytest
from pytest import param

from baybe.objectives.pareto import ParetoObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.surrogates import BayesianLinearSurrogate, NGBoostSurrogate
from baybe.surrogates.composite import CompositeSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.random_forest import RandomForestSurrogate
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
    t1 = NumericalTarget("t1")
    t2 = NumericalTarget("t2", minimize=True)
    searchspace = NumericalDiscreteParameter("p", [0, 1]).to_searchspace()
    objective = ParetoObjective([t1, t2])
    measurements = pd.DataFrame({"p": [0], "t1": [0], "t2": [0]})
    BotorchRecommender(surrogate_model=surrogate).recommend(
        2, searchspace, objective, measurements
    )


@pytest.mark.parametrize(
    "model_cls, params",
    [
        param(RandomForestSurrogate, {"n_estimators": 5}, id="rf_1"),
        param(RandomForestSurrogate, {"criterion": "squared_error"}, id="rf_2"),
        param(RandomForestSurrogate, {"max_features": 0.4}, id="rf_3"),
        param(RandomForestSurrogate, {"monotonic_cst": None}, id="rf_4"),
        param(NGBoostSurrogate, {"n_estimators": 5}, id="ng_1"),
        param(NGBoostSurrogate, {"learning_rate": 0.05}, id="ng_2"),
        param(NGBoostSurrogate, {"verbose": True}, id="ng_3"),
        param(NGBoostSurrogate, {"early_stopping_rounds": None}, id="ng_4"),
        param(BayesianLinearSurrogate, {"max_iter": 10}, id="lin_1"),
        param(BayesianLinearSurrogate, {"alpha_1": 0.3}, id="lin_2"),
        param(BayesianLinearSurrogate, {"fit_intercept": True}, id="lin_3"),
    ],
)
def test_valid_model_params(model_cls, params):
    """Ensure valid model_params can be set."""
    model_cls(model_params=params)


@pytest.mark.parametrize(
    "model_cls, params",
    [
        param(RandomForestSurrogate, {"n_estimators": 5.3}, id="rf_float_not_int"),
        param(
            RandomForestSurrogate,
            {"criterion": "squared_error123"},
            id="rf_wrong_literal",
        ),
        param(RandomForestSurrogate, {"max_features": True}, id="rf_unsupported_bool"),
        param(
            RandomForestSurrogate, {"min_samples_split": None}, id="rf_unsupported_none"
        ),
        param(RandomForestSurrogate, {"bla": None}, id="rf_extra_key"),
        param(NGBoostSurrogate, {"n_estimators": 5.3}, id="ng_float_not_int"),
        param(NGBoostSurrogate, {"learning_rate": 4}, id="ng_int_not_float"),
        param(NGBoostSurrogate, {"verbose": 1.0}, id="ng_float_not_bool"),
        param(
            NGBoostSurrogate, {"early_stopping_rounds": 3.4}, id="ng_unsupported_float"
        ),
        param(NGBoostSurrogate, {"col_sample": None}, id="ng_unsupported_none"),
        param(NGBoostSurrogate, {"bla": 3.4}, id="ng_extra_key"),
        param(BayesianLinearSurrogate, {"max_iter": 10.3}, id="lin_float_not_int"),
        param(BayesianLinearSurrogate, {"alpha_1": 4}, id="lin_int_not_float"),
        param(BayesianLinearSurrogate, {"fit_intercept": 0.0}, id="lin_float_not_bool"),
        param(BayesianLinearSurrogate, {"copy_X": None}, id="lin_unsupported_none"),
        param(BayesianLinearSurrogate, {"bla": 0.0}, id="lin_extra_key"),
    ],
)
def test_invalid_model_params(model_cls, params):
    """Ensure valid model_params cannot be set."""
    with pytest.raises(TypeError, match="model_params"):
        model_cls(model_params=params)
