"""Surrogate tests."""

from contextlib import nullcontext
from unittest.mock import patch

import pandas as pd
import pytest
from pytest import param

from baybe.exceptions import IncompatibleSurrogateError
from baybe.objectives.base import Objective
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace.core import SearchSpace
from baybe.surrogates import (
    BayesianLinearSurrogate,
    MeanPredictionSurrogate,
    NGBoostSurrogate,
)
from baybe.surrogates.base import IndependentGaussianSurrogate, Surrogate
from baybe.surrogates.composite import CompositeSurrogate
from baybe.surrogates.custom import CustomONNXSurrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.random_forest import RandomForestSurrogate
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import get_subclasses, is_all_instance
from baybe.utils.dataframe import create_fake_input


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
    "objective",
    [
        SingleTargetObjective(NumericalTarget("t1")),
        DesirabilityObjective(
            [NumericalTarget("t1"), NumericalTarget("t2")],
            scalarizer="MEAN",
            require_normalization=False,
            as_pre_transformation=False,
        ),
        DesirabilityObjective(
            [NumericalTarget("t1"), NumericalTarget("t2")],
            scalarizer="MEAN",
            require_normalization=False,
            as_pre_transformation=True,
        ),
        ParetoObjective([NumericalTarget("t1"), NumericalTarget("t2")]),
    ],
    ids=["single", "desirability", "desirability-pre", "pareto"],
)
@patch.object(
    GaussianProcessSurrogate,
    "_fit",
    side_effect=GaussianProcessSurrogate._fit,
    autospec=True,
)
def test_caching_via_recommender(mock, objective):
    """Surrogates are correctly cached when requested via a recommender."""
    searchspace = NumericalDiscreteParameter("p", [0, 1]).to_searchspace()
    measurements = create_fake_input(searchspace.parameters, objective.targets)
    recommender = BotorchRecommender(GaussianProcessSurrogate())

    for _ in range(2):
        recommender.get_surrogate(searchspace, objective, measurements)
        assert mock.call_count == objective._n_models


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
    measurements = pd.DataFrame({"p": [0, 1], "t1": [0, 1], "t2": [0, 1]})
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


@pytest.mark.parametrize(
    "target_names",
    [["Target_max"], ["Target_max_bounded", "Target_min_bounded"]],
    ids=["single", "multi"],
)
@pytest.mark.parametrize(
    "parameter_names",
    [["Conti_finite1"], ["Categorical_1", "Conti_finite1"]],
    ids=["conti", "hybrid"],
)
@pytest.mark.parametrize(
    "surrogate_model",
    [
        param(GaussianProcessSurrogate(), id="gp"),
        param(MeanPredictionSurrogate(), id="mean"),
        param(RandomForestSurrogate(), id="rf"),
        param(NGBoostSurrogate(), id="ng"),
        param(BayesianLinearSurrogate(), id="lin"),
    ],
)
def test_continuous_incompatibility(campaign):
    """Using surrogates without gradients on continuous spaces fails expectedly."""
    data = create_fake_input(campaign.parameters, campaign.targets)
    campaign.add_measurements(data)

    skip = False
    s = campaign.get_surrogate()
    if isinstance(s, GaussianProcessSurrogate):
        skip = True
    elif isinstance(s, CompositeSurrogate) and is_all_instance(
        s._surrogates_flat, GaussianProcessSurrogate
    ):
        skip = True

    with (
        nullcontext()
        if skip
        else pytest.raises(
            IncompatibleSurrogateError,
            match="does not support the required gradient computation",
        )
    ):
        campaign.recommend(1)


@pytest.mark.parametrize(
    "surrogate",
    [
        cls()
        for cls in get_subclasses(IndependentGaussianSurrogate)
        if cls != CustomONNXSurrogate
    ],
    ids=lambda surrogate: surrogate.__class__.__name__,
)
def test_batching_incompatibility(
    surrogate: Surrogate, searchspace: SearchSpace, objective: Objective
):
    """Surrogates that do not support batch predictions reject batch requests."""
    measurements = create_fake_input(
        searchspace.parameters, objective.targets, n_rows=2
    )
    surrogate.fit(searchspace, objective, measurements)

    with pytest.raises(
        IncompatibleSurrogateError,
        match="cannot be used for joint posterior evaluation",
    ):
        surrogate.posterior(measurements, joint=True)
