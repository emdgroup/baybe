"""Test for imposing continuous constraints."""

import warnings

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.constraints import ContinuousLinearConstraint
from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace
from baybe.targets.numerical import NumericalTarget
from tests.conftest import run_iterations
from tests.constraints.test_cardinality_constraint_continuous import (
    _validate_cardinality_constrained_batch,
)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_1"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_equality1(campaign, n_iterations, batch_size):
    """Test equality constraint with equal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert np.allclose(1.0 * res["Conti_finite1"] + 1.0 * res["Conti_finite2"], 0.3)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_equality2(campaign, n_iterations, batch_size):
    """Test equality constraint with unequal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert np.allclose(1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"], 0.3)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_3"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_inequality1(campaign, n_iterations, batch_size):
    """Test inequality constraint with equal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert (1.0 * res["Conti_finite1"] + 1.0 * res["Conti_finite2"]).ge(0.299).all()


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_4"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_inequality2(campaign, n_iterations, batch_size):
    """Test inequality constraint with unequal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert (1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"]).ge(0.299).all()


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_6"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_inequality3(campaign, n_iterations, batch_size):
    """Test inequality constraint with unequal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert (1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"]).le(0.301).all()


@pytest.mark.parametrize("recommender", [RandomRecommender(), BotorchRecommender()])
def test_cardinality_constraint(recommender):
    """Cardinality constraints are taken into account by the recommender."""
    MIN_CARDINALITY = 4
    MAX_CARDINALITY = 7
    BATCH_SIZE = 10

    parameters = [NumericalContinuousParameter(str(i), (0, 1)) for i in range(10)]
    constraints = [
        ContinuousCardinalityConstraint(
            [p.name for p in parameters], MIN_CARDINALITY, MAX_CARDINALITY
        )
    ]
    searchspace = SearchSpace.from_product(parameters, constraints)

    if isinstance(recommender, BayesianRecommender):
        objective = NumericalTarget("t", "MAX").to_objective()
        measurements = pd.DataFrame(searchspace.continuous.sample_uniform(2))
        measurements["t"] = np.random.random(len(measurements))
    else:
        objective = None
        measurements = None

    with warnings.catch_warnings(record=True) as w:
        recommendation = recommender.recommend(
            BATCH_SIZE, searchspace, objective, measurements
        )

    # Assert that the constraint conditions hold
    _validate_cardinality_constrained_batch(
        recommendation, searchspace.continuous, BATCH_SIZE, w
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Conti_finite1", "Conti_finite3", "Conti_finite2"]],
)
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_1"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
def test_hybridspace_eq(campaign, n_iterations, batch_size):
    """Test equality constraint with equal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert np.allclose(1.0 * res["Conti_finite1"] + 1.0 * res["Conti_finite2"], 0.3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Conti_finite1", "Conti_finite3", "Conti_finite2"]],
)
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_3"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("n_grid_points", [5], ids=["grid5"])
def test_hybridspace_ineq(campaign, n_iterations, batch_size):
    """Test inequality constraint with equal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert (1.0 * res["Conti_finite1"] + 1.0 * res["Conti_finite2"]).ge(0.299).all()


@pytest.mark.parametrize(
    ("parameters", "coefficients", "rhs", "op"),
    [
        param(["A", "B"], [1.0], 0.0, "=", id="eq_too_few_coeffs"),
        param(["A", "B"], [1.0, 2.0, 3.0], 0.0, "=", id="eq_too_many_coeffs"),
        param(["A", "B"], [1.0, 2.0], "bla", "=", id="eq_invalid_rhs"),
        param(["A", "B"], [1.0], 0.0, ">=", id="ineq_too_few_coeffs"),
        param(["A", "B"], [1.0, 2.0, 3.0], 0.0, ">=", id="ineq_too_many_coeffs"),
        param(["A", "B"], [1.0, 2.0], "bla", ">=", id="ineq_invalid_rhs"),
        param(["A", "B"], [1.0, 2.0], 0.0, "invalid", id="ineq_invalid_operator1"),
        param(["A", "B"], [1.0, 2.0], 0.0, 2.0, id="ineq_invalid_operator1"),
    ],
)
def test_invalid_constraints(parameters, coefficients, rhs, op):
    """Test invalid continuous constraint creations."""
    with pytest.raises(ValueError):
        ContinuousLinearConstraint(
            parameters=parameters, operator=op, coefficients=coefficients, rhs=rhs
        )
