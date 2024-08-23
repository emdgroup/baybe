"""Test for imposing continuous constraints."""

import sys

import numpy as np
import pytest

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from tests.conftest import run_iterations
from tests.constraints.test_cardinality_constraint_continuous import _validate_samples


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter_names", [["Conti_finite1", "Conti_finite2", "Conti_finite3"]]
)
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_5"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_cardinality_constraint(campaign, n_iterations, batch_size):
    """Test cardinality constraint for both random recommender and botorch
    recommender."""  # noqa

    MIN_CARDINALITY = 0
    MAX_CARDINALITY = 2
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    recommendations = campaign.measurements

    print(recommendations)

    # Assert that conditions listed in_validate_samples() are fulfilled
    for i_batch in range(2):
        _validate_samples(
            recommendations.loc[
                0 + i_batch * batch_size : (i_batch + 1) * batch_size - 1,
                ["Conti_finite1", "Conti_finite2", "Conti_finite3"],
            ],
            max_cardinality=MAX_CARDINALITY,
            min_cardinality=MIN_CARDINALITY,
            batch_size=batch_size,
            threshold=sys.float_info.min,
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


def test_invalid_constraints():
    """Test invalid continuous constraint creations."""
    # number of parameters and coefficients doesn't match

    with pytest.raises(ValueError):
        ContinuousLinearEqualityConstraint(
            parameters=["A", "B"], coefficients=[1.0], rhs=0.0
        )

    with pytest.raises(ValueError):
        ContinuousLinearEqualityConstraint(
            parameters=["A", "B"], coefficients=[1.0, 2.0, 3.0], rhs=0.0
        )

    with pytest.raises(ValueError):
        ContinuousLinearInequalityConstraint(
            parameters=["A", "B"], coefficients=[1.0], rhs=0.0
        )

    with pytest.raises(ValueError):
        ContinuousLinearInequalityConstraint(
            parameters=["A", "B"], coefficients=[1.0, 2.0, 3.0], rhs=0.0
        )
