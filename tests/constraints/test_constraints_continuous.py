"""Test for imposing continuous constraints."""

import numpy as np
import pytest
from pytest import param

from baybe.campaign import Campaign
from baybe.constraints import ContinuousLinearConstraint
from baybe.searchspace import SearchSpace
from tests.conftest import run_iterations

TOLERANCE = 0.01


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize(
    ("constraint_names", "coef1", "coef2", "expected_value", "check_type"),
    [
        param(
            ["ContiConstraint_2"],
            1.0,
            3.0,
            0.3,
            "eq",
            id="eq",
        ),
        param(
            ["ContiConstraint_4"],
            1.0,
            3.0,
            0.299,
            "ge",
            id="ge",
        ),
        param(
            ["ContiConstraint_6"],
            1.0,
            3.0,
            0.301,
            "le",
            id="le",
        ),
    ],
)
def test_intrapoint_linear_constraints(
    campaign, n_iterations, batch_size, coef1, coef2, expected_value, check_type
):
    """Test intrapoint linear constraints with various operators and weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements

    result = coef1 * res["Conti_finite1"] + coef2 * res["Conti_finite2"]

    if check_type == "eq":
        assert np.allclose(result, expected_value)
    elif check_type == "ge":
        assert result.ge(expected_value).all()
    elif check_type == "le":
        assert result.le(expected_value).all()


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize(
    ("constraint_names", "calculation", "expected_value", "check_type"),
    [
        param(
            ["InterConstraint_1"],
            lambda grouped: grouped["Conti_finite1"].sum(),
            0.3,
            "eq",
            id="equality_single_param",
        ),
        param(
            ["InterConstraint_2"],
            lambda grouped: 2 * grouped["Conti_finite1"].sum(),
            0.3,
            "ge",
            id="inequality_ge_single_param",
        ),
        param(
            ["InterConstraint_3"],
            lambda grouped: (
                grouped["Conti_finite1"].sum() + 2 * grouped["Conti_finite2"].sum()
            ),
            0.3,
            "eq",
            id="equality_multiple_params",
        ),
        param(
            ["InterConstraint_4"],
            lambda grouped: (
                2 * grouped["Conti_finite1"].sum() - grouped["Conti_finite2"].sum()
            ),
            0.3,
            "ge",
            id="inequality_ge_multiple_params",
        ),
        param(
            ["InterConstraint_5"],
            lambda grouped: (
                2 * grouped["Conti_finite1"].sum() - grouped["Conti_finite2"].sum()
            ),
            0.3,
            "le",
            id="inequality_le_multiple_params",
        ),
    ],
)
def test_interpoint_linear_constraints(
    non_sequential_recommender,
    parameters,
    constraints,
    objective,
    n_iterations,
    batch_size,
    calculation,
    expected_value,
    check_type,
):
    """Test interpoint linear constraints with various operators and parameters."""
    campaign = Campaign(
        searchspace=SearchSpace.from_product(
            parameters=parameters, constraints=constraints
        ),
        recommender=non_sequential_recommender,
        objective=objective,
    )
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements

    res_grouped = res.groupby("BatchNr")
    interpoint_result = calculation(res_grouped)

    if check_type == "eq":
        assert np.allclose(interpoint_result, expected_value, atol=TOLERANCE)
    elif check_type == "ge":
        assert interpoint_result.ge(expected_value - TOLERANCE).all()
    elif check_type == "le":
        assert interpoint_result.le(expected_value + TOLERANCE).all()


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize(
    "constraint_names", [["ContiConstraint_4", "InterConstraint_2"]]
)
def test_interpoint_intrapoint_mix(
    non_sequential_recommender,
    parameters,
    constraints,
    objective,
    n_iterations,
    batch_size,
):
    """Test mixing interpoint and intrapoint inequality constraints."""
    campaign = Campaign(
        searchspace=SearchSpace.from_product(
            parameters=parameters, constraints=constraints
        ),
        recommender=non_sequential_recommender,
        objective=objective,
    )
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements

    interpoint_result = 2 * res.groupby("BatchNr")["Conti_finite1"].sum()
    assert interpoint_result.ge(0.3 - TOLERANCE).all()
    assert (
        (1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"])
        .ge(0.3 - TOLERANCE)
        .all()
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("constraint_names", "expected_value", "check_type"),
    [
        param(["ContiConstraint_1"], 0.3, "eq", id="equality"),
        param(["ContiConstraint_3"], 0.299, "ge", id="inequality_ge"),
    ],
)
def test_hybridspace_linear_constraints(
    campaign, n_iterations, batch_size, expected_value, check_type
):
    """Test linear constraints in hybrid search spaces."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements

    result = 1.0 * res["Conti_finite1"] + 1.0 * res["Conti_finite2"]

    if check_type == "eq":
        assert np.allclose(result, expected_value)
    elif check_type == "ge":
        assert result.ge(expected_value).all()


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
