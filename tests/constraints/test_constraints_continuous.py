"""Test for imposing continuous constraints."""

import numpy as np
import pytest
import torch
from pytest import param

from baybe.constraints import ContinuousLinearConstraint
from baybe.parameters.numerical import NumericalContinuousParameter
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
    campaign_non_sequential,
    n_iterations,
    batch_size,
    calculation,
    expected_value,
    check_type,
):
    """Test interpoint linear constraints with various operators and parameters."""
    run_iterations(campaign_non_sequential, n_iterations, batch_size, add_noise=False)
    res = campaign_non_sequential.measurements

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
    campaign_non_sequential,
    n_iterations,
    batch_size,
):
    """Test mixing interpoint and intrapoint inequality constraints."""
    run_iterations(campaign_non_sequential, n_iterations, batch_size, add_noise=False)
    res = campaign_non_sequential.measurements

    interpoint_result = 2 * res.groupby("BatchNr")["Conti_finite1"].sum()
    assert interpoint_result.ge(0.3 - TOLERANCE).all()
    assert (
        (1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"])
        .ge(0.3 - TOLERANCE)
        .all()
    )


@pytest.mark.parametrize("flatten", [True, False], ids=["1d", "2d"])
@pytest.mark.parametrize("interpoint", [False, True], ids=["intra", "inter"])
def test_to_botorch(flatten: bool, interpoint: bool):
    """BoTorch conversion of constraints yields the correct indices."""
    n_disc_parameters = 3
    batch_size = 2
    cont_parameters = [NumericalContinuousParameter(f"c{i}", (0, 1)) for i in range(5)]
    constraint_tuple = ContinuousLinearConstraint(
        ["c1", "c3"], "<=", [3, 5], interpoint=interpoint
    ).to_botorch(
        cont_parameters,
        idx_offset=n_disc_parameters,
        batch_size=batch_size if flatten or interpoint else None,
        flatten=flatten,
    )

    if flatten:
        if interpoint:
            idxs = [[4, 6, 12, 14]]
        else:
            idxs = [[4, 6], [12, 14]]
    elif interpoint:
        idxs = [[[0, 4], [0, 6], [1, 4], [1, 6]]]
    else:
        idxs = [[4, 6]]
    for idx, t in zip(idxs, constraint_tuple):
        assert t[0].equal(torch.tensor(idx))


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Conti_finite1", "Conti_finite3", "Conti_finite2"]],
)
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_1"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_hybridspace_eq(campaign, n_iterations, batch_size):
    """Test equality constraint with equal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements

    assert np.allclose(1.0 * res["Conti_finite1"] + 1.0 * res["Conti_finite2"], 0.3)


@pytest.mark.slow
@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Conti_finite1", "Conti_finite3", "Conti_finite2"]],
)
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_3"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_hybridspace_ineq(campaign, n_iterations, batch_size):
    """Test inequality constraint with equal weights."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements

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
