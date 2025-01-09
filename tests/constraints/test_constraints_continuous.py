"""Test for imposing continuous constraints."""

import numpy as np
import pytest
from pytest import param

from baybe.constraints import ContinuousLinearConstraint
from tests.conftest import run_iterations

TOLERANCE = 0.01


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


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_1"]])
def test_interpoint_equality_single_parameter(campaign, n_iterations, batch_size):
    """Test single parameter interpoint equality constraint."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    for batch in range(n_iterations):
        res_batch = res[res["BatchNr"] == batch + 1]
        assert np.isclose(res_batch["Conti_finite1"].sum(), 0.3)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_2"]])
def test_interpoint_inequality_single_parameter(campaign, n_iterations, batch_size):
    """Test single parameter interpoint inequality constraint."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    for batch in range(n_iterations):
        res_batch = res[res["BatchNr"] == batch + 1]
        assert 2 * res_batch["Conti_finite1"].sum() >= 0.3 - TOLERANCE


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_3"]])
def test_interpoint_equality_multiple_parameters(campaign, n_iterations, batch_size):
    """Test interpoint equality constraint involving multiple parameters."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    res_grouped = res.groupby("BatchNr")
    interpoint_result = (
        res_grouped["Conti_finite1"].sum() + res_grouped["Conti_finite2"].sum()
    )
    assert np.allclose(interpoint_result, 0.3, atol=TOLERANCE)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_4"]])
def test_geq_interpoint_inequality_multiple_parameters(
    campaign, n_iterations, batch_size
):
    """Test geq-interpoint inequality constraint involving multiple parameters."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    res_grouped = res.groupby("BatchNr")
    interpoint_result = (
        2 * res_grouped["Conti_finite1"].sum() - res_grouped["Conti_finite2"].sum()
    )
    print(f"{interpoint_result=}")
    assert interpoint_result.ge(0.3 - TOLERANCE).all()


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_5"]])
def test_leq_interpoint_inequality_multiple_parameters(
    campaign, n_iterations, batch_size
):
    """Test leq-interpoint inequality constraint involving multiple parameters."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    res_grouped = res.groupby("BatchNr")
    interpoint_result = (
        2 * res_grouped["Conti_finite1"].sum() - res_grouped["Conti_finite2"].sum()
    )
    assert interpoint_result.le(0.3 + TOLERANCE).all()


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize(
    "constraint_names", [["ContiConstraint_4", "InterConstraint_2"]]
)
def test_interpoint_normal_mix(campaign, n_iterations, batch_size):
    """Test mixing interpoint and normal inequality constraints."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    interpoint_result = 2 * res.groupby("BatchNr")["Conti_finite1"].sum()
    assert interpoint_result.ge(0.3 - TOLERANCE).all()
    assert (
        (1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"])
        .ge(0.3 - TOLERANCE)
        .all()
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
