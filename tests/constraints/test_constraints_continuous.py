"""Test for imposing continuous constraints."""

import numpy as np
import pytest

from baybe.constraints import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from tests.conftest import run_iterations


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
@pytest.mark.parametrize("constraint_names", [["InterConstraint_1"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_interpoint_equality_single_parameter(campaign, n_iterations, batch_size):
    """Test single parameter inter-point equality constraint."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert np.isclose(
        res.at[0, "Conti_finite1"] + 3.0 * res.at[1, "Conti_finite1"], 0.3
    )


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_interpoint_inequality_single_parameter(campaign, n_iterations, batch_size):
    """Test single parameter inter-point inequality constraint."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert res.at[0, "Conti_finite1"] + 3.0 * res.at[1, "Conti_finite1"] >= 0.299


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_3"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_interpoint_equality_multiple_parameters(campaign, n_iterations, batch_size):
    """Test inter-point equality constraint involving multiple parameters."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert np.isclose(
        res.at[0, "Conti_finite1"] + 3.0 * res.at[1, "Conti_finite2"], 0.3
    )


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_4"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_interpoint_inequality_multiple_parameters(campaign, n_iterations, batch_size):
    """Test inter-point inequality constraint involving multiple parameters."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert res.at[0, "Conti_finite1"] + 3.0 * res.at[1, "Conti_finite2"] >= 0.299


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("constraint_names", [["InterConstraint_2"]])
@pytest.mark.parametrize("batch_size", [1], ids=["b1"])
def test_interpoint_small_batch_size(campaign, n_iterations, batch_size):
    """Fail if requesting too small batch."""
    print(f"{batch_size=}")
    with pytest.raises(ValueError):
        run_iterations(campaign, n_iterations, batch_size, add_noise=False)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize(
    "constraint_names", [["ContiConstraint_4", "InterConstraint_2"]]
)
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_interpoint_normal_mix(campaign, n_iterations, batch_size):
    """Test mixing interpoint and normal inequality constraints."""
    run_iterations(campaign, n_iterations, batch_size, add_noise=False)
    res = campaign.measurements
    print(res)

    assert res.at[0, "Conti_finite1"] + 3.0 * res.at[1, "Conti_finite1"] >= 0.299
    assert (1.0 * res["Conti_finite1"] + 3.0 * res["Conti_finite2"]).ge(0.299).all()


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

    with pytest.raises(ValueError):
        ContinuousLinearEqualityConstraint(
            parameters=["A", "B"], coefficients=["C", 0, 1], rhs=0.0
        )

    with pytest.raises(ValueError):
        ContinuousLinearInequalityConstraint(
            parameters=["A", "B"], coefficients=["C", 0, 1], rhs=0.0
        )

    with pytest.raises(ValueError):
        ContinuousLinearEqualityConstraint(
            parameters=["A"], coefficients=["A", 0], rhs=0.0
        )

    with pytest.raises(ValueError):
        ContinuousLinearInequalityConstraint(
            parameters=["A"], coefficients=["A", 0], rhs=0.0
        )
