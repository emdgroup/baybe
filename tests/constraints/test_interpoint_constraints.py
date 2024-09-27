"""Test for imposing interpoint constraints."""

import numpy as np
import pytest

from tests.conftest import run_iterations


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("interpoint_constraints_names", [["InterConstraint_1"]])
def test_interpoint_equality_single_parameter(
    campaign, n_iterations, batch_size, interpoint_constraints
):
    """Test single parameter inter-point equality constraint."""
    run_iterations(
        campaign,
        n_iterations,
        batch_size,
        add_noise=False,
        interpoint_constraints=interpoint_constraints,
    )
    res = campaign.measurements
    for batch in range(n_iterations):
        res_batch = res[res["BatchNr"] == batch + 1]
        assert np.isclose(res_batch["Conti_finite1"].sum(), 0.3)


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("interpoint_constraints_names", [["InterConstraint_2"]])
def test_interpoint_inequality_single_parameter(
    campaign, n_iterations, batch_size, interpoint_constraints
):
    """Test single parameter inter-point inequality constraint."""
    run_iterations(
        campaign,
        n_iterations,
        batch_size,
        add_noise=False,
        interpoint_constraints=interpoint_constraints,
    )
    res = campaign.measurements
    for batch in range(n_iterations):
        res_batch = res[res["BatchNr"] == batch + 1]
        assert 2 * res_batch["Conti_finite1"].sum() >= 0.299


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("interpoint_constraints_names", [["InterConstraint_3"]])
def test_interpoint_equality_multiple_parameters(
    campaign, n_iterations, batch_size, interpoint_constraints
):
    """Test single parameter inter-point equality constraint."""
    run_iterations(
        campaign,
        n_iterations,
        batch_size,
        add_noise=False,
        interpoint_constraints=interpoint_constraints,
    )
    res = campaign.measurements
    for batch in range(n_iterations):
        res_batch = res[res["BatchNr"] == batch + 1]
        assert np.isclose(
            res_batch["Conti_finite1"].sum() + res_batch["Conti_finite2"].sum(), 0.3
        )


@pytest.mark.parametrize("parameter_names", [["Conti_finite1", "Conti_finite2"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
@pytest.mark.parametrize("interpoint_constraints_names", [["InterConstraint_4"]])
def test_interpoint_inequality_multiple_parameters(
    campaign, n_iterations, batch_size, interpoint_constraints
):
    """Test single parameter inter-point inequality constraint."""
    run_iterations(
        campaign,
        n_iterations,
        batch_size,
        add_noise=False,
        interpoint_constraints=interpoint_constraints,
    )
    res = campaign.measurements
    for batch in range(n_iterations):
        res_batch = res[res["BatchNr"] == batch + 1]
        assert (
            2 * res_batch["Conti_finite1"].sum() - res_batch["Conti_finite2"].sum()
            >= 0.299
        )
