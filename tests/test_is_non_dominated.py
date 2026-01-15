"""Test is_non_dominated functionality for objective."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.exceptions import (
    IncompatibilityError,
    NoMeasurementsError,
    NothingToComputeError,
)
from baybe.parameters import NumericalDiscreteParameter


@pytest.mark.parametrize(
    ("objective_name", "target_names"),
    [
        param("pareto", ["Target_max", "Target_min"], id="pareto_max_min"),
        param("pareto", ["Target_min", "Target_max"], id="pareto_min_max"),
        param(
            "pareto",
            ["Target_min", "Target_max", "Target_match_triangular"],
            id="pareto",
        ),
        param("single", ["Target_max"], id="single"),
        param(
            "desirability",
            ["Target_match_triangular", "Target_min_bounded"],
            id="desirability_match_min",
        ),
    ],
)
@pytest.mark.parametrize(
    "consider_campaign_measurements",
    [True, False],
    ids=["consider_m", "not_consider_m"],
)
def test_is_non_dominated_func_call(
    ongoing_campaign,
    objective,
    parameters,
    batch_size,
    targets,
    fake_measurements,
    consider_campaign_measurements,
):
    """Test function call and expected output size."""
    # Non dominated points for measurements
    non_dominated_campaign_default = ongoing_campaign.is_non_dominated()
    assert len(ongoing_campaign.measurements) == len(non_dominated_campaign_default), (
        "The non-dominated points are computed for the campaign's measurements, but "
        f"the output data of {ongoing_campaign.is_non_dominated.__name__} "
        f"does not have the same length ({len(non_dominated_campaign_default)}) as the "
        f"campaign's measurements ({len(ongoing_campaign.measurements)})."
    )

    # From campaign
    non_dominated_campaign = ongoing_campaign.is_non_dominated(
        fake_measurements, consider_campaign_measurements=consider_campaign_measurements
    )
    assert len(fake_measurements) == len(non_dominated_campaign)

    # From objective, not considering campaign measurements in any case
    non_dominated_objective = ongoing_campaign.objective.is_non_dominated(
        fake_measurements
    )
    assert len(fake_measurements) == len(non_dominated_objective)

    # If the flag is False, the results should be equal. If the flag is True, the
    # results may differ, but not always. For example, if all points in the campaign are
    # dominated by points in the fake_measurements, the results will be identical.
    # Hence, not testing for that case here.
    if not consider_campaign_measurements:
        assert non_dominated_campaign.equals(non_dominated_objective)


@pytest.mark.parametrize(
    "objective_name, target_names",
    [
        param(None, ["Target_max"], id="none_max"),
    ],
)
def test_incompatibility(campaign, objective, fake_measurements):
    """Test for incompatibility when objective is ``None``."""
    with pytest.raises(IncompatibilityError):
        campaign.is_non_dominated()

    with pytest.raises(IncompatibilityError):
        campaign.is_non_dominated(fake_measurements)


@pytest.mark.parametrize(
    "objective_name, target_names",
    [
        param("pareto", ["Target_max", "Target_min"], id="pareto_max_min"),
        param(
            "desirability",
            ["Target_match_triangular", "Target_min_bounded"],
            id="desirability_max_min_bound",
        ),
        param("single", ["Target_max"], id="single_max"),
    ],
)
def test_logic_consider_campaign_measurements(campaign, objective, fake_measurements):
    """Test that exceptions are raised for invalid input combinations."""
    # Test flag when campaign has no measurements
    with pytest.raises(NoMeasurementsError):
        campaign.is_non_dominated(consider_campaign_measurements=True)

    with pytest.raises(NothingToComputeError):
        campaign.is_non_dominated(consider_campaign_measurements=False)

    with pytest.warns(UserWarning):
        campaign.is_non_dominated(
            fake_measurements, consider_campaign_measurements=True
        )

    campaign.is_non_dominated(fake_measurements, consider_campaign_measurements=False)

    # Test flag when campaign has measurements
    campaign.add_measurements(fake_measurements)

    with pytest.raises(NothingToComputeError):
        campaign.is_non_dominated(consider_campaign_measurements=False)

    campaign.is_non_dominated(consider_campaign_measurements=True)
    campaign.is_non_dominated(fake_measurements, consider_campaign_measurements=True)
    campaign.is_non_dominated(fake_measurements, consider_campaign_measurements=False)


@pytest.mark.parametrize(
    "parameters",
    [
        param(
            [
                NumericalDiscreteParameter(
                    name="param1",
                    values=tuple(np.linspace(0, 200, 5)),
                    tolerance=0.1,
                ),
                NumericalDiscreteParameter(
                    name="param2",
                    values=tuple(np.linspace(0, 200, 5)),
                    tolerance=0.1,
                ),
            ],
            id="2params",
        ),
    ],
)
@pytest.mark.parametrize(
    "target_names, idx_non_dominated, objective_name",
    [
        param(["Target_min_bounded", "Target_min"], [0], "pareto", id="pareto_min_min"),
        param(["Target_max", "Target_min"], [0, 2, 5], "pareto", id="pareto_max_min"),
        param(["Target_min", "Target_max"], [0, 2, 3], "pareto", id="pareto_min_max"),
        param(["Target_max_bounded", "Target_max"], [2], "pareto", id="pareto_max_max"),
        param(
            ["Target_match_bell", "Target_match_triangular"],
            [1, 4],
            "pareto",
            id="pareto_match_bell_trnglr",
        ),
        param(["Target_min"], [0], "single", id="single_min"),
        param(["Target_max"], [2], "single", id="single_max"),
        param(["Target_match_bell"], [1, 4], "single", id="single_bell"),
        param(
            ["Target_min_bounded", "Target_max_bounded"],
            [
                3,
            ],
            "desirability",
            id="desirability_min_max",
        ),
        param(
            ["Target_max_bounded", "Target_min_bounded"],
            [
                5,
            ],
            "desirability",
            id="desirability_max_min",
        ),
    ],
)
def test_is_non_dominated_logic(campaign, targets, idx_non_dominated, target_names):
    """Test is_non_dominated logic for different target and objective combinations."""
    # Construct data
    p1 = np.hstack(
        (
            np.linspace(0, 100, 3),
            np.linspace(100, 0, 3),
        )
    )
    p2 = np.hstack(
        (
            np.linspace(0, 100, 3),
            np.linspace(0, 100, 3),
        )
    )

    data_dict = {"param1": p1, "param2": p2}

    data_dict[target_names[0]] = p1 * 0.25 + p2 * 0.75
    if len(target_names) > 1:
        data_dict[target_names[1]] = p1 * 0.75 + p2 * 0.25

    measurements = pd.DataFrame(data_dict)

    campaign.add_measurements(measurements)
    non_dominated = campaign.is_non_dominated()

    assert set(np.where(non_dominated)[0].tolist()) == set(idx_non_dominated)


@pytest.mark.xfail(reason="Bug in Botorch 0.14.0; issue #2924", strict=False)
def test_botorch_is_non_dominated():
    """Test for bug in botorch 0.14.0 is_non_dominated().

    If first value is Nan, it wrongly returns True.
    False is expected in this case, but it returns True.
    See https://github.com/pytorch/botorch/issues/2924.
    """
    import torch
    from botorch.utils.multi_objective.pareto import is_non_dominated

    # First two entries are nans across the two arrays
    nans = torch.full((2, 2, 3), torch.nan)
    rands = torch.rand((2, 5, 3))
    # Nans at the beginning
    Y = torch.hstack([nans, rands])
    non_dominated = is_non_dominated(Y)
    # if the value is Nan, it should always return False
    assert not any(non_dominated[0, :2])  # first two idx are nans
    assert not any(non_dominated[:2, 0])  # producing the bug, should be False when nan
