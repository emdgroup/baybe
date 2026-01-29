"""Test identification of non-dominated configurations."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.campaign import Campaign
from baybe.exceptions import IncompatibilityError, NothingToComputeError
from baybe.parameters import NumericalDiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter


# TODO: Remove once batch size fixture has been deactivated globally
@pytest.fixture
def batch_size():
    return 2


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
def test_consistency(
    ongoing_campaign, fake_measurements, consider_campaign_measurements
):
    """Identifying non-dominated configurations yields consistent results, regardless
    of which entry point is used (campaign w/o measurements or objective)."""  # noqa
    # With campaign measurements
    non_dominated_default = ongoing_campaign.identify_non_dominated_configurations(
        consider_campaign_measurements=consider_campaign_measurements
    )
    non_dominated_default_as_arg = (
        ongoing_campaign.identify_non_dominated_configurations(
            ongoing_campaign.measurements,
            consider_campaign_measurements=not consider_campaign_measurements,
        )
    )
    assert non_dominated_default.equals(non_dominated_default_as_arg)
    assert len(ongoing_campaign.measurements) == len(non_dominated_default)

    # With external configurations
    non_dominated_campaign = ongoing_campaign.identify_non_dominated_configurations(
        fake_measurements, consider_campaign_measurements=consider_campaign_measurements
    )
    non_dominated_objective = (
        ongoing_campaign.objective.identify_non_dominated_configurations(
            fake_measurements
        )
    )
    assert len(fake_measurements) == len(non_dominated_campaign)
    assert len(fake_measurements) == len(non_dominated_objective)

    # Equality is only guaranteed if campaign measurements are not considered
    if not consider_campaign_measurements:
        assert non_dominated_campaign.equals(non_dominated_objective)


def test_missing_objective():
    """Identification of non-dominated configurations without object is rejected."""
    campaign = Campaign(searchspace=NumericalContinuousParameter("p", (0, 1)))
    with pytest.raises(IncompatibilityError, match="no 'Objective' is defined"):
        campaign.identify_non_dominated_configurations()


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
def test_logic_consider_campaign_measurements(campaign, fake_measurements):
    """Test that exceptions are raised for invalid input combinations."""
    # Test flag when campaign has no measurements
    with pytest.raises(NothingToComputeError):
        campaign.identify_non_dominated_configurations(
            consider_campaign_measurements=True
        )

    with pytest.raises(NothingToComputeError):
        campaign.identify_non_dominated_configurations(
            consider_campaign_measurements=False
        )

    with pytest.warns(UserWarning):
        campaign.identify_non_dominated_configurations(
            fake_measurements, consider_campaign_measurements=True
        )

    campaign.identify_non_dominated_configurations(
        fake_measurements, consider_campaign_measurements=False
    )

    # Test flag when campaign has measurements
    campaign.add_measurements(fake_measurements)

    campaign.identify_non_dominated_configurations(consider_campaign_measurements=True)
    campaign.identify_non_dominated_configurations(consider_campaign_measurements=False)
    campaign.identify_non_dominated_configurations(
        fake_measurements, consider_campaign_measurements=True
    )
    campaign.identify_non_dominated_configurations(
        fake_measurements, consider_campaign_measurements=False
    )


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
def test_identify_non_dominated_configurations_logic(
    campaign, idx_non_dominated, target_names
):
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
    non_dominated = campaign.identify_non_dominated_configurations()

    assert set(np.where(non_dominated)[0].tolist()) == set(idx_non_dominated)
