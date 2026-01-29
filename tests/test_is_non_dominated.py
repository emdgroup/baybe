"""Test identification of non-dominated configurations."""

import contextlib

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.campaign import Campaign
from baybe.exceptions import IncompatibilityError, NothingToComputeError
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter


# TODO: Remove once batch size fixture has been deactivated globally
@pytest.fixture
def batch_size():
    return 2


@pytest.mark.parametrize(
    ("objective_cls", "target_names"),
    [
        param(ParetoObjective, ["Target_max", "Target_min"], id="pareto_max_min"),
        param(ParetoObjective, ["Target_min", "Target_max"], id="pareto_min_max"),
        param(
            ParetoObjective,
            ["Target_min", "Target_max", "Target_match_triangular"],
            id="pareto",
        ),
        param(SingleTargetObjective, ["Target_max"], id="single"),
        param(
            DesirabilityObjective,
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


@pytest.mark.parametrize("add_measurements", [True, False], ids=["add", "no_add"])
@pytest.mark.parametrize("external_configurations", [True, False], ids=["ext", "int"])
@pytest.mark.parametrize("consider", [True, False], ids=["consider", "no_consider"])
def test_invalid_argument_configurations(
    campaign,
    fake_measurements,
    add_measurements,
    external_configurations,
    consider,
):
    """For invalid argument configurations, the appropriate error/warning is raised."""
    if add_measurements:
        campaign.add_measurements(fake_measurements)

    if not add_measurements and not external_configurations:
        # No data available -> should raise
        context = pytest.raises(NothingToComputeError)
    elif not add_measurements and external_configurations and consider:
        # Only external data, trying to consider campaign -> should warn
        context = pytest.warns(UserWarning)
    else:
        # Has campaign measurements or not considering them -> fine
        context = contextlib.nullcontext()

    configurations = fake_measurements if external_configurations else None
    with context:
        campaign.identify_non_dominated_configurations(
            configurations, consider_campaign_measurements=consider
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
    ("target_names", "idx_non_dominated", "objective_cls"),
    [
        param(
            ["Target_min_bounded", "Target_min"],
            [0],
            ParetoObjective,
            id="pareto_min_min",
        ),
        param(
            ["Target_max", "Target_min"],
            [0, 2, 5],
            ParetoObjective,
            id="pareto_max_min",
        ),
        param(
            ["Target_min", "Target_max"],
            [0, 2, 3],
            ParetoObjective,
            id="pareto_min_max",
        ),
        param(
            ["Target_max_bounded", "Target_max"],
            [2],
            ParetoObjective,
            id="pareto_max_max",
        ),
        param(
            ["Target_match_bell", "Target_match_triangular"],
            [1, 4],
            ParetoObjective,
            id="pareto_match_bell_trnglr",
        ),
        param(["Target_min"], [0], SingleTargetObjective, id="single_min"),
        param(["Target_max"], [2], SingleTargetObjective, id="single_max"),
        param(["Target_match_bell"], [1, 4], SingleTargetObjective, id="single_bell"),
        param(
            ["Target_min_bounded", "Target_max_bounded"],
            [
                3,
            ],
            DesirabilityObjective,
            id="desirability_min_max",
        ),
        param(
            ["Target_max_bounded", "Target_min_bounded"],
            [
                5,
            ],
            DesirabilityObjective,
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
