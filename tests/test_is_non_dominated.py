"""Test is non dominated functionality for objective."""

import pytest
from pytest import mark, param

from baybe.exceptions import IncompatibilityError
from baybe.objectives import (
    DesirabilityObjective,
    ParetoObjective,
    SingleTargetObjective,
)
from baybe.targets import NumericalTarget
from baybe.utils.dataframe import create_fake_input


@mark.parametrize(
    "objective",
    [
        param(
            DesirabilityObjective(
                (
                    NumericalTarget("t1", mode="MAX", bounds=(0, 100)),
                    NumericalTarget("t2", mode="MIN", bounds=(0, 100)),
                )
            ),
            id="desirability",
        ),
        param(
            ParetoObjective(
                (
                    NumericalTarget("t1", mode="MAX"),
                    NumericalTarget("t2", mode="MIN"),
                )
            ),
            id="pareto_max_min",
        ),
        param(
            ParetoObjective(
                (
                    NumericalTarget("t1", mode="MAX"),
                    NumericalTarget("t2", mode="MATCH", bounds=(0, 100)),
                )
            ),
            id="pareto_max_match",
        ),
    ],
)
def test_is_non_dominated_func_call(
    ongoing_campaign, objective, parameters, batch_size
):
    """Test function call for accepted objective types with multi output."""
    non_dominated_campaign_default = ongoing_campaign.is_non_dominated()
    assert len(ongoing_campaign.measurements) == len(non_dominated_campaign_default)

    fake_measures = create_fake_input(
        parameters, ongoing_campaign.objective.targets, batch_size
    )
    non_dominated_campaign = ongoing_campaign.is_non_dominated(fake_measures)
    assert len(fake_measures) == len(non_dominated_campaign)

    non_dominated_objective = objective.is_non_dominated(fake_measures)
    assert len(fake_measures) == len(non_dominated_objective)


@mark.parametrize(
    "objective",
    [
        param(
            SingleTargetObjective(
                NumericalTarget("t1", "MAX"),
            ),
            id="single",
        ),
        param(None),
    ],
)
def test_incompatibility(campaign, objective, fake_measurements):
    """Test for incompatibility for non multi output targets."""
    with pytest.raises(IncompatibilityError):
        campaign.is_non_dominated()
        campaign.is_non_dominated(fake_measurements)

    if objective is not None:
        with pytest.raises(IncompatibilityError):
            objective.is_non_dominated(fake_measurements)


@pytest.mark.xfail(reason="Bug in Botorch 0.14.0")
def test_botorch_is_non_dominated():
    """Test for bug in botorch 0.14.0 is_non_dominated().

    If first value is Nan, it wrongly returns True.
    False is expected in this case, but it returns True.
    See https://github.com/pytorch/botorch/issues/2924.
    """
    import torch
    from botorch.utils.multi_objective.pareto import is_non_dominated

    nans = torch.full((2, 2, 3), torch.nan)
    rands = torch.rand((2, 5, 3))
    # Nans at the beginning
    Y = torch.hstack([nans, rands])
    non_dominated = is_non_dominated(Y)
    # if the value is Nan, it should always return False
    assert all(non_dominated[0][:2])
