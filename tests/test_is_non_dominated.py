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
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.targets.numerical import NumericalTarget


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


# Target definitions
t_max = NumericalTarget("t_max")
t_min = NumericalTarget("t_min", minimize=True)
t_max_b = NumericalTarget.normalized_sigmoid(
    "t_max_bounded", anchors=[(0, 0.05), (100, 0.95)]
)
t_min_b = NumericalTarget.normalized_sigmoid(
    "t_min_bounded", anchors=[(0, 0.95), (100, 0.05)]
)
t_bell = NumericalTarget.match_bell(name="t_bell", match_value=50, sigma=5)
t_triang = NumericalTarget.match_triangular("t_triang", cutoffs=(0, 100))


@pytest.mark.parametrize(
    ("targets", "idx_non_dominated", "objective_cls"),
    [
        param([t_min_b, t_min], [0], ParetoObjective, id="pareto_min_min"),
        param([t_max, t_min], [0, 2, 5], ParetoObjective, id="pareto_max_min"),
        param([t_min, t_max], [0, 2, 3], ParetoObjective, id="pareto_min_max"),
        param([t_max_b, t_max], [2], ParetoObjective, id="pareto_max_max"),
        param([t_bell, t_triang], [1, 4], ParetoObjective, id="pareto_match"),
        param([t_min], [0], SingleTargetObjective, id="single_min"),
        param([t_max], [2], SingleTargetObjective, id="single_max"),
        param([t_bell], [1, 4], SingleTargetObjective, id="single_bell"),
        param([t_min_b, t_max_b], [3], DesirabilityObjective, id="des_min_max"),
        param([t_max_b, t_min_b], [5], DesirabilityObjective, id="des_max_min"),
    ],
)
def test_identify_non_dominated_configurations_logic(objective, idx_non_dominated):
    """The correct set of non-dominated configurations is identified."""
    df = pd.DataFrame(
        {
            "p1": [0, 50, 100, 100, 50, 0],
            "p2": [0, 50, 100, 0, 50, 100],
        }
    )
    target_names = [t.name for t in objective.targets]
    df[target_names[0]] = df["p1"] * 0.25 + df["p2"] * 0.75
    if len(objective.targets) > 1:
        df[target_names[1]] = df["p1"] * 0.75 + df["p2"] * 0.25

    non_dominated = objective.identify_non_dominated_configurations(df)
    assert set(np.where(non_dominated)[0]) == set(idx_non_dominated)
