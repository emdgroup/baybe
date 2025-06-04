"""Tests for transfer-learning."""

from copy import deepcopy

import pandas as pd
import pytest

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.interval import Interval

source = "B"
target = "A"

objective = SingleTargetObjective(target=NumericalTarget(name="y", mode="MAX"))
parameters = [
    NumericalContinuousParameter(name="x", bounds=Interval(0, 10)),
    TaskParameter(name="task", values=(target, source), active_values=(target,)),
]
searchspace = SearchSpace.from_product(parameters=parameters)

lookup = pd.DataFrame(
    {
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
        "task": [target] * 2 + [source] * 2,
    }
)


@pytest.mark.parametrize("observed_target_data", [True, False])
@pytest.mark.parametrize("observed_source_data", [True, False])
def test_recommendation(
    observed_target_data: bool,
    observed_source_data: bool,
):
    """Test a BO iteration with multi-task model using different parameters."""
    campaign = deepcopy(
        Campaign(
            searchspace=searchspace,
            objective=objective,
        )
    )

    lookup_sub = lookup.copy()
    # Add data
    if not observed_target_data:
        lookup_sub = lookup_sub.query("task!=@target")
    if not observed_source_data:
        lookup_sub = lookup_sub.query("task!=@source")
    if lookup_sub.shape[0] > 0:
        campaign.add_measurements(lookup_sub)

    _ = campaign.recommend(batch_size=1)
