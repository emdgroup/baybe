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

objective = SingleTargetObjective(target=NumericalTarget(name="y", mode="MAX"))
parameters = [
    NumericalContinuousParameter(name="x", bounds=Interval(0, 10)),
    TaskParameter(name="task", values=("A", "B"), active_values=("A",)),
]
searchspace = SearchSpace.from_product(parameters=parameters)


def get_lookup(observed_test_data):
    lookup = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "task": ["A", "A", "B", "B"] if observed_test_data else ["B"] * 4,
        }
    )
    return lookup


@pytest.mark.parametrize("observed_test_data", [True, False])
def test_recommendation(
    observed_test_data: bool,
):
    """Test a BO iteration with multi-task model using different parameters."""
    lookup = get_lookup(observed_test_data=observed_test_data)
    campaign = deepcopy(
        Campaign(
            searchspace=searchspace,
            objective=objective,
        )
    )
    campaign.add_measurements(lookup)
    _ = campaign.recommend(batch_size=1)
