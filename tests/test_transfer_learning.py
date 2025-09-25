"""Tests for transfer-learning."""

import pandas as pd
import pytest

from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget
from baybe.utils.interval import Interval


@pytest.mark.parametrize("observed_target_data", [True, False])
@pytest.mark.parametrize("observed_source_data", [True, False])
def test_recommendation(
    observed_target_data: bool,
    observed_source_data: bool,
):
    """Transfer learning recommendation works in different training data settings.

    Regardless of whether source/target tasks are missing/present in the training data.
    """
    # Setup test data
    source = "B"
    target = "A"
    objective = NumericalTarget(name="y").to_objective()
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
    if not observed_target_data:
        lookup = lookup.query("task!=@target")
    if not observed_source_data:
        lookup = lookup.query("task!=@source")

    # Run test
    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    if lookup.shape[0] > 0:
        campaign.add_measurements(lookup)

    campaign.recommend(batch_size=1)


def test_multiple_active_tasks():
    """Transfer learning recommendation works with multiple active task values."""
    # Setup test data
    source = "B"
    target = "A"
    objective = NumericalTarget(name="y").to_objective()
    parameters = [
        NumericalContinuousParameter(name="x", bounds=Interval(0, 10)),
        TaskParameter(
            name="task", values=(target, source), active_values=(target, source)
        ),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    lookup = pd.DataFrame(
        {
            # This dataframe was set so that both tasks are recommended
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 1.0, 2.0],
            "task": [target] * 2 + [source] * 2,
        }
    )

    # Run test
    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )

    campaign.add_measurements(lookup)

    recommended = campaign.recommend(batch_size=10)
    assert set(recommended["task"]) == {target, source}
