"""Tests for transfer learning."""

from typing import Literal

import pandas as pd
import pytest

from baybe import Campaign
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget


@pytest.fixture
def campaign(
    training_data: Literal["source", "target", "both"],
    active_tasks: Literal["target_only", "both"],
) -> Campaign:
    """A transfer-learning campaign with various active tasks and training data."""
    assert training_data in ["source", "target", "both"]
    assert active_tasks in ["target_only", "both"]

    source = "B"
    target = "A"
    parameters = [
        NumericalContinuousParameter("x", (0, 5)),
        TaskParameter(
            "task",
            values=(target, source),
            active_values=(
                (target,) if active_tasks == "target_only" else (target, source)
            ),
        ),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    objective = NumericalTarget(name="y").to_objective()
    recommender = BotorchRecommender()
    lookup = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "task": [target] * 2 + [source] * 2,
        }
    )

    if training_data == "source":
        lookup = lookup[lookup["task"] == source]
    elif training_data == "target":
        lookup = lookup[lookup["task"] == target]

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=recommender,
    )
    campaign.add_measurements(lookup)

    return campaign


@pytest.mark.parametrize("active_tasks", ["target_only", "both"])
@pytest.mark.parametrize("training_data", ["source", "target", "both"])
def test_recommendation(campaign: Campaign):
    """Transfer learning recommendation works regardless of which task are
    present in the training data and which tasks are active.
    """  # noqa: D205
    campaign.recommend(1)
