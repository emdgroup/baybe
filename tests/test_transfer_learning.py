"""Tests for transfer-learning."""

import pandas as pd
import pytest

from baybe import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import NumericalContinuousParameter, TaskParameter
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe.targets import NumericalTarget
from baybe.utils.interval import Interval


@pytest.mark.parametrize(
    "task_stratified_outtransform,observed_test_data",
    ([False, True], [False, False]),
)
def test_recommendation(task_stratified_outtransform: bool, observed_test_data: bool):
    """Test a BO iteration with multi-task model."""
    objective = SingleTargetObjective(target=NumericalTarget(name="y", mode="MAX"))
    parameters = [
        NumericalContinuousParameter(name="x", bounds=Interval(0, 10)),
        TaskParameter(name="task", values=("A", "B"), active_values=("A",)),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    lookup = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "task": ["A", "A", "B", "B"] if observed_test_data else ["B"] * 4,
        }
    )

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            recommender=BotorchRecommender(
                surrogate_model=GaussianProcessSurrogate(
                    task_stratified_outtransform=task_stratified_outtransform
                )
            ),
            initial_recommender=RandomRecommender(),
        ),
    )
    campaign.add_measurements(lookup)
    _ = campaign.recommend(batch_size=1)


def test_recommendation_without_active_task_data():
    """Test when no data is available for the active task initially."""
    objective = SingleTargetObjective(target=NumericalTarget(name="y", mode="MAX"))
    parameters = [
        NumericalContinuousParameter(name="x", bounds=Interval(0, 10)),
        TaskParameter(name="task", values=("A", "B"), active_values=("A",)),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)

    # Only provide data for the inactive task
    lookup = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "task": ["B", "B", "B", "B"],  # Only task B data
        }
    )

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            recommender=BotorchRecommender(
                surrogate_model=GaussianProcessSurrogate(
                    task_stratified_outtransform=False
                    # Stratified transform won't work without data for both tasks
                )
            ),
            initial_recommender=RandomRecommender(),
        ),
    )

    campaign.add_measurements(lookup)
    # Should be able to recommend even without active task data
    recommendation = campaign.recommend(batch_size=1)
    # Verify that the recommendation is for the active task
    assert recommendation["task"].iloc[0] == "A"


@pytest.mark.skip(
    reason="Stratified standardization with missing tasks not supported yet"
)
def test_recommendation_with_stratified_transform():
    """Test the behavior of stratified standardization with task parameter.

    Note: This test is skipped until the proper fix for stratified transform
    is implemented in BoTorch or BayBE.
    """
    objective = SingleTargetObjective(target=NumericalTarget(name="y", mode="MAX"))
    parameters = [
        NumericalContinuousParameter(name="x", bounds=Interval(0, 10)),
        TaskParameter(name="task", values=("A", "B"), active_values=("A",)),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)

    # Create data with both task values
    lookup = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [1.0, 2.0, 3.0, 4.0],
            "task": ["A", "A", "B", "B"],
        }
    )

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=TwoPhaseMetaRecommender(
            recommender=BotorchRecommender(
                surrogate_model=GaussianProcessSurrogate(
                    task_stratified_outtransform=True  # Enable stratified transform
                )
            ),
            initial_recommender=RandomRecommender(),
        ),
    )

    campaign.add_measurements(lookup)
    recommendation = campaign.recommend(batch_size=1)
    assert recommendation["task"].iloc[0] == "A"
