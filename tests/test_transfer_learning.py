"""Tests for transfer-learning."""

from copy import deepcopy

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


@pytest.mark.parametrize("task_stratified_outtransform", [True, False])
def test_recommendation(task_stratified_outtransform):
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
            "task": ["A", "A", "B", "B"],
        }
    )
    campaign = deepcopy(
        Campaign(
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
    )
    campaign.add_measurements(lookup)
    _ = campaign.recommend(batch_size=1)


# TODO once recommendation without data for active task is fixed
#  add test where no active task data is added at the start
