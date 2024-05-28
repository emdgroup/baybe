"""Integration tests."""

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace.core import SearchSpace
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import get_subclasses

nonpredictive_recommenders = [
    param(cls(allow_recommending_already_measured=True), id=cls.__name__)
    for cls in get_subclasses(NonPredictiveRecommender)
]

p1 = NumericalDiscreteParameter("p1", [1, 2])
t1 = NumericalTarget("t1", "MAX")
objective = t1.to_objective()
measurements = pd.DataFrame(
    {p1.name: p1.values, t1.name: np.random.random(len(p1.values))}
)


@pytest.fixture(name="searchspace")
def fixture_searchspace():
    return SearchSpace.from_product([p1])


@pytest.mark.parametrize("recommender", nonpredictive_recommenders)
def test_nonbayesian_recommender_with_measurements(recommender, searchspace):
    """Calling a non-Bayesian recommender with training data raises a warning."""
    with pytest.warns(
        UserWarning,
        match=(
            f"'{recommender.__class__.__name__}' does not utilize any training data"
        ),
    ):
        recommender.recommend(
            batch_size=1, searchspace=searchspace, measurements=measurements
        )


@pytest.mark.parametrize("recommender", nonpredictive_recommenders)
def test_nonbayesian_recommender_with_objective(recommender, searchspace):
    """Calling a non-Bayesian recommender with an objective raises a warning."""
    with pytest.warns(
        UserWarning,
        match=(f"'{recommender.__class__.__name__}' does not consider any objectives"),
    ):
        recommender.recommend(
            batch_size=1, searchspace=searchspace, objective=objective
        )
