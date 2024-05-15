"""Tests for the continuous cardinality constraint."""

import numpy as np

from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.bayesian import SequentialGreedyRecommender
from baybe.searchspace.core import SearchSpace
from baybe.targets.numerical import NumericalTarget


def test_continuous_cardinality_constraint():
    """
    Recommendations generated under a cardinality constraint have the expected number
    of nonzero elements.
    """  # noqa

    # Settings
    N_PARAMETERS = 5
    MAX_NONZERO = 3
    MIN_NONZERO = 1
    BATCH_SIZE = 5
    N_MEASUREMENTS = 10

    # Construct optimization problem and generate recommendations
    parameters = [
        NumericalContinuousParameter(name=f"x_{i}", bounds=(0, 1))
        for i in range(N_PARAMETERS)
    ]
    constraints = [
        # ContinuousCardinalityConstraint(
        #     min_nonzero=MIN_NONZERO, max_nonzero=MAX_NONZERO
        # )
    ]
    target = NumericalTarget(name="target", mode="MAX")
    searchspace = SearchSpace.from_product(parameters, constraints)
    measurements = searchspace.continuous.samples_random(N_MEASUREMENTS)
    measurements[target.name] = np.random.random(N_MEASUREMENTS)
    recommender = SequentialGreedyRecommender()
    rec = recommender.recommend(
        batch_size=BATCH_SIZE,
        searchspace=searchspace,
        objective=target.to_objective(),
        measurements=measurements,
    )

    # Assert that cardinality constraint is fulfilled
    n_nonzero = np.sum(~np.isclose(rec, 0.0), axis=1)
    assert np.all(n_nonzero >= MIN_NONZERO) and np.all(n_nonzero <= MAX_NONZERO)
