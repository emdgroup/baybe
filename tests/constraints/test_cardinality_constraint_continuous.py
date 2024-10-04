"""Tests for the continuous cardinality constraint."""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytest

from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.parameters import NumericalContinuousParameter
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace.core import SearchSpace, SubspaceContinuous


def _validate_samples(
    samples: pd.DataFrame, max_cardinality: int, min_cardinality: int, batch_size: int
):
    """Validate if cardinality-constrained samples fulfill the necessary conditions.

    Conditions to check:
    * Cardinality is in requested range
    * The batch contains right number of samples
    * The samples are free of duplicates (except all zeros)

    Args:
        samples: Samples to check
        max_cardinality: Maximum allowed cardinality
        min_cardinality: Minimum required cardinality
        batch_size: Requested batch size
    """
    # Assert that cardinality constraint is fulfilled
    n_nonzero = np.sum(~np.isclose(samples, 0.0), axis=1)
    assert np.all(n_nonzero >= min_cardinality) and np.all(n_nonzero <= max_cardinality)

    # Assert that we obtain as many samples as requested
    assert len(samples) == batch_size

    # If there are duplicates, they must all come from the case cardinality = 0
    assert np.all(samples[samples.duplicated()] == 0.0)


# Combinations of cardinalities to be tested
cardinality_bounds_combinations = sorted(combinations_with_replacement(range(0, 10), 2))


@pytest.mark.parametrize(
    "cardinality_bounds",
    cardinality_bounds_combinations,
    ids=[str(x) for x in cardinality_bounds_combinations],
)
def test_sampling_cardinality_constraint(cardinality_bounds: tuple[int, int]):
    """Sampling on unit-cube with cardinality constraints respects all constraints and
    produces distinct samples."""  # noqa

    N_PARAMETERS = 10
    BATCH_SIZE = 10
    min_cardinality, max_cardinality = cardinality_bounds

    parameters = tuple(
        NumericalContinuousParameter(name=f"x_{i}", bounds=(0, 1))
        for i in range(N_PARAMETERS)
    )

    constraints = (
        ContinuousCardinalityConstraint(
            parameters=[f"x_{i}" for i in range(N_PARAMETERS)],
            min_cardinality=min_cardinality,
            max_cardinality=max_cardinality,
        ),
    )

    subspace = SubspaceContinuous(parameters=parameters, constraints_nonlin=constraints)
    samples = subspace.sample_uniform(BATCH_SIZE)

    # Assert that conditions listed in_validate_samples() are fulfilled
    _validate_samples(samples, max_cardinality, min_cardinality, BATCH_SIZE)


def test_polytope_sampling_with_cardinality_constraint():
    """Polytope sampling with cardinality constraints respects all involved
    constraints and produces distinct samples."""  # noqa

    N_PARAMETERS = 6
    MAX_CARDINALITY = 4
    MIN_CARDINALITY = 2
    BATCH_SIZE = 20
    TOLERANCE = 1e-3

    parameters = [
        NumericalContinuousParameter(name=f"x_{i+1}", bounds=(0, 1))
        for i in range(N_PARAMETERS)
    ]
    params_equality = ["x_1", "x_2", "x_3", "x_4"]
    coeffs_equality = [0.9, 0.6, 2.8, 6.1]
    rhs_equality = 4.2
    params_inequality = ["x_1", "x_2", "x_5", "x_6"]
    coeffs_inequality = [4.7, 1.4, 4.6, 8.6]
    rhs_inequality = 1.3
    params_cardinality = ["x_1", "x_2", "x_3", "x_5"]
    constraints = [
        ContinuousLinearConstraint(
            parameters=params_equality,
            operator="=",
            coefficients=coeffs_equality,
            rhs=rhs_equality,
        ),
        ContinuousLinearConstraint(
            parameters=params_inequality,
            operator=">=",
            coefficients=coeffs_inequality,
            rhs=rhs_equality,
        ),
        ContinuousCardinalityConstraint(
            parameters=params_cardinality,
            max_cardinality=MAX_CARDINALITY,
            min_cardinality=MIN_CARDINALITY,
        ),
    ]
    searchspace = SearchSpace.from_product(parameters, constraints)

    samples = searchspace.continuous.sample_uniform(BATCH_SIZE)

    # Assert that conditions listed in_validate_samples() are fulfilled
    _validate_samples(
        samples[params_cardinality], MAX_CARDINALITY, MIN_CARDINALITY, BATCH_SIZE
    )

    # Assert that linear equality constraint is fulfilled
    assert np.allclose(
        np.sum(samples[params_equality] * coeffs_equality, axis=1),
        rhs_equality,
        atol=TOLERANCE,
    )

    # Assert that linear non-equality constraint is fulfilled
    assert (
        np.sum(samples[params_inequality] * coeffs_inequality, axis=1)
        .ge(rhs_inequality - TOLERANCE)
        .all()
    )


@pytest.mark.parametrize(
    "parameter_names", [["Conti_finite1", "Conti_finite2", "Conti_finite3"]]
)
@pytest.mark.parametrize("constraint_names", [["ContiConstraint_5"]])
@pytest.mark.parametrize("batch_size", [5], ids=["b5"])
def test_random_recommender_with_cardinality_constraint(
    parameters: list[NumericalContinuousParameter],
    constraints: list[ContinuousCardinalityConstraint],
    batch_size: int,
):
    """Recommendations generated by a `RandomRecommender` under a cardinality constraint
    have the expected number of nonzero elements."""  # noqa

    searchspace = SearchSpace.from_product(
        parameters=parameters, constraints=constraints
    )
    recommender = RandomRecommender()
    recommendations = recommender.recommend(
        searchspace=searchspace,
        batch_size=batch_size,
    )

    # Assert that conditions listed in_validate_samples() are fulfilled
    _validate_samples(
        recommendations, max_cardinality=2, min_cardinality=1, batch_size=batch_size
    )
