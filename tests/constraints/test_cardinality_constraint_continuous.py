"""Tests for the continuous cardinality constraint."""

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytest

from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace, SubspaceContinuous


def _validate_samples(
    samples: pd.DataFrame,
    max_cardinality: int,
    min_cardinality: int,
    batch_size: int,
    threshold: float = 0.0,
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
        threshold: Threshold for checking whether a value is treated as zero.
    """
    # Assert that cardinality constraint is fulfilled
    if threshold == 0.0:
        # When threshold is zero, abs(value) > threshold is treated as non-zero.
        n_nonzero = len(samples.columns) - np.sum(samples.abs().le(threshold), axis=1)
    else:
        # When threshold is non-zero, abs(value) >= threshold is treated as non-zero.
        n_nonzero = np.sum(samples.abs().ge(threshold), axis=1)

    assert np.all(n_nonzero >= min_cardinality) and np.all(n_nonzero <= max_cardinality)

    # Assert that we obtain as many samples as requested
    assert samples.shape[0] == batch_size

    # If all rows are duplicates of the first row, they must all come from the case
    # cardinality = 0 (all rows are zeros)
    all_zero_rows = (samples == 0).all(axis=1)
    duplicated_rows = samples.duplicated()
    assert ~np.all(duplicated_rows[1:]) | np.all(all_zero_rows)


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
