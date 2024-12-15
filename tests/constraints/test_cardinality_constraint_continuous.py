"""Tests for the continuous cardinality constraint."""

import warnings
from itertools import combinations_with_replacement
from warnings import WarningMessage

import numpy as np
import pandas as pd
import pytest

from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace, SubspaceContinuous
from baybe.utils.cardinality_constraints import count_near_zeros


def _validate_cardinality_constrained_batch(
    batch: pd.DataFrame,
    min_cardinality: int,
    max_cardinality: int,
    batch_size: int,
    parameters: tuple[NumericalContinuousParameter],
    captured_warnings: list[WarningMessage | None],
):
    """Validate that a cardinality-constrained batch fulfills the necessary conditions.

    Args:
        batch: Batch to validate.
        min_cardinality: Minimum required cardinality.
        max_cardinality: Maximum allowed cardinality.
        batch_size: Requested batch size.
        parameters: A list of parameters for which recommendations are provided.
        captured_warnings: A list of captured warnings.
    """
    # Assert that the maximum cardinality constraint is fulfilled
    n_nonzeros = len(parameters) - count_near_zeros(parameters, batch)
    assert np.all(n_nonzeros <= max_cardinality)

    # Check whether the minimum cardinality constraint is fulfilled
    is_min_cardinality_fulfilled = np.all(n_nonzeros >= min_cardinality)

    # A warning must be raised when the minimum cardinality constraint is not fulfilled
    if not is_min_cardinality_fulfilled:
        w_message = "Minimum cardinality constraints are not guaranteed."
        assert any(str(w.message) == w_message for w in captured_warnings)

    # Assert that we obtain as many samples as requested
    assert batch.shape[0] == batch_size

    # Sanity check: If all recommendations in the batch are identical, something is
    # fishy – unless the cardinality is 0, in which case the entire batch must contain
    # zeros. Technically, the probability of getting such a degenerate batch
    # is not zero, hence this is not a strict requirement. However, in earlier BoTorch
    # versions, this simply happened due to a bug in their sampler:
    # https://github.com/pytorch/botorch/issues/2351
    # We thus include this check as a safety net for catching regressions. If it
    # turns out the check fails because we observe degenerate batches as actual
    # recommendations, we need to invent something smarter.
    if len(unique_row := batch.drop_duplicates()) == 1:
        assert (unique_row.iloc[0] == 0.0).all() and (max_cardinality == 0)


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

    with warnings.catch_warnings(record=True) as w:
        samples = subspace.sample_uniform(BATCH_SIZE)

    # Assert that the constraint conditions hold
    _validate_cardinality_constrained_batch(
        samples, min_cardinality, max_cardinality, BATCH_SIZE, parameters, w
    )


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

    with warnings.catch_warnings(record=True) as w:
        samples = searchspace.continuous.sample_uniform(BATCH_SIZE)

    # Assert that the constraint conditions hold
    _validate_cardinality_constrained_batch(
        samples[params_cardinality],
        MIN_CARDINALITY,
        MAX_CARDINALITY,
        BATCH_SIZE,
        tuple(p for p in parameters if p.name in params_cardinality),
        w,
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
