"""Tests for the continuous cardinality constraint."""

import warnings
from collections.abc import Sequence
from itertools import combinations_with_replacement
from warnings import WarningMessage

import numpy as np
import pandas as pd
import pytest

from baybe.constraints.continuous import (
    ContinuousCardinalityConstraint,
    ContinuousLinearConstraint,
)
from baybe.constraints.utils import is_cardinality_fulfilled
from baybe.exceptions import MinimumCardinalityViolatedWarning
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders import BotorchRecommender
from baybe.recommenders.pure.bayesian.base import BayesianRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.targets import NumericalTarget


def _validate_cardinality_constrained_batch(
    batch: pd.DataFrame,
    subspace_continuous: SubspaceContinuous,
    batch_size: int,
    captured_warnings: Sequence[WarningMessage],
):
    """Validate that a cardinality-constrained batch fulfills the necessary conditions.

    Args:
        batch: The batch to validate.
        subspace_continuous: The continuous subspace from which to recommend the points.
        batch_size: The number of points to be recommended.
        captured_warnings: A list of captured warnings.
    """
    # Assert that the maximum cardinality constraint is fulfilled
    assert is_cardinality_fulfilled(batch, subspace_continuous, check_minimum=False)

    # Check whether the minimum cardinality constraint is fulfilled
    is_min_cardinality_fulfilled = is_cardinality_fulfilled(
        batch, subspace_continuous, check_maximum=False
    )

    # A warning must be raised when the minimum cardinality constraint is not fulfilled
    cardinality_warnings = [
        w
        for w in captured_warnings
        if issubclass(w.category, MinimumCardinalityViolatedWarning)
    ]
    assert is_min_cardinality_fulfilled != bool(cardinality_warnings)

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
    max_cardinalities = [
        c.max_cardinality for c in subspace_continuous.constraints_cardinality
    ]
    if len(unique_row := batch.drop_duplicates()) == 1:
        assert (unique_row.iloc[0] == 0.0).all() and all(
            max_cardinality == 0 for max_cardinality in max_cardinalities
        )


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

    subspace_continous = SubspaceContinuous(
        parameters=parameters, constraints_nonlin=constraints
    )

    with warnings.catch_warnings(record=True) as w:
        samples = subspace_continous.sample_uniform(BATCH_SIZE)

    # Assert that the constraint conditions hold
    _validate_cardinality_constrained_batch(samples, subspace_continous, BATCH_SIZE, w)


def test_polytope_sampling_with_cardinality_constraint():
    """Polytope sampling with cardinality constraints respects all involved
    constraints and produces distinct samples."""  # noqa

    N_PARAMETERS = 6
    MAX_CARDINALITY = 4
    MIN_CARDINALITY = 2
    BATCH_SIZE = 20
    TOLERANCE = 1e-3

    parameters = [
        NumericalContinuousParameter(name=f"x_{i + 1}", bounds=(0, 1))
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
    subspace_continous = SubspaceContinuous.from_product(parameters, constraints)

    with warnings.catch_warnings(record=True) as w:
        samples = subspace_continous.sample_uniform(BATCH_SIZE)

    # Assert that the constraint conditions hold
    _validate_cardinality_constrained_batch(samples, subspace_continous, BATCH_SIZE, w)

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


def test_min_cardinality_warning():
    """Providing candidates violating minimum cardinality constraint raises a
    warning.
    """  # noqa
    N_PARAMETERS = 2
    MIN_CARDINALITY = 2
    MAX_CARDINALITY = 2
    BATCH_SIZE = 20

    lower_bound = -0.5
    upper_bound = 0.5
    stepsize = 0.05
    parameters = [
        NumericalContinuousParameter(
            name=f"x_{i + 1}", bounds=(lower_bound, upper_bound)
        )
        for i in range(N_PARAMETERS)
    ]

    constraints = [
        ContinuousCardinalityConstraint(
            parameters=[p.name for p in parameters],
            max_cardinality=MAX_CARDINALITY,
            min_cardinality=MIN_CARDINALITY,
        ),
    ]

    searchspace = SearchSpace.from_product(parameters, constraints)
    objective = NumericalTarget("t").to_objective()

    # Create a scenario in which
    # - The optimum of the target function is at the origin
    # - The Botorch recommender is likely to provide candidates at the origin,
    # which violates the minimum cardinality constraint.
    def custom_target(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """A custom target function with maximum at the origin."""
        return -abs(x1) - abs(x2)

    def prepare_measurements() -> pd.DataFrame:
        """Prepare measurements."""
        x1 = np.arange(lower_bound, upper_bound + stepsize, stepsize)
        # Exclude 0 from the array
        X1, X2 = np.meshgrid(x1[abs(x1) > stepsize / 2], x1[abs(x1) > stepsize / 2])

        return pd.DataFrame(
            {
                "x_1": X1.flatten(),
                "x_2": X2.flatten(),
                "t": custom_target(X1.flatten(), X2.flatten()),
            }
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        BotorchRecommender().recommend(
            BATCH_SIZE, searchspace, objective, prepare_measurements()
        )
    assert any(
        issubclass(w.category, MinimumCardinalityViolatedWarning)
        for w in captured_warnings
    )


def test_empty_constraints_after_cardinality_constraint():
    """Constraints that have no more parameters left due to activated
    cardinality constraints do not cause crashes."""  # noqa

    N_PARAMETERS = 2

    parameters = [
        NumericalContinuousParameter(name=f"x_{i + 1}", bounds=(0, 1))
        for i in range(N_PARAMETERS)
    ]
    constraints = [
        ContinuousLinearConstraint(
            parameters=["x_1"],
            operator="=",
            coefficients=[1.0],
            rhs=0.3,
        ),
        ContinuousLinearConstraint(
            parameters=["x_2"],
            operator="<=",
            coefficients=[1.0],
            rhs=0.6,
        ),
        ContinuousCardinalityConstraint(
            parameters=["x_1", "x_2"],
            max_cardinality=1,
            min_cardinality=1,
        ),
    ]
    subspace = SubspaceContinuous.from_product(parameters, constraints)
    subspace.sample_uniform(1)


@pytest.mark.parametrize("recommender", [RandomRecommender(), BotorchRecommender()])
def test_cardinality_constraint(recommender):
    """Cardinality constraints are taken into account by the recommender."""
    MIN_CARDINALITY = 4
    MAX_CARDINALITY = 7
    BATCH_SIZE = 10

    parameters = [NumericalContinuousParameter(str(i), (0, 1)) for i in range(10)]
    constraints = [
        ContinuousCardinalityConstraint(
            [p.name for p in parameters], MIN_CARDINALITY, MAX_CARDINALITY
        )
    ]
    searchspace = SearchSpace.from_product(parameters, constraints)

    if isinstance(recommender, BayesianRecommender):
        objective = NumericalTarget("t").to_objective()
        measurements = pd.DataFrame(searchspace.continuous.sample_uniform(2))
        measurements["t"] = np.random.random(len(measurements))
    else:
        objective = None
        measurements = None

    with warnings.catch_warnings(record=True) as w:
        recommendation = recommender.recommend(
            BATCH_SIZE, searchspace, objective, measurements
        )

    # Assert that the constraint conditions hold
    _validate_cardinality_constrained_batch(
        recommendation, searchspace.continuous, BATCH_SIZE, w
    )
