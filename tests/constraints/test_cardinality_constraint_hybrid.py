"""Tests for cardinality constraints in hybrid search spaces.

Regression tests for https://github.com/emdgroup/baybe/issues/567.
"""

import numpy as np
import pandas as pd
import pytest

from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.constraints.discrete import DiscreteCardinalityConstraint
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.recommenders import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget

BATCH_SIZE = 5
MAX_CARDINALITY = 1

# --- Shared helpers -----------------------------------------------------------

_discrete_params = [
    NumericalDiscreteParameter(f"d{i}", values=(0.0, 0.5, 1.0)) for i in range(3)
]
_continuous_params = [
    NumericalContinuousParameter(f"c{i}", bounds=(0, 1)) for i in range(2)
]

# --- Test configurations ------------------------------------------------------


def _config_continuous_only():
    """Hybrid space with a continuous cardinality constraint."""
    parameters = [
        CategoricalParameter("cat", values=("A", "B", "C")),
        *_continuous_params,
    ]
    constraints = [
        ContinuousCardinalityConstraint(
            parameters=[p.name for p in _continuous_params],
            max_cardinality=MAX_CARDINALITY,
        )
    ]
    return parameters, constraints


def _config_discrete_only():
    """Hybrid space with a discrete cardinality constraint."""
    parameters = [
        *_discrete_params,
        NumericalContinuousParameter("c_extra", bounds=(0, 1)),
    ]
    constraints = [
        DiscreteCardinalityConstraint(
            parameters=[p.name for p in _discrete_params],
            max_cardinality=MAX_CARDINALITY,
        )
    ]
    return parameters, constraints


def _config_both():
    """Hybrid space with both a discrete and a continuous cardinality constraint."""
    parameters = [*_discrete_params, *_continuous_params]
    constraints = [
        DiscreteCardinalityConstraint(
            parameters=[p.name for p in _discrete_params],
            max_cardinality=MAX_CARDINALITY,
        ),
        ContinuousCardinalityConstraint(
            parameters=[p.name for p in _continuous_params],
            max_cardinality=MAX_CARDINALITY,
        ),
    ]
    return parameters, constraints


# --- Parametrized test --------------------------------------------------------


@pytest.mark.parametrize(
    ("config_fn", "discrete_names", "continuous_names"),
    [
        pytest.param(
            _config_continuous_only,
            None,
            [p.name for p in _continuous_params],
            id="continuous_only",
        ),
        pytest.param(
            _config_discrete_only,
            [p.name for p in _discrete_params],
            None,
            id="discrete_only",
        ),
        pytest.param(
            _config_both,
            [p.name for p in _discrete_params],
            [p.name for p in _continuous_params],
            id="both",
        ),
    ],
)
def test_cardinality_constraint_hybrid(config_fn, discrete_names, continuous_names):
    """Cardinality constraints are respected in hybrid search spaces."""
    parameters, constraints = config_fn()
    searchspace = SearchSpace.from_product(parameters, constraints)
    objective = NumericalTarget("t").to_objective()

    # Build measurements covering all parameter columns
    all_param_names = [p.name for p in parameters]
    rng = np.random.default_rng(42)
    measurements = pd.DataFrame({name: rng.random(3) for name in all_param_names})
    # Snap categorical values to valid entries
    for p in parameters:
        if isinstance(p, CategoricalParameter):
            measurements[p.name] = rng.choice(p.values, size=len(measurements))
        elif isinstance(p, NumericalDiscreteParameter):
            measurements[p.name] = rng.choice(list(p.values), size=len(measurements))
    measurements["t"] = rng.random(len(measurements))

    rec = BotorchRecommender().recommend(
        BATCH_SIZE, searchspace, objective, measurements
    )

    # Validate continuous cardinality
    if continuous_names is not None:
        n_nonzero = (rec[continuous_names].abs() > 1e-3).sum(axis=1)
        assert (n_nonzero <= MAX_CARDINALITY).all(), (
            f"Continuous cardinality constraint violated: {n_nonzero.tolist()}"
        )

    # Validate discrete cardinality
    if discrete_names is not None:
        n_nonzero = (rec[discrete_names] != 0.0).sum(axis=1)
        assert (n_nonzero <= MAX_CARDINALITY).all(), (
            f"Discrete cardinality constraint violated: {n_nonzero.tolist()}"
        )
