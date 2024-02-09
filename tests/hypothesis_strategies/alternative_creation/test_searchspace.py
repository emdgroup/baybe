"""Test alternative ways of creation not considered in the strategies."""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from pytest import param

from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.categorical import TaskParameter
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete
from tests.hypothesis_strategies.parameters import numerical_discrete_parameter

# Discrete inputs for testing
s_x = pd.Series([1, 2, 3], name="x")
p_x = NumericalDiscreteParameter(name="x", values=[1, 2, 3])
p_x_over = NumericalDiscreteParameter(name="x", values=[1, 2, 3, 4])
p_x_under = NumericalDiscreteParameter(name="x", values=[1, 2])
s_y = pd.Series(["a", "b", "c"], name="y")
p_y = CategoricalParameter(name="y", values=["a", "b", "c"])
df_discrete = pd.concat([s_x, s_y], axis=1)

# Continuous inputs for testing
s_a = pd.Series([1, 2, 3], name="a")
p_a = NumericalContinuousParameter(name="a", bounds=(1, 3))
p_a_over = NumericalContinuousParameter(name="a", bounds=(1, 4))
p_a_under = NumericalContinuousParameter(name="a", bounds=(1, 2))
s_b = pd.Series([10, 15, 20], name="b")
p_b = NumericalContinuousParameter(name="b", bounds=(10, 20))
df_continuous = pd.concat([s_a, s_b], axis=1)

# Mixed inputs for testing
df = pd.concat([df_discrete, df_continuous], axis=1)


@pytest.mark.parametrize(
    ("df", "parameters", "expected"),
    [
        param(s_x.to_frame(), [p_x], [p_x], id="num-match"),
        param(s_x.to_frame(), [p_x_over], [p_x_over], id="num_overparametrized"),
        param(s_x.to_frame(), [p_x_under], ValueError, id="num_underparametrized"),
        param(df_discrete, [p_x, p_x], ValueError, id="duplicate-name"),
        param(s_x.to_frame(), [p_x, p_y], ValueError, id="no_match"),
        param(s_y.to_frame(), [p_y], [p_y], id="cat-match"),
        param(df_discrete, [p_x, p_y], [p_x, p_y], id="both-match"),
        param(df_discrete, [p_x], [p_x, p_y], id="one-unspecified"),
    ],
)
def test_discrete_space_creation_from_dataframe(df, parameters, expected):
    """Parameters are automatically inferred and exceptions are triggered."""
    if isinstance(expected, list):
        subspace = SubspaceDiscrete.from_dataframe(df, parameters)
        actual = subspace.parameters
        assert actual == expected, (actual, expected)
    else:
        with pytest.raises(expected):
            SubspaceDiscrete.from_dataframe(df, parameters)


@pytest.mark.parametrize(
    ("df", "parameters", "expected"),
    [
        param(s_a.to_frame(), [p_a], [p_a], id="match"),
        param(s_a.to_frame(), [p_a_over], [p_a_over], id="overparametrized"),
        param(s_a.to_frame(), [p_a_under], ValueError, id="underparametrized"),
        param(df_continuous, [p_a, p_a], ValueError, id="duplicate-name"),
        param(s_a.to_frame(), [p_a, p_b], ValueError, id="no_match"),
        param(df_continuous, [p_a], [p_a, p_b], id="one-unspecified"),
    ],
)
def test_continuous_space_creation_from_dataframe(df, parameters, expected):
    """Parameters are automatically inferred and exceptions are triggered."""
    if isinstance(expected, list):
        subspace = SubspaceContinuous.from_dataframe(df, parameters)
        actual = subspace.parameters
        assert actual == expected, (actual, expected)
    else:
        with pytest.raises(expected):
            SubspaceContinuous.from_dataframe(df, parameters)


@pytest.mark.parametrize(
    ("df", "parameters", "expected"),
    [
        param(df, [p_x, p_y, p_a, p_b], [p_x, p_y, p_a, p_b], id="match"),
        param(df, [p_x, p_x, p_x, p_x], ValueError, id="duplicates"),
        param(df, [p_x], ValueError, id="missing"),
    ],
)
def test_searchspace_creation_from_dataframe(df, parameters, expected):
    """Parameters are automatically inferred and exceptions are triggered."""
    if isinstance(expected, list):
        subspace = SearchSpace.from_dataframe(df, parameters)
        actual = subspace.parameters
        assert actual == expected, (actual, expected)
    else:
        with pytest.raises(expected):
            SearchSpace.from_dataframe(df, parameters)


@given(
    parameters=st.lists(
        numerical_discrete_parameter(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x.name,
    )
)
def test_discrete_space_creation_from_simplex_inner(parameters):
    """Candidates from a simplex space satisfy the simplex constraint."""
    max_possible = sum(max(p.values) for p in parameters)
    min_possible = sum(min(p.values) for p in parameters)
    total = (max_possible + min_possible) / 2
    tolerance = 1e-6
    subspace = SubspaceDiscrete.from_simplex(
        parameters, total=total, boundary_only=False, tolerance=tolerance
    )
    assert (subspace.exp_rep.sum(axis=1) <= total + tolerance).all()


def test_discrete_space_creation_from_simplex_boundary():
    """Candidates from a simplex boundary space satisfy the boundary constraint."""
    total = 1.0
    tolerance = 1e-6
    parameters = [
        NumericalDiscreteParameter(name=str(i), values=np.linspace(0.0, 1.0, 5))
        for i in range(5)
    ]
    subspace = SubspaceDiscrete.from_simplex(
        parameters, total=total, boundary_only=True, tolerance=tolerance
    )
    assert np.allclose(subspace.exp_rep.sum(axis=1), total, atol=tolerance)


def test_discrete_space_creation_from_simplex_mixed():
    """Additional non-simplex parameters enter in form of a Cartesian product."""
    total = 1.0
    parameters = [
        NumericalDiscreteParameter(name="x1", values=[0.0, 0.5, 1.0]),
        NumericalDiscreteParameter(name="x2", values=[0.0, 0.5, 1.0]),
        TaskParameter(name="t1", values=["A", "B"]),
        TaskParameter(name="t2", values=["A", "B"]),
    ]
    subspace = SubspaceDiscrete.from_simplex(
        parameters, total=total, boundary_only=False
    )
    assert len(subspace.exp_rep) == 6 * 4  # <-- (# simplex part) x (# task part)
    assert not any(subspace.exp_rep.duplicated())
