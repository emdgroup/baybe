"""Test alternative ways of creation not considered in the strategies."""

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.categorical import TaskParameter
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete
from tests.hypothesis_strategies.parameters import numerical_discrete_parameters

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
        param(s_x.to_frame(), (p_x,), (p_x,), id="num-match"),
        param(s_x.to_frame(), (p_x_over,), (p_x_over,), id="num_overparametrized"),
        param(s_x.to_frame(), (p_x_under,), ValueError, id="num_underparametrized"),
        param(df_discrete, (p_x, p_x), ValueError, id="duplicate-name"),
        param(s_x.to_frame(), (p_x, p_y), ValueError, id="no_match"),
        param(s_y.to_frame(), (p_y,), (p_y,), id="cat-match"),
        param(df_discrete, (p_x, p_y), (p_x, p_y), id="both-match"),
        param(df_discrete, (p_x,), (p_x, p_y), id="one-unspecified"),
    ],
)
def test_discrete_space_creation_from_dataframe(df, parameters, expected):
    """Parameters are automatically inferred and exceptions are triggered."""
    if isinstance(expected, tuple):
        subspace = SubspaceDiscrete.from_dataframe(df, parameters)
        actual = subspace.parameters
        assert actual == expected, (actual, expected)
    else:
        with pytest.raises(expected):
            SubspaceDiscrete.from_dataframe(df, parameters)


@pytest.mark.parametrize(
    ("df", "parameters", "expected"),
    [
        param(s_a.to_frame(), (p_a,), (p_a,), id="match"),
        param(s_a.to_frame(), (p_a_over,), (p_a_over,), id="overparametrized"),
        param(s_a.to_frame(), (p_a_under,), ValueError, id="underparametrized"),
        param(df_continuous, (p_a, p_a), ValueError, id="duplicate-name"),
        param(s_a.to_frame(), (p_a, p_b), ValueError, id="no_match"),
        param(df_continuous, (p_a,), (p_a, p_b), id="one-unspecified"),
    ],
)
def test_continuous_space_creation_from_dataframe(df, parameters, expected):
    """Parameters are automatically inferred and exceptions are triggered."""
    if isinstance(expected, tuple):
        subspace = SubspaceContinuous.from_dataframe(df, parameters)
        actual = subspace.parameters
        assert actual == expected, (actual, expected)
    else:
        with pytest.raises(expected):
            SubspaceContinuous.from_dataframe(df, parameters)


@pytest.mark.parametrize(
    ("df", "parameters", "expected"),
    [
        param(df, (p_x, p_y, p_a, p_b), (p_x, p_y, p_a, p_b), id="match"),
        param(df, (p_x, p_x, p_x, p_x), ValueError, id="duplicates"),
        param(df, (p_x,), ValueError, id="missing"),
    ],
)
def test_searchspace_creation_from_dataframe(df, parameters, expected):
    """Parameters are automatically inferred and exceptions are triggered."""
    if isinstance(expected, tuple):
        subspace = SearchSpace.from_dataframe(df, parameters)
        actual = subspace.parameters
        assert actual == expected, (actual, expected)
    else:
        with pytest.raises(expected):
            SearchSpace.from_dataframe(df, parameters)


def test_discrete_searchspace_creation_from_degenerate_dataframe():
    """A degenerate dataframe with index but no columns yields an empty space."""
    df = pd.DataFrame(index=[0])
    subspace = SubspaceDiscrete.from_dataframe(df)
    assert_frame_equal(subspace.exp_rep, pd.DataFrame())


@pytest.mark.parametrize("boundary_only", (False, True))
@given(
    parameters=st.lists(
        numerical_discrete_parameters(min_value=0.0, max_value=1.0),
        min_size=1,
        max_size=5,
        unique_by=lambda x: x.name,
    )
)
def test_discrete_space_creation_from_simplex_inner(parameters, boundary_only):
    """Candidates from a simplex space satisfy the simplex constraint."""
    tolerance = 1e-6
    max_possible = sum(max(p.values) for p in parameters)
    min_possible = sum(min(p.values) for p in parameters)

    if boundary_only:
        # Ensure there exists configurations both inside and outside the simplex
        max_sum = (max_possible + min_possible) / 2
    else:
        # We use the maximum parameter sum because it can be exactly achieved (for other
        # values, except for the minimum, it's not guaranteed there actually exists
        # a parameter combination that can exactly hit it)
        max_sum = max_possible

    subspace = SubspaceDiscrete.from_simplex(
        max_sum, parameters, boundary_only=boundary_only, tolerance=tolerance
    )

    if boundary_only:
        assert np.allclose(subspace.exp_rep.sum(axis=1), max_sum, atol=tolerance)
    else:
        assert (subspace.exp_rep.sum(axis=1) <= max_sum + tolerance).all()


p_d1 = NumericalDiscreteParameter(name="d1", values=[0.0, 0.5, 1.0])
p_d2 = NumericalDiscreteParameter(name="d2", values=[0.0, 0.5, 1.0])
p_t1 = TaskParameter(name="t1", values=["A", "B"])
p_t2 = TaskParameter(name="t2", values=["A", "B"])


@pytest.mark.parametrize(
    ("simplex_parameters", "product_parameters", "n_elements"),
    [
        param([p_d1, p_d2], [p_t1, p_t2], 6 * 4, id="both"),
        param([p_d1, p_d2], [], 6, id="simplex-only"),
        param([], [p_t1, p_t2], 4, id="task_only"),
    ],
)
def test_discrete_space_creation_from_simplex_mixed(
    simplex_parameters, product_parameters, n_elements
):
    """Additional non-simplex parameters enter in form of a Cartesian product."""
    max_sum = 1.0
    subspace = SubspaceDiscrete.from_simplex(
        max_sum,
        simplex_parameters,
        product_parameters=product_parameters,
        boundary_only=False,
    )
    assert len(subspace.exp_rep) == n_elements  # <-- (# simplex part) x (# task part)
    assert not any(subspace.exp_rep.duplicated())
    assert len(subspace.parameters) == len(subspace.exp_rep.columns)
    assert all(p.name in subspace.exp_rep.columns for p in subspace.parameters)


@pytest.mark.parametrize("boundary_only", (False, True))
def test_discrete_space_creation_from_simplex_restricted(boundary_only):
    """The number of nonzero simplex parameters is controllable."""
    params = [
        NumericalDiscreteParameter(f"p{i}", np.linspace(0, 1, 11)) for i in range(10)
    ]
    subspace = SubspaceDiscrete.from_simplex(
        max_sum=1.0,
        simplex_parameters=params,
        min_nonzero=2,
        max_nonzero=4,
        boundary_only=True,
    )
    n_nonzero = (subspace.exp_rep > 0.0).sum(axis=1)
    if boundary_only:
        assert np.allclose(subspace.exp_rep.sum(axis=1), 1.0)
    assert n_nonzero.min() == 2
    assert n_nonzero.max() == 4
    assert len(subspace.parameters) == len(subspace.exp_rep.columns)
    assert all(p.name in subspace.exp_rep.columns for p in subspace.parameters)
