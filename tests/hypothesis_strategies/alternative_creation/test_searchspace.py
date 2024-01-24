"""Test alternative ways of creation not considered in the strategies."""

import pandas as pd
import pytest
from pytest import param

from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace, SubspaceContinuous
from baybe.searchspace.discrete import SubspaceDiscrete

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
