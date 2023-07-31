"""Tests for the searchspace module."""
import pandas as pd
import pytest
import torch
from baybe.exceptions import EmptySearchSpaceError

from baybe.parameters import Categorical, NumericContinuous, NumericDiscrete
from baybe.searchspace import (
    SearchSpace,
    SearchSpaceType,
    SubspaceContinuous,
    SubspaceDiscrete,
)


def test_empty_parameters():
    """Creation of a search space with no parameters raises an exception."""
    with pytest.raises(EmptySearchSpaceError):
        SearchSpace()


def test_bounds_order():
    """
    Asserts that the bounds are created in the correct order (discrete parameters
    first, continuous next).
    """
    parameters = [
        NumericDiscrete(name="A_disc", values=[1.0, 2.0, 3.0]),
        NumericContinuous(name="A_cont", bounds=(4.0, 6.0)),
        NumericDiscrete(name="B_disc", values=[7.0, 8.0, 9.0]),
        NumericContinuous(name="B_cont", bounds=(10.0, 12.0)),
    ]
    searchspace = SearchSpace.from_product(parameters=parameters)
    expected = torch.tensor([[1.0, 7.0, 4.0, 10.0], [3.0, 9.0, 6.0, 12.0]]).double()
    assert torch.equal(
        searchspace.param_bounds_comp,
        expected,
    )


def test_empty_parameter_bounds():
    """
    Asserts that the correct bounds (with correct shapes) are produced for empty
    search spaces.
    """
    parameters = []
    searchspace_discrete = SubspaceDiscrete.from_product(parameters=parameters)
    searchspace_continuous = SubspaceContinuous(parameters=parameters)
    expected = torch.empty(2, 0)
    assert torch.equal(searchspace_discrete.param_bounds_comp, expected)
    assert torch.equal(searchspace_continuous.param_bounds_comp, expected)


def test_discrete_searchspace_creation_from_dataframe():
    """A purely discrete search space is created from an example dataframe."""
    num_specified = NumericDiscrete(name="num_specified", values=[1, 2, 3])
    num_unspecified = NumericDiscrete(name="num_unspecified", values=[4, 5, 6])
    cat_specified = Categorical(name="cat_specified", values=["a", "b", "c"])
    cat_unspecified = Categorical(name="cat_unspecified", values=["d", "e", "f"])

    all_params = [num_specified, num_unspecified, cat_specified, cat_unspecified]

    df = pd.DataFrame({param.name: param.values for param in all_params})
    searchspace = SearchSpace(
        SubspaceDiscrete.from_dataframe(df, parameters=[num_specified, cat_specified])
    )

    assert searchspace.type == SearchSpaceType.DISCRETE
    assert searchspace.parameters == all_params
    assert df.equals(searchspace.discrete.exp_rep)


def test_continuous_searchspace_creation_from_bounds():
    """A purely continuous search space is created from example bounds."""
    parameters = [
        NumericContinuous("param1", (0, 1)),
        NumericContinuous("param2", (-1, 1)),
    ]
    bounds = pd.DataFrame({p.name: p.bounds.to_tuple() for p in parameters})
    searchspace = SearchSpace(continuous=SubspaceContinuous.from_bounds(bounds))

    assert searchspace.type == SearchSpaceType.CONTINUOUS
    assert searchspace.parameters == parameters


def test_hyperrectangle_searchspace_creation():
    """
    A purely continuous hyperrectangle-shaped search space is created that spans a
    certain set of points.
    """
    points = pd.DataFrame(
        {
            "param1": [0, 1, 2],
            "param2": [-1, 0, 1],
        }
    )
    searchspace = SearchSpace(continuous=SubspaceContinuous.from_dataframe(points))

    parameters = [
        NumericContinuous("param1", (0, 2)),
        NumericContinuous("param2", (-1, 1)),
    ]

    assert searchspace.type == SearchSpaceType.CONTINUOUS
    assert searchspace.parameters == parameters
