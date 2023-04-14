"""Tests for the searchspace module."""
import pytest
import torch

from baybe.parameters import NumericContinuous, NumericDiscrete
from baybe.searchspace import SearchSpace, SubspaceContinuous, SubspaceDiscrete


def test_empty_parameters():
    """Creation of a search space with no parameters raises an exception."""
    with pytest.raises(ValueError):
        SearchSpace.create(parameters=[])


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
    searchspace = SearchSpace.create(parameters=parameters)
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
    searchspace_discrete = SubspaceDiscrete.create(parameters=parameters)
    searchspace_continuous = SubspaceContinuous(parameters=parameters)
    expected = torch.empty(2, 0)
    assert torch.equal(searchspace_discrete.param_bounds_comp, expected)
    assert torch.equal(searchspace_continuous.param_bounds_comp, expected)
