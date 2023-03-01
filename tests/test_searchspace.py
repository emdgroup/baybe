"""Tests for the SearchSpace class."""

from baybe.parameters import NumericContinuous, NumericDiscrete
from baybe.searchspace import SearchSpace


def test_bounds():
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
    searchspace = SearchSpace(parameters)
    assert searchspace.bounds == [(1.0, 3.0), (7.0, 9.0), (4.0, 6.0), (10.0, 12.0)]
