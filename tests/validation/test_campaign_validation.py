"""Validation tests for Campaign."""

import pytest

from baybe import Campaign
from baybe.objectives import ParetoObjective
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.targets import NumericalTarget


def test_overlapping_target_parameter_names():
    """Overlapping names between parameters and targets are not allowed."""
    targets = (
        NumericalTarget("n1", "MAX"),
        NumericalTarget("n2", "MAX"),
        NumericalTarget("n3", "MAX"),
    )
    parameters = (
        NumericalDiscreteParameter("n1", [1, 2, 3]),
        NumericalDiscreteParameter("dies", [4, 5, 6]),
        NumericalDiscreteParameter("n2", [7, 8, 9]),
    )
    searchspace = SearchSpace.from_product(parameters)
    objective = ParetoObjective(targets)

    with pytest.raises(ValueError, match="appear in both collections: {'n2', 'n1'}."):
        Campaign(searchspace=searchspace, objective=objective)
