"""Tests for shared discrete constraint behavior."""

import pytest
from pytest import param

from baybe.constraints.base import DiscreteConstraint, DiscreteFilteringConstraint
from baybe.constraints.conditions import SubSelectionCondition, ThresholdCondition
from baybe.constraints.discrete import (
    DiscreteBatchConstraint,
    DiscreteCardinalityConstraint,
    DiscreteCustomConstraint,
    DiscreteDependenciesConstraint,
    DiscreteLinearConstraint,
    DiscretePermutationInvarianceConstraint,
    DiscreteProductConstraint,
    DiscreteRepetitionLimitConstraint,
    DiscreteSelectionConstraint,
)
from baybe.utils.basic import get_subclasses
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

_CONSTRAINTS = [
    (DiscreteBatchConstraint, ["A"], {}),
    (DiscreteCardinalityConstraint, ["A", "B"], {"max_cardinality": 1}),
    (DiscreteCustomConstraint, ["A", "B"], {"validator": lambda df: df}),
    (
        DiscreteDependenciesConstraint,
        ["A", "B"],
        {
            "conditions": [SubSelectionCondition(["on"])] * 2,
            "affected_parameters": [["C"], ["D"]],
        },
    ),
    (DiscreteLinearConstraint, ["A", "B"], {"operator": "<=", "rhs": 1.0}),
    (DiscretePermutationInvarianceConstraint, ["A", "B"], {}),
    (DiscreteProductConstraint, ["A", "B"], {"operator": "<=", "rhs": 1.0}),
    (DiscreteRepetitionLimitConstraint, ["A", "B"], {"n_max_repetitions": 1}),
    (
        DiscreteSelectionConstraint,
        ["A", "B"],
        {"conditions": [ThresholdCondition(0.0, ">")] * 2},
    ),
]
_FILTERING_CONSTRAINTS = [
    case for case in _CONSTRAINTS if issubclass(case[0], DiscreteFilteringConstraint)
]

##### API #####


def test_shared_api_covers_all_discrete_constraint_classes():
    """The shared API cases cover every built-in discrete constraint class."""
    assert {case[0] for case in _CONSTRAINTS} == {
        cls
        for cls in get_subclasses(DiscreteConstraint)
        if cls.__module__.startswith("baybe.constraints.")
    }


@pytest.mark.parametrize(
    ("constraint_type", "parameters", "kwargs"),
    _CONSTRAINTS,
    ids=[case[0].__name__ for case in _CONSTRAINTS],
)
def test_discrete_constraint_valid_parameters(constraint_type, parameters, kwargs):
    """Each shared API case has a valid baseline configuration."""
    assert constraint_type(parameters=parameters, **kwargs).parameters == tuple(
        parameters
    )


@pytest.mark.parametrize(
    ("constraint_type", "parameters", "kwargs"),
    _CONSTRAINTS,
    ids=[case[0].__name__ for case in _CONSTRAINTS],
)
@pytest.mark.parametrize(
    ("invalid_parameters", "error", "match"),
    [
        param([], ValueError, "Length of 'parameters' must be >= 1", id="empty"),
        param(["A", "A"], ValueError, "must be unique", id="duplicate"),
        param([1, 2], TypeError, "must be <class 'str'>", id="type-int"),
        param(["A", None], TypeError, "must be <class 'str'>", id="type-none"),
    ],
)
def test_discrete_constraint_invalid_parameters(
    constraint_type, parameters, kwargs, invalid_parameters, error, match
):
    """Discrete constraints require non-empty, unique string parameter names."""
    with pytest.raises(error, match=match):
        constraint_type(parameters=invalid_parameters, **kwargs)


@pytest.mark.parametrize(
    ("constraint_type", "parameters", "kwargs"),
    _FILTERING_CONSTRAINTS,
    ids=[case[0].__name__ for case in _FILTERING_CONSTRAINTS],
)
def test_discrete_filtering_constraint_invalid_exclude(
    constraint_type, parameters, kwargs
):
    """The exclusion flag must be Boolean."""
    with pytest.raises(TypeError, match="exclude"):
        constraint_type(parameters=parameters, exclude=1, **kwargs)
