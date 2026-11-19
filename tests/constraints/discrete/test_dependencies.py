"""Tests for the discrete dependencies constraint."""

from contextlib import nullcontext

import pytest
from pytest import param

from baybe.constraints.conditions import SubSelectionCondition, ThresholdCondition
from baybe.constraints.discrete import DiscreteDependenciesConstraint
from baybe.symmetries.dependency import DependencySymmetry
from tests.constraints.discrete.utils import assert_filter, assert_partial_rejected
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

##### API #####


@pytest.mark.parametrize(
    ("parameters", "conditions", "affected", "error", "match"),
    [
        param(
            ["s"],
            [],
            [],
            ValueError,
            "condition",
            id="empty-conditions",
        ),
        param(
            ["s", "t"],
            [SubSelectionCondition(["on"])],
            [["a"]],
            ValueError,
            "condition",
            id="too-few-conditions",
        ),
        param(
            ["s"],
            [SubSelectionCondition(["on"])] * 2,
            [["a"], ["b"]],
            ValueError,
            "condition",
            id="too-many-conditions",
        ),
        param(
            ["s"],
            [SubSelectionCondition(["on"])],
            [],
            ValueError,
            "condition",
            id="group-count",
        ),
        param(
            ["s"],
            [1],
            [["a"]],
            TypeError,
            "Condition",
            id="invalid-condition",
        ),
        param(
            ["s"],
            [SubSelectionCondition(["on"])],
            [["s"]],
            ValueError,
            "affected",
            id="self-dependency",
        ),
        param(
            ["s"],
            [SubSelectionCondition(["on"])],
            ["a"],
            ValueError,
            "Each group in 'affected_parameters' must be a sequence, not a string",
            id="string-group",
        ),
        param(
            ["s"],
            [SubSelectionCondition(["on"])],
            "a",
            ValueError,
            "Argument passed to 'affected_parameters'.*cannot be a string",
            id="string-groups",
        ),
        param(
            ["s"],
            [SubSelectionCondition(["on"])],
            [[1]],
            TypeError,
            "str",
            id="invalid-name",
        ),
    ],
)
def test_dependencies_invalid_groups(parameters, conditions, affected, error, match):
    """Every causing parameter requires a condition and valid affected group."""
    with pytest.raises(error, match=match):
        DiscreteDependenciesConstraint(parameters, conditions, affected)


def test_dependencies_to_symmetries():
    """Conversion preserves each causing parameter, condition, and affected group."""
    first = SubSelectionCondition(["on"])
    second = ThresholdCondition(0.0, ">")
    constraint = DiscreteDependenciesConstraint(
        ["switch", "amount"], [first, second], [["a", "b"], ["c"]]
    )
    assert constraint.to_symmetries() == (
        DependencySymmetry("switch", first, ["a", "b"]),
        DependencySymmetry("amount", second, ["c"]),
    )


##### Behavior #####


@pytest.mark.parametrize("exclude", [False, True], ids=["keep", "exclude"])
def test_dependencies_filter(exclude, backend):
    """Keep the first equivalent row while respecting other columns and inversion."""
    data = [
        {"switch": "on", "value": 1, "group": "a"},
        {"switch": "on", "value": 2, "group": "a"},
        {"switch": "off", "value": 1, "group": "a"},
        {"switch": "off", "value": 2, "group": "a"},
        {"switch": "off", "value": 2, "group": "b"},
        {"switch": "other", "value": 2, "group": "a"},
    ]
    constraint = DiscreteDependenciesConstraint(
        ["switch"], [SubSelectionCondition(["on"])], [["value"]], exclude=exclude
    )
    expected = [data[3]] if exclude else [data[i] for i in (0, 1, 2, 4, 5)]
    assert_filter(constraint, data, expected, backend)


@pytest.mark.parametrize(
    ("parameters", "expected_indices"),
    # TODO: Overlaps should be rejected because the semantics have not yet been
    #  fully defined. However, the current implementation simply ignores this and
    #  instead produces an order-dependent result. Adjust test docstring one fixed.
    [
        param(["first", "second"], (0, 2, 3, 4), id="first-then-second"),
        param(["second", "first"], (0, 2, 3, 5), id="second-then-first"),
        param(
            ["first", "second"],
            None,
            id="overlap-rejected",
            marks=pytest.mark.xfail(
                raises=pytest.fail.Exception,
                reason="Overlapping dependencies are not yet rejected.",
                strict=True,
            ),
        ),
    ],
)
def test_overlapping_dependencies(parameters, expected_indices, backend):
    """Capture the current behavior for a parameter with multiple dependencies."""
    data = [
        {"first": "off", "second": "on", "value": 1},
        {"first": "off", "second": "on", "value": 2},
        {"first": "on", "second": "on", "value": 1},
        {"first": "on", "second": "on", "value": 2},
        {"first": "off", "second": "off", "value": 1},
        {"first": "on", "second": "off", "value": 2},
    ]
    with (
        pytest.raises(ValueError, match="overlap")
        if expected_indices is None
        else nullcontext()
    ):
        constraint = DiscreteDependenciesConstraint(
            parameters,
            [SubSelectionCondition(["on"]), SubSelectionCondition(["on"])],
            [["value"], ["value"]],
        )
    if expected_indices is not None:
        assert_filter(constraint, data, [data[i] for i in expected_indices], backend)


@pytest.mark.parametrize("exclude", [False, True], ids=["keep", "exclude"])
@pytest.mark.parametrize(
    "data",
    [[{"switch": "on"}], [{"value": 1}]],
    ids=["missing-affected", "missing-causing"],
)
def test_dependencies_partial_rejected(exclude, data, backend):
    """Both causing and affected parameters are required for evaluation."""
    constraint = DiscreteDependenciesConstraint(
        ["switch"], [SubSelectionCondition(["on"])], [["value"]], exclude=exclude
    )
    assert_partial_rejected(constraint, data, backend)
