"""Validation tests for targets."""

import operator as op

import pytest
from attrs import evolve
from pytest import param

from baybe.targets.binary import BinaryTarget
from baybe.targets.numerical import NumericalTarget

X = object()


@pytest.mark.parametrize(
    ("name", "transformation", "minimize", "match"),
    [
        param(None, X, X, "must be <class 'str'>", id="name_none"),
        param(0, X, X, "must be <class 'str'>", id="name_int"),
        param(X, 0, X, "must be callable", id="transformation_int"),
        param(X, X, None, "must be <class 'bool'>", id="minimize_none"),
        param(X, X, 0, "must be <class 'bool'>", id="minimize_int"),
    ],
)
def test_numerical_target_invalid_arguments(name, transformation, minimize, match):
    """Providing invalid argument types raises an error."""
    name = "t" if name is X else name
    transformation = None if transformation is X else transformation
    minimize = False if minimize is X else minimize
    with pytest.raises(TypeError, match=match):
        NumericalTarget.from_modern_interface(name, transformation, minimize=minimize)


@pytest.mark.parametrize(
    "other",
    [
        param({"name": "t2"}, id="name"),
        param({"minimize": True}, id="minimize"),
        param({"metadata": {"unit": "g"}}, id="metadata"),
    ],
)
@pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])
def test_numerical_target_invalid_combination(other, operator):
    """Targets with different attributes (except transformation) cannot be combined."""
    t = NumericalTarget("t")
    with pytest.raises(ValueError, match="if they are identical in all attributes"):
        operator(t, evolve(t, **other))


def test_numerical_target_invalid_operator():
    """Target division is not supported."""
    t = NumericalTarget("t")
    with pytest.raises(TypeError, match="unsupported operand"):
        t / t


@pytest.mark.parametrize(
    ("choices", "error", "match"),
    [
        param((None, 0), TypeError, "'success_value' must be", id="wrong_type"),
        param((0, 0), ValueError, "must be different", id="identical"),
    ],
)
def test_binary_target_invalid_values(choices, error, match):
    """Providing invalid choice values raises an error."""
    with pytest.raises(error, match=match):
        BinaryTarget(
            name="invalid_value",
            success_value=choices[0],
            failure_value=choices[1],
        )
