"""Tests for threshold conditions."""

import pytest
from pytest import param

from baybe.constraints.conditions import ThresholdCondition
from tests.constraints.conditions.utils import apply_condition
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

##### API #####


@pytest.mark.parametrize(
    "operator",
    [
        param("~", id="unknown"),
        param("<>", id="wrong"),
        param(">=>", id="extended"),
    ],
)
def test_threshold_condition_invalid_operator(operator):
    """Threshold conditions reject unknown comparison operators."""
    with pytest.raises(ValueError, match="'operator' must be in"):
        ThresholdCondition(threshold=1.0, operator=operator)


@pytest.mark.parametrize(
    ("threshold", "match"),
    [
        param(float("nan"), "cannot be 'nan'", id="nan"),
        param(float("inf"), "cannot be 'inf'", id="inf"),
        param(-float("inf"), "cannot be 'inf'", id="neg-inf"),
        param("high", "could not convert string to float", id="non-numeric"),
    ],
)
def test_threshold_condition_invalid_threshold(threshold, match):
    """Threshold conditions require a finite numerical threshold."""
    with pytest.raises(ValueError, match=match):
        ThresholdCondition(threshold=threshold, operator="<")


@pytest.mark.parametrize(
    ("operator", "tolerance", "match"),
    [
        param("<", 0.1, "only valid with the following operators", id="unsupported"),
        param("=", None, "cannot be None or <= 0.0", id="none"),
        param("=", 0.0, "cannot be None or <= 0.0", id="zero"),
        param("=", -1.0, "must be >= 0", id="negative"),
        param("=", float("nan"), "cannot be 'nan'", id="nan"),
    ],
)
def test_threshold_condition_invalid_tolerance(operator, tolerance, match):
    """Threshold conditions reject invalid or inapplicable tolerances."""
    with pytest.raises(ValueError, match=match):
        ThresholdCondition(threshold=1.0, operator=operator, tolerance=tolerance)


##### Behavior #####


@pytest.mark.parametrize(
    ("operator", "expected"),
    [
        param("<", [1.0], id="lt"),
        param("<=", [1.0, 2.0], id="le"),
        param("=", [2.0], id="eq"),
        param("==", [2.0], id="eq-alias"),
        param("!=", [1.0, 3.0], id="ne"),
        param(">", [3.0], id="gt"),
        param(">=", [2.0, 3.0], id="ge"),
    ],
)
def test_threshold_condition_operators(operator, expected, backend):
    """Threshold conditions implement every supported comparison operator.

    Null values never satisfy a condition.
    """
    condition = ThresholdCondition(threshold=2.0, operator=operator)
    assert apply_condition(condition, [1.0, 2.0, 3.0, None], backend) == expected


@pytest.mark.parametrize(
    ("kwargs", "values", "expected"),
    [
        param(
            {"tolerance": 0.1},
            [1.89, 1.9, None, 2.0, 2.1, 2.11],
            [1.9, 2.0, 2.1],
            id="explicit",
        ),
        param({}, [2.0, 2.0 + 1e-9, 2.0 + 1e-7], [2.0, 2.0 + 1e-9], id="default"),
    ],
)
def test_threshold_condition_tolerance(kwargs, values, expected, backend):
    """Equality keeps values within the explicit or default tolerance window."""
    condition = ThresholdCondition(threshold=2.0, operator="=", **kwargs)
    assert apply_condition(condition, values, backend) == expected


@pytest.mark.parametrize(
    ("operator", "tolerance"),
    [
        param("=", 1e-8, id="tolerance-operator"),
        param("<", None, id="comparison-operator"),
    ],
)
def test_threshold_condition_default_tolerance(operator, tolerance):
    """Only tolerance-enabled operators receive a default tolerance."""
    assert ThresholdCondition(threshold=2.0, operator=operator).tolerance == tolerance


@pytest.mark.parametrize(
    "values",
    [
        param(["a", "b"], id="strings"),
        param([True, False], id="booleans"),
    ],
)
def test_threshold_condition_non_numeric_series(values, backend):
    condition = ThresholdCondition(threshold=2.0, operator="<")
    with pytest.raises(ValueError, match="non-numerical data"):
        apply_condition(condition, values, backend)
