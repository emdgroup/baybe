"""Validation tests for objective."""

from unittest.mock import Mock

import pytest
from cattrs import IterableValidationError
from pytest import param

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.single import SingleTargetObjective
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget

t1 = NumericalTarget("t1", mode="MAX", bounds=(0, 1))
t2 = NumericalTarget("t2", mode="MAX", bounds=(0, 1))
t3 = NumericalTarget("unnormalized", mode="MAX")
t_mock = Mock(spec=Target)


@pytest.mark.parametrize(
    "target",
    [
        param(None, id="none"),
        param("no_target", id="string"),
    ],
)
def test_invalid_target(target):
    """Providing an invalid target object raises an exception."""
    with pytest.raises(TypeError):
        SingleTargetObjective(target)


@pytest.mark.parametrize(
    ("targets", "error"),
    [
        param(None, TypeError, id="none"),
        param([t1, "t2"], TypeError, id="wrong_type"),
        param([t1], ValueError, id="too_short"),
        param([t1, t1], ValueError, id="duplicate_names"),
        param([t1, t3], ValueError, id="unnormalized"),
        param([t1, t_mock], TypeError, id="unsupported_subclass"),
    ],
)
def test_invalid_targets(targets, error):
    """Providing invalid target objects raises an exception."""
    with pytest.raises(error):
        DesirabilityObjective(targets)


@pytest.mark.parametrize(
    ("weights", "error"),
    [
        param(None, TypeError, id="none"),
        param([1.0, "abc"], IterableValidationError, id="wrong_type"),
        param([1.0, -1.0], ValueError, id="negative"),
        param([1.0, 0.0], ValueError, id="zero"),
        param([1.0, float("inf")], ValueError, id="inf"),
        param([1.0, float("nan")], ValueError, id="nan"),
        param([1.0], ValueError, id="wrong_length"),
    ],
)
def test_invalid_weights(weights, error):
    """Providing invalid weights raises an exception."""
    with pytest.raises(error):
        DesirabilityObjective([t1, t2], weights)


@pytest.mark.parametrize(
    "scalarizer",
    [
        param(None, id="none"),
        param("invalid", id="non_existing"),
    ],
)
def test_invalid_scalarizer(scalarizer):
    """Providing an invalid scalarizer raises an exception."""
    with pytest.raises(ValueError):
        DesirabilityObjective([t1, t2], scalarizer=scalarizer)
