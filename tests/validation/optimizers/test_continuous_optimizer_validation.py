"""Validation tests for ContinuousOptimizer."""

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from pytest import param

from baybe.constraints.continuous import ContinuousCardinalityConstraint
from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.optimizers.continuous import ContinuousOptimizer
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace import SearchSpace

_P1 = NumericalContinuousParameter("x1", bounds=(0, 1))
_P2 = NumericalContinuousParameter("x2", bounds=(-1, 0))
_PD = NumericalDiscreteParameter("d", values=[1, 2, 3])

_SS_DISCRETE = SearchSpace.from_product([_PD])
_SS_SUBSETS = SearchSpace.from_product(
    [_P1, _P2],
    constraints=[
        ContinuousCardinalityConstraint(
            parameters=["x1", "x2"], min_cardinality=1, max_cardinality=1
        )
    ],
)


@pytest.mark.parametrize(
    ("n_starts", "error", "match"),
    [
        param(10, None, None, id="valid"),
        param(1, None, None, id="valid_minimum"),
        param(1.5, TypeError, "must be <class 'int'>", id="float"),
        param("5", TypeError, "must be <class 'int'>", id="string"),
        param(0, ValueError, "must be > 0", id="zero"),
        param(-1, ValueError, "must be > 0", id="negative"),
    ],
)
def test_n_starts(n_starts, error, match):
    """Invalid ``n_starts`` values raise an error."""
    with pytest.raises(error, match=match) if error is not None else nullcontext():
        ContinuousOptimizer(n_starts=n_starts)


@pytest.mark.parametrize(
    ("n_initial_samples", "error", "match"),
    [
        param(64, None, None, id="valid"),
        param(1, None, None, id="valid_minimum"),
        param(1.5, TypeError, "must be <class 'int'>", id="float"),
        param("64", TypeError, "must be <class 'int'>", id="string"),
        param(0, ValueError, "must be > 0", id="zero"),
        param(-5, ValueError, "must be > 0", id="negative"),
    ],
)
def test_n_initial_samples(n_initial_samples, error, match):
    """Invalid ``n_initial_samples`` values raise an error."""
    with pytest.raises(error, match=match) if error is not None else nullcontext():
        ContinuousOptimizer(n_initial_samples=n_initial_samples)


def test_sequential():
    """An unparseable value for ``sequential`` raises an error."""
    with pytest.raises(ValueError, match="Cannot convert 'something'"):
        ContinuousOptimizer(sequential="something")


@pytest.mark.parametrize(
    ("searchspace", "match"),
    [
        param(_SS_DISCRETE, "non-empty continuous space", id="discrete_only"),
        param(_SS_SUBSETS, "containing no subsets", id="with_subsets"),
    ],
)
def test_raises_on_incompatible_searchspace(searchspace, match):
    """Incompatible search spaces raise an error."""
    with pytest.raises(IncompatibleSearchSpaceError, match=match):
        ContinuousOptimizer()(1, MagicMock(), searchspace)
