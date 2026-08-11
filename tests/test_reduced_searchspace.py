"""Tests for _ReducedSearchSpace and SearchSpace._drop_parameters."""

import pytest

from baybe.parameters import (
    CategoricalParameter,
    NumericalContinuousParameter,
    TaskParameter,
)
from baybe.searchspace import SearchSpace


@pytest.fixture(name="reduced_searchspace")
def fixture_reduced_searchspace():
    """A reduced search space with the task parameter removed."""
    ss = SearchSpace.from_product(
        [
            CategoricalParameter("Color", ["red", "blue"]),
            NumericalContinuousParameter("x", (0.0, 1.0)),
            TaskParameter("Task", ["A", "B"]),
        ]
    )
    return ss._drop_parameters({"Task"})


def test_reduced_parameters(reduced_searchspace):
    """Verify that the reduced space exposes only the remaining parameters."""
    names = {p.name for p in reduced_searchspace.parameters}
    assert names == {"Color", "x"}


def test_reduced_parameter_names(reduced_searchspace):
    """Verify that parameter_names derives from the remaining parameters."""
    assert set(reduced_searchspace.parameter_names) == {"Color", "x"}


def test_reduced_comp_rep_columns(reduced_searchspace):
    """Verify that comp_rep_columns matches the remaining parameters' columns."""
    expected = set()
    for p in reduced_searchspace.parameters:
        expected.update(p.comp_rep_columns)
    assert set(reduced_searchspace.comp_rep_columns) == expected


def test_reduced_n_tasks(reduced_searchspace):
    """Verify that n_tasks is 1 when no task parameter is present."""
    assert reduced_searchspace.n_tasks == 1


def test_reduced_get_n_comp_rep_columns(reduced_searchspace):
    """Verify that _get_n_comp_rep_columns works on the reduced space."""
    from baybe.parameters.selectors import TypeSelector

    # String-based selection
    assert reduced_searchspace._get_n_comp_rep_columns("x") == 1
    assert reduced_searchspace._get_n_comp_rep_columns("Color") == 2

    # Selector-based selection
    assert (
        reduced_searchspace._get_n_comp_rep_columns(
            TypeSelector([NumericalContinuousParameter])
        )
        == 1
    )
    assert reduced_searchspace._get_n_comp_rep_columns(lambda _: True) == 3


def test_reduced_blocked_attributes(reduced_searchspace):
    """Verify that all non-allowed attributes raise NotImplementedError."""
    from baybe.searchspace.core import _ReducedSearchSpace

    allowed = _ReducedSearchSpace._ALLOWED_ATTRIBUTES

    # All public non-dunder attributes of SearchSpace
    all_attrs = {name for name in dir(SearchSpace) if not name.startswith("_")}

    # Attributes that should be blocked (not in allowlist)
    blocked = all_attrs - allowed

    for attr in sorted(blocked):
        with pytest.raises(AttributeError, match=attr):
            getattr(reduced_searchspace, attr)


def test_reduced_repr(reduced_searchspace):
    """Verify that repr does not crash on a reduced search space."""
    result = repr(reduced_searchspace)
    assert "_ReducedSearchSpace" in result


def test_reduced_str(reduced_searchspace):
    """Verify that str does not crash on a reduced search space."""
    result = str(reduced_searchspace)
    assert isinstance(result, str)


def test_reduced_eq(reduced_searchspace):
    """Verify that equality comparison works on reduced search spaces."""
    ss = SearchSpace.from_product(
        [
            CategoricalParameter("Color", ["red", "blue"]),
            NumericalContinuousParameter("x", (0.0, 1.0)),
            TaskParameter("Task", ["A", "B"]),
        ]
    )
    other = ss._drop_parameters({"Task"})
    assert reduced_searchspace == other


def test_reduced_unknown_parameter_raises():
    """Verify that removing a nonexistent parameter raises an error."""
    ss = SearchSpace.from_product(
        [
            CategoricalParameter("Color", ["red", "blue"]),
            NumericalContinuousParameter("x", (0.0, 1.0)),
        ]
    )
    with pytest.raises(ValueError, match="not found"):
        ss._drop_parameters({"nonexistent"})


def test_reduced_multiple_parameters_removed():
    """Verify that multiple parameters can be removed at once."""
    ss = SearchSpace.from_product(
        [
            CategoricalParameter("Color", ["red", "blue"]),
            NumericalContinuousParameter("x", (0.0, 1.0)),
            TaskParameter("Task", ["A", "B"]),
        ]
    )
    reduced = ss._drop_parameters({"Task", "Color"})
    assert reduced.parameter_names == ("x",)
