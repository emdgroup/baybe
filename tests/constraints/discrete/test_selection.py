"""Tests for the discrete selection constraint."""

import narwhals.stable.v2 as nw
import pytest
from pytest import param

from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.discrete import DiscreteSelectionConstraint
from tests.constraints.discrete.utils import (
    assert_filter,
    assert_partial_rejected,
)
from tests.constraints.utils import requires_fulltest

pytestmark = requires_fulltest

##### API #####


@pytest.mark.parametrize(
    ("parameters", "error", "match"),
    [
        param([], ValueError, "Length of 'parameters' must be >= 1", id="empty"),
        param(["A", "A"], ValueError, "must have unique values", id="duplicate"),
        param([1, 2], TypeError, "must be <class 'str'>", id="type-int"),
        param(["A", None], TypeError, "must be <class 'str'>", id="type-none"),
    ],
)
def test_selection_constraint_invalid_parameters(parameters, error, match):
    """Selection constraints require a non-empty list of unique parameter names."""
    with pytest.raises(error, match=match):
        DiscreteSelectionConstraint(
            parameters=parameters,
            conditions=[ThresholdCondition(threshold=0.0, operator=">")],
        )


def test_selection_constraint_invalid_conditions():
    """Selection constraints require at least one condition."""
    with pytest.raises(ValueError, match="Length of 'conditions' must be >= 1"):
        DiscreteSelectionConstraint(parameters=["temperature"], conditions=[])


@pytest.mark.parametrize(
    "combiner",
    [
        param("NAND", id="unknown"),
        param("and", id="lowercase"),
        param("", id="empty"),
    ],
)
def test_selection_constraint_invalid_combiner(combiner):
    """Selection constraints reject unknown logic combiners."""
    with pytest.raises(ValueError, match="'combiner' must be in"):
        DiscreteSelectionConstraint(
            parameters=["temperature"],
            conditions=[ThresholdCondition(threshold=0.0, operator=">")],
            combiner=combiner,
        )


##### Behavior #####


_SELECTION_DATA = [
    {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
    {"temperature": 1.0, "pressure": 0.0, "solvent": "ethanol"},
    {"temperature": 0.0, "pressure": 1.0, "solvent": "methanol"},
    {"temperature": 0.0, "pressure": 0.0, "solvent": "acetone"},
]


_SELECTION_PARTIAL_DATA = (
    nw.from_dicts(_SELECTION_DATA, backend=nw.Implementation.POLARS)
    .select("temperature", "solvent")
    .rows(named=True)
)


@pytest.mark.parametrize(
    ("combiner", "exclude", "expected_rows"),
    [
        param(
            "AND",
            False,
            [
                {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
            ],
            id="AND-keep",
        ),
        param(
            "AND",
            True,
            [
                {"temperature": 1.0, "pressure": 0.0, "solvent": "ethanol"},
                {"temperature": 0.0, "pressure": 1.0, "solvent": "methanol"},
                {"temperature": 0.0, "pressure": 0.0, "solvent": "acetone"},
            ],
            id="AND-exclude",
        ),
        param(
            "OR",
            False,
            [
                {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
                {"temperature": 1.0, "pressure": 0.0, "solvent": "ethanol"},
                {"temperature": 0.0, "pressure": 1.0, "solvent": "methanol"},
            ],
            id="OR-keep",
        ),
        param(
            "OR",
            True,
            [
                {"temperature": 0.0, "pressure": 0.0, "solvent": "acetone"},
            ],
            id="OR-exclude",
        ),
        param(
            "XOR",
            False,
            [
                {"temperature": 1.0, "pressure": 0.0, "solvent": "ethanol"},
                {"temperature": 0.0, "pressure": 1.0, "solvent": "methanol"},
            ],
            id="XOR-keep",
        ),
        param(
            "XOR",
            True,
            [
                {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
                {"temperature": 0.0, "pressure": 0.0, "solvent": "acetone"},
            ],
            id="XOR-exclude",
        ),
    ],
)
def test_selection_constraint(combiner, exclude, expected_rows, backend):
    """Selection constraints cover all combiner and exclusion regimes."""
    constraint = DiscreteSelectionConstraint(
        parameters=["temperature", "pressure"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        combiner=combiner,
        exclude=exclude,
    )
    assert_filter(constraint, _SELECTION_DATA, expected_rows, backend)


@pytest.mark.parametrize(
    ("combiner", "exclude", "expected_rows"),
    [
        param(
            "AND",
            False,
            [
                {"temperature": 1.0, "solvent": "water"},
                {"temperature": 1.0, "solvent": "ethanol"},
            ],
            id="and-keep",
        ),
        param(
            "OR",
            True,
            [
                {"temperature": 0.0, "solvent": "methanol"},
                {"temperature": 0.0, "solvent": "acetone"},
            ],
            id="or-exclude",
        ),
    ],
)
def test_selection_constraint_partial_accepted(
    combiner, exclude, expected_rows, backend
):
    """Accepted partial selection filtering uses only the available columns."""
    constraint = DiscreteSelectionConstraint(
        parameters=["temperature", "pressure"],
        conditions=[
            ThresholdCondition(0.0, ">"),
            ThresholdCondition(0.0, ">"),
        ],
        combiner=combiner,
        exclude=exclude,
    )
    assert_filter(constraint, _SELECTION_PARTIAL_DATA, expected_rows, backend)


@pytest.mark.parametrize(
    ("combiner", "exclude"),
    [
        param("AND", True, id="and-exclude"),
        param("OR", False, id="or-keep"),
        param("XOR", False, id="xor-keep"),
        param("XOR", True, id="xor-exclude"),
    ],
)
def test_selection_constraint_partial_rejected(combiner, exclude, backend):
    """Rejected partial selection filtering raises."""
    constraint = DiscreteSelectionConstraint(
        parameters=["temperature", "pressure"],
        conditions=[
            ThresholdCondition(0.0, ">"),
            ThresholdCondition(0.0, ">"),
        ],
        combiner=combiner,
        exclude=exclude,
    )
    assert_partial_rejected(constraint, _SELECTION_PARTIAL_DATA, backend)
