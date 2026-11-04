"""Test for imposing discrete constraints."""

import os
from collections.abc import Mapping, Sequence
from inspect import signature

import narwhals.stable.v2 as nw
import pytest
from narwhals.testing import assert_frame_equal as assert_nw_frame_equal
from pytest import param

from baybe.constraints.base import DiscreteFilteringConstraint
from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.discrete import (
    DiscreteCardinalityConstraint,
    DiscreteCustomConstraint,
    DiscreteLinearConstraint,
    DiscreteProductConstraint,
    DiscreteSelectionConstraint,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Requires the optional Polars dependency.",
)


@pytest.fixture(
    params=[nw.Implementation.PANDAS, nw.Implementation.POLARS],
    ids=["pandas", "polars"],
    name="backend",
)
def fixture_backend(request: pytest.FixtureRequest) -> nw.Implementation:
    """Return the dataframe backend used to construct test frames."""
    return request.param


def _assert_filter(
    constraint: DiscreteFilteringConstraint,
    data: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
    backend: nw.Implementation,
) -> None:
    """Assert that a constraint keeps exactly the expected rows."""
    frame = nw.from_dicts(data, backend=backend)
    result = constraint(frame.to_native())
    expected = (
        nw.from_dicts(expected_rows, backend=backend) if expected_rows else frame[[]]
    )
    assert_nw_frame_equal(nw.from_native(result, eager_only=True), expected)


def _assert_partial_rejected(
    constraint: DiscreteFilteringConstraint,
    data: Sequence[Mapping[str, object]],
    backend: nw.Implementation,
) -> None:
    """Assert that a constraint rejects a non-evaluable partial frame."""
    assert constraint._can_evaluate(set(data[0])) is False
    frame = nw.from_dicts(data, backend=backend)
    with pytest.raises(ValueError, match="missing from the provided dataframe"):
        constraint(frame.to_native())


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        param((["A", "B"], "=", 8.0, 0.01), {}, id="positional"),
        param((["A", "B"], "="), {"rhs": 8.0, "tolerance": 0.01}, id="mixed"),
        param(
            (),
            {"parameters": ["A", "B"], "operator": "=", "rhs": 8.0, "tolerance": 0.01},
            id="keyword",
        ),
    ],
)
def test_product_constructor(args, kwargs):
    """Modern Product call styles produce the same constraint."""
    result = DiscreteProductConstraint(*args, **kwargs)
    assert result == DiscreteProductConstraint(
        ["A", "B"], operator="=", rhs=8.0, tolerance=0.01
    )
    assert tuple(signature(DiscreteProductConstraint).parameters) == (
        "parameters",
        "operator",
        "rhs",
        "tolerance",
        "exclude",
    )


def test_simple_dependency():
    """Dependency constraint filtering is not implemented yet."""
    raise NotImplementedError


_EXCLUSION_DATA = [
    {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
    {"temperature": 1.0, "pressure": 0.0, "solvent": "ethanol"},
    {"temperature": 0.0, "pressure": 1.0, "solvent": "methanol"},
    {"temperature": 0.0, "pressure": 0.0, "solvent": "acetone"},
]


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
def test_exclusion(combiner, exclude, expected_rows, backend):
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
    _assert_filter(constraint, _EXCLUSION_DATA, expected_rows, backend)


_SELECTION_PARTIAL_DATA = (
    nw.from_dicts(_EXCLUSION_DATA, backend=nw.Implementation.POLARS)
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
    _assert_filter(constraint, _SELECTION_PARTIAL_DATA, expected_rows, backend)


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
    _assert_partial_rejected(constraint, _SELECTION_PARTIAL_DATA, backend)


_LINEAR_DATA = [
    {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
    {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
    {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
    {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
    {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
    {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
]


@pytest.mark.parametrize(
    ("coefficients", "operator", "rhs", "tolerance", "exclude", "expected_rows"),
    [
        param(
            None,
            "<",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
            ],
            id="operator-lt",
        ),
        param(
            None,
            "<=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
            ],
            id="operator-le",
        ),
        param(
            None,
            "=",
            2.0,
            None,
            False,
            [{"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"}],
            id="operator-eq",
        ),
        param(
            None,
            "==",
            2.0,
            None,
            False,
            [{"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"}],
            id="operator-eq-alias",
        ),
        param(
            None,
            "!=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="operator-ne",
        ),
        param(
            None,
            ">",
            2.0,
            None,
            False,
            [
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="operator-gt",
        ),
        param(
            None,
            ">=",
            2.0,
            None,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="operator-ge",
        ),
        param(
            (2.0, 1.0),
            "<=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
            ],
            id="coefficients-scaled",
        ),
        param(
            (1.0, -1.0),
            "<=",
            0.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
            ],
            id="coefficients-negative",
        ),
        param(
            (2.0, 1.0),
            "=",
            3.0,
            None,
            False,
            [{"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"}],
            id="coefficients-equality",
        ),
        param(
            None,
            "=",
            2.5,
            0.5,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
            ],
            id="tolerance",
        ),
        param(
            None,
            "<=",
            2.0,
            None,
            True,
            [
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 5.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 10.0, "amount_2": 0.0, "solvent": "hexane"},
            ],
            id="exclude",
        ),
    ],
)
def test_linear_constraint(
    coefficients,
    operator,
    rhs,
    tolerance,
    exclude,
    expected_rows,
    backend,
):
    """Cover all operator, coefficient, tolerance, and exclusion regimes."""
    kwargs = {} if coefficients is None else {"coefficients": coefficients}
    constraint = DiscreteLinearConstraint(
        parameters=["amount_1", "amount_2"],
        operator=operator,
        rhs=rhs,
        tolerance=tolerance,
        exclude=exclude,
        **kwargs,
    )
    _assert_filter(constraint, _LINEAR_DATA, expected_rows, backend)


def test_linear_constraint_partial_rejected(backend):
    """Linear constraints defer until all parameters are available."""
    constraint = DiscreteLinearConstraint(
        parameters=["amount_1", "amount_2"],
        operator="<=",
        rhs=2.0,
    )
    _assert_partial_rejected(constraint, [{"amount_1": 1.0}], backend)


_PRODUCT_DATA = [
    {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
    {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
    {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
    {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
    {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
    {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
]


@pytest.mark.parametrize(
    ("operator", "rhs", "tolerance", "exclude", "expected_rows"),
    [
        param(
            "<",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
            ],
            id="operator-lt",
        ),
        param(
            "<=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
            ],
            id="operator-le",
        ),
        param(
            "=",
            2.0,
            None,
            False,
            [{"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"}],
            id="operator-eq",
        ),
        param(
            "==",
            2.0,
            None,
            False,
            [{"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"}],
            id="operator-eq-alias",
        ),
        param(
            "!=",
            2.0,
            None,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "solvent": "water"},
                {"amount_1": 1.0, "amount_2": 0.0, "solvent": "ethanol"},
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="operator-ne",
        ),
        param(
            ">",
            2.0,
            None,
            False,
            [
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="operator-gt",
        ),
        param(
            ">=",
            2.0,
            None,
            False,
            [
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="operator-ge",
        ),
        param(
            "=",
            1.5,
            0.5,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "solvent": "methanol"},
                {"amount_1": 2.0, "amount_2": 1.0, "solvent": "acetone"},
            ],
            id="tolerance",
        ),
        param(
            "<=",
            2.0,
            None,
            True,
            [
                {"amount_1": 2.0, "amount_2": 2.0, "solvent": "water"},
                {"amount_1": 5.0, "amount_2": 2.0, "solvent": "hexane"},
            ],
            id="exclude",
        ),
    ],
)
def test_product_constraint(operator, rhs, tolerance, exclude, expected_rows, backend):
    """Cover all product-constraint comparison regimes."""
    constraint = DiscreteProductConstraint(
        parameters=["amount_1", "amount_2"],
        operator=operator,
        rhs=rhs,
        tolerance=tolerance,
        exclude=exclude,
    )
    _assert_filter(constraint, _PRODUCT_DATA, expected_rows, backend)


def test_product_constraint_partial_rejected(backend):
    """Product constraints defer until all parameters are available."""
    constraint = DiscreteProductConstraint(
        parameters=["amount_1", "amount_2"],
        operator="<=",
        rhs=2.0,
    )
    _assert_partial_rejected(constraint, [{"amount_1": 1.0}], backend)


def test_mixture():
    """Mixture constraint filtering is not implemented yet."""
    raise NotImplementedError


_CUSTOM_DATA = [
    {"temperature": 1.0, "pressure": 1.0, "solvent": "water"},
    {"temperature": 2.0, "pressure": 1.0, "solvent": "ethanol"},
    {"temperature": 1.0, "pressure": 2.0, "solvent": "methanol"},
    {"temperature": 2.0, "pressure": 2.0, "solvent": "acetone"},
]


@pytest.mark.parametrize(
    ("exclude", "expected_rows"),
    [
        param(
            False,
            [
                {"temperature": 2.0, "pressure": 1.0, "solvent": "ethanol"},
                {"temperature": 1.0, "pressure": 2.0, "solvent": "methanol"},
                {"temperature": 2.0, "pressure": 2.0, "solvent": "acetone"},
            ],
            id="keep",
        ),
        param(
            True,
            [{"temperature": 1.0, "pressure": 1.0, "solvent": "water"}],
            id="exclude",
        ),
    ],
)
def test_custom_constraint(exclude, expected_rows, backend):
    """Custom constraints apply a backend-native validator with optional inversion."""

    def validator(df):
        nw_df = nw.from_native(df, eager_only=True)
        assert nw_df.implementation == backend
        assert set(nw_df.columns) == {"temperature", "pressure"}
        return (nw_df["temperature"] + nw_df["pressure"] >= 3.0).to_native()

    constraint = DiscreteCustomConstraint(
        parameters=["temperature", "pressure"],
        validator=validator,
        exclude=exclude,
    )
    _assert_filter(constraint, _CUSTOM_DATA, expected_rows, backend)


def test_custom_constraint_partial_rejected(backend):
    """Custom constraints defer until all parameters are available."""
    constraint = DiscreteCustomConstraint(
        parameters=["temperature", "pressure"],
        validator=lambda df: df,
    )
    _assert_partial_rejected(constraint, [{"temperature": 1.0}], backend)


_CARDINALITY_DATA = [
    {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
    {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
    {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
    {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 1.0, "group": "d"},
]


@pytest.mark.parametrize(
    ("min_cardinality", "max_cardinality", "exclude", "expected_rows"),
    [
        param(
            1,
            2,
            False,
            [
                {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
            ],
            id="bounds",
        ),
        param(
            1,
            2,
            True,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 1.0, "group": "d"},
            ],
            id="bounds-exclude",
        ),
        param(
            0,
            1,
            False,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
                {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
            ],
            id="upper-only",
        ),
        param(
            2,
            2,
            False,
            [
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
            ],
            id="exact",
        ),
        param(
            1,
            3,
            False,
            [
                {"amount_1": 1.0, "amount_2": 0.0, "amount_3": 0.0, "group": "b"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 0.0, "group": "c"},
                {"amount_1": 1.0, "amount_2": 1.0, "amount_3": 1.0, "group": "d"},
            ],
            id="lower-only",
        ),
        param(
            1,
            3,
            True,
            [
                {"amount_1": 0.0, "amount_2": 0.0, "amount_3": 0.0, "group": "a"},
            ],
            id="lower-only-exclude",
        ),
    ],
)
def test_cardinality_constraint(
    min_cardinality, max_cardinality, exclude, expected_rows, backend
):
    """Cardinality constraints cover bound and exclusion regimes."""
    constraint = DiscreteCardinalityConstraint(
        parameters=["amount_1", "amount_2", "amount_3"],
        min_cardinality=min_cardinality,
        max_cardinality=max_cardinality,
        exclude=exclude,
    )
    _assert_filter(constraint, _CARDINALITY_DATA, expected_rows, backend)


_CARDINALITY_PARTIAL_DATA = (
    nw.from_dicts(_CARDINALITY_DATA, backend=nw.Implementation.POLARS)
    .select("amount_1", "amount_2", "group")
    .rows(named=True)
)


def test_cardinality_constraint_partial_accepted(backend):
    """Max-only cardinality filtering is valid on a non-empty subset."""
    constraint = DiscreteCardinalityConstraint(
        parameters=["amount_1", "amount_2", "amount_3"],
        max_cardinality=1,
    )
    _assert_filter(
        constraint,
        _CARDINALITY_PARTIAL_DATA,
        [
            {"amount_1": 0.0, "amount_2": 0.0, "group": "a"},
            {"amount_1": 1.0, "amount_2": 0.0, "group": "b"},
        ],
        backend,
    )


def test_cardinality_constraint_partial_rejected(backend):
    """Excluded cardinality constraints require all parameters."""
    constraint = DiscreteCardinalityConstraint(
        parameters=["amount_1", "amount_2", "amount_3"],
        min_cardinality=2,
        exclude=True,
    )
    _assert_partial_rejected(constraint, _CARDINALITY_PARTIAL_DATA, backend)
