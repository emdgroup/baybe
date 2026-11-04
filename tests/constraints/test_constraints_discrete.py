"""Test for imposing discrete constraints."""

import math
import os
from collections.abc import Mapping, Sequence
from inspect import signature

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from narwhals.testing import assert_frame_equal as assert_nw_frame_equal
from pandas.testing import assert_frame_equal
from pytest import param

from baybe.constraints.base import DiscreteFilteringConstraint
from baybe.constraints.conditions import _THRESHOLD_OPERATORS, ThresholdCondition
from baybe.constraints.discrete import (
    DiscreteLinearConstraint,
    DiscreteProductConstraint,
    DiscreteSelectionConstraint,
)
from baybe.parameters import NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.settings import Settings

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


def _frame(data: dict[str, list], backend: nw.Implementation) -> nw.DataFrame:
    """Create a Narwhals frame from column data for the given backend."""
    return nw.from_dict(data, backend=backend)


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


@pytest.fixture(
    params=[5, pytest.param(8, marks=pytest.mark.slow)],
    name="n_grid_points",
    ids=["g5", "g8"],
)
def fixture_n_grid_points(request):
    """Number of grid points used in e.g. the mixture tests.

    Test an even number (5 grid points will cause 4 sections) and a number that causes
    division into numbers that have no perfect floating point representation (8 grid
    points will cause 7 sections).
    """
    return request.param


@pytest.mark.parametrize(
    "parameter_names",
    [["Switch_1", "Switch_2", "Fraction_1", "Solvent_1", "Frame_A", "Frame_B"]],
)
@pytest.mark.parametrize("constraint_names", [["Constraint_1"]])
def test_simple_dependency(campaign, n_grid_points, mock_substances, mock_categories):
    """Test declaring dependencies by declaring them in a single constraints entry."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number entries with both switches on
    num_entries = (
        (candidates["Switch_1"] == "on") & (candidates["Switch_2"] == "right")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances) * len(
        mock_categories
    ) * len(mock_categories)

    # Number entries with Switch_1 off
    num_entries = (
        (candidates["Switch_1"] == "off") & (candidates["Switch_2"] == "right")
    ).sum()
    assert num_entries == len(mock_categories) * len(mock_categories)

    # Number entries with both switches on
    num_entries = (
        (candidates["Switch_1"] == "on") & (candidates["Switch_2"] == "left")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances)

    # Number entries with both switches on
    num_entries = (
        (candidates["Switch_1"] == "off") & (candidates["Switch_2"] == "left")
    ).sum()
    assert num_entries == 1


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


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_9"]])
def test_prodsum2(campaign):
    """Tests product constrain."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number of entries with product under 30
    num_entries = (candidates[["Fraction_1", "Fraction_2"]].prod(axis=1) < 30).sum()
    assert num_entries == 0


@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Solvent_2", "Solvent_3", "Fraction_1", "Fraction_2", "Fraction_3"]],
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_7", "Constraint_11", "Constraint_12"]]
)
def test_mixture(campaign, n_grid_points, mock_substances):
    """Tests various constraints in a mixture use case."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number of searchspace entries where fractions do not sum to 100.0
    num_entries = (
        candidates[["Fraction_1", "Fraction_2", "Fraction_3"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum()
    )
    assert num_entries == 0

    # Number of searchspace entries that have duplicate solvent labels
    num_entries = (
        candidates[["Solvent_1", "Solvent_2", "Solvent_3"]].nunique(axis=1).ne(3).sum()
    )
    assert num_entries == 0

    # Number of searchspace entries with permutation-invariant combinations
    num_entries = (
        candidates[["Solvent_1", "Solvent_2", "Solvent_3"]]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(candidates[["Fraction_1", "Fraction_2", "Fraction_3"]])
        .duplicated()
        .sum()
    )
    assert num_entries == 0

    # Number of unique 1-solvent entries
    num_entries = (
        (candidates[["Fraction_1", "Fraction_2", "Fraction_3"]] == 0.0)
        .sum(axis=1)
        .eq(2)
        .sum()
    )
    assert num_entries == math.comb(len(mock_substances), 1) * 1

    # Number of unique 2-solvent entries
    num_entries = (
        (candidates[["Fraction_1", "Fraction_2", "Fraction_3"]] == 0.0)
        .sum(axis=1)
        .eq(1)
        .sum()
    )
    assert num_entries == math.comb(len(mock_substances), 2) * (n_grid_points - 2)

    # Number of unique 3-solvent entries
    num_entries = (
        (candidates[["Fraction_1", "Fraction_2", "Fraction_3"]] == 0.0)
        .sum(axis=1)
        .eq(0)
        .sum()
    )
    assert (
        num_entries
        == math.comb(len(mock_substances), 3)
        * ((n_grid_points - 3) * (n_grid_points - 2))
        // 2
    )


@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Some_Setting", "Temperature", "Pressure"]],
)
@pytest.mark.parametrize("constraint_names", [["Constraint_13"]])
def test_custom(campaign):
    """Tests custom constraint (uses config from exclude test)."""
    candidates = campaign.searchspace.discrete.get_candidates()

    num_entries = (
        candidates["Pressure"].apply(lambda x: x > 5)
        & candidates["Temperature"].apply(lambda x: x > 120)
        & candidates["Solvent_1"].eq("water")
    ).sum()
    assert num_entries == 0

    (
        candidates["Pressure"].apply(lambda x: x > 3)
        & candidates["Temperature"].apply(lambda x: x > 180)
        & candidates["Solvent_1"].eq("C2")
    ).sum()
    assert num_entries == 0

    (
        candidates["Pressure"].apply(lambda x: x > 3)
        & candidates["Temperature"].apply(lambda x: x < 150)
        & candidates["Solvent_1"].eq("C3")
    ).sum()
    assert num_entries == 0


@pytest.mark.parametrize(
    "parameter_names",
    [["Some_Setting", "Fraction_1", "Fraction_2", "Fraction_3"]],
)
@pytest.mark.parametrize("constraint_names", [["Constraint_14"]])
def test_cardinality(campaign):
    """Test discrete cardinality constraint."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number of non-zeros
    non_zeros = (candidates[["Fraction_1", "Fraction_2", "Fraction_3"]] != 0.0).sum(
        axis=1
    )

    # number of non-zeros fulfills cardinality
    min_cardinality = 1
    max_cardinality = 2
    assert non_zeros.between(min_cardinality, max_cardinality).all()


@pytest.mark.parametrize(
    ("combiner", "exclude", "partial_ok"),
    [
        param("AND", False, True, id="AND-keep"),
        param("AND", True, False, id="AND-exclude"),
        param("OR", False, False, id="OR-keep"),
        param("OR", True, True, id="OR-exclude"),
        param("XOR", False, False, id="XOR-keep"),
        param("XOR", True, False, id="XOR-exclude"),
    ],
)
def test_filtering_partial_evaluation(combiner, exclude, partial_ok):
    """Partial evaluation is only allowed when the drop decision cannot reverse.

    In particular, an ``XOR`` combination must never be evaluated partially, since
    its result can flip as further operands become available.
    """
    constraint = DiscreteSelectionConstraint(
        parameters=["A", "B"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        combiner=combiner,
        exclude=exclude,
    )
    # Only a subset of the involved parameters is available
    assert constraint._can_evaluate({"A"}) is partial_ok
    # All parameters available -> always evaluable
    assert constraint._can_evaluate({"A", "B"}) is True


def test_filtering_xor_partial_does_not_drop_valid_rows():
    """The builder defers XOR filtering until every operand is available."""
    constraint = DiscreteSelectionConstraint(
        parameters=["A", "B"],
        conditions=[
            ThresholdCondition(threshold=0.0, operator=">"),
            ThresholdCondition(threshold=0.0, operator=">"),
        ],
        combiner="XOR",
    )

    # Row where only "A" is known so far; "B" is still missing.
    partial_df = pd.DataFrame({"A": [1.0, 0.0]})
    # The constraint is not callable yet, so incremental construction defers it.
    assert constraint._can_evaluate(set(partial_df.columns)) is False
    with pytest.raises(ValueError, match="missing from the provided dataframe"):
        constraint(partial_df)

    # Once both columns are present, XOR keeps rows where exactly one holds.
    full_df = pd.DataFrame({"A": [1.0, 1.0, 0.0], "B": [0.0, 1.0, 0.0]})
    filtered = constraint(full_df)
    # Rows 1 (both > 0) and 2 (neither > 0) violate XOR; row 0 is kept.
    assert list(filtered.index) == [0]


_DF = pd.DataFrame(
    {
        "a": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
        "b": [1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0],
    }
)
"""The unconstrained parameter product used for the threshold operator tests."""

_PARAMETERS = [
    NumericalDiscreteParameter(name, values=_DF[name].unique()) for name in _DF
]
"""The parameters spanning the unconstrained product."""

_PRODUCT_ROWS = {
    "<": [0, 1, 3],
    "<=": [0, 1, 2, 3, 4, 6],
    "=": [2, 4, 6],
    "==": [2, 4, 6],
    "!=": [0, 1, 3, 5, 7, 8],
    ">": [5, 7, 8],
    ">=": [2, 4, 5, 6, 7, 8],
}
"""The rows of ``_DF`` whose column product satisfies the operator against 4.0."""

_SELECTION_ROWS = {
    "<": [0, 1, 2],
    "<=": [0, 1, 2, 3, 4, 5],
    "=": [3, 4, 5],
    "==": [3, 4, 5],
    "!=": [0, 1, 2, 6, 7, 8],
    ">": [6, 7, 8],
    ">=": [3, 4, 5, 6, 7, 8],
}
"""The rows of ``_DF`` whose column ``a`` satisfies the operator against 2.0."""

_THRESHOLD_CASES = [
    param(
        DiscreteProductConstraint(parameters=["a", "b"], operator=operator, rhs=4.0),
        rows,
        id=f"product-{operator}",
    )
    for operator, rows in _PRODUCT_ROWS.items()
] + [
    param(
        DiscreteSelectionConstraint(
            parameters=["a"], conditions=[ThresholdCondition(2.0, operator)]
        ),
        rows,
        id=f"selection-{operator}",
    )
    for operator, rows in _SELECTION_ROWS.items()
]
"""Threshold constraints paired with the rows of ``_DF`` they are expected to keep."""


def test_threshold_rows_completeness():
    """The expected filtering results cover all available threshold operators."""
    tables = (_PRODUCT_ROWS, _SELECTION_ROWS)
    assert all(set(table) == set(_THRESHOLD_OPERATORS) for table in tables)


@pytest.mark.parametrize(
    "use_polars",
    [
        param(False, id="pandas"),
        param(True, id="polars"),
    ],
)
@pytest.mark.parametrize(("constraint", "expected_rows"), _THRESHOLD_CASES)
def test_threshold_operators(constraint, expected_rows, use_polars):
    """Threshold constraints filter correctly for all operators and backends."""
    with Settings(use_polars_for_constraints=use_polars):
        searchspace = SearchSpace.from_product(_PARAMETERS, [constraint])

    assert_frame_equal(
        searchspace.discrete.get_candidates()
        .sort_values(["a", "b"])
        .reset_index(drop=True),
        _DF.loc[expected_rows].reset_index(drop=True),
    )
