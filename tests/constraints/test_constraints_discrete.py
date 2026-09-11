"""Test for imposing discrete constraints."""

import itertools
import math

import pandas as pd
import pytest
from pytest import param

from baybe.constraints.conditions import ThresholdCondition
from baybe.constraints.discrete import DiscreteSumConstraint


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


@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "Some_Setting", "Temperature", "Pressure"]],
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_4", "Constraint_5", "Constraint_6"]]
)
def test_exclusion(campaign, mock_substances):
    """Tests exclusion constraint."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number of entries with either first/second substance and a temperature above 151
    num_entries = (
        candidates["Temperature"].apply(lambda x: x > 151)
        & candidates["Solvent_1"].apply(lambda x: x in list(mock_substances)[:2])
    ).sum()
    assert num_entries == 0

    # Number of entries with either last / second last substance and a pressure above 5
    num_entries = (
        candidates["Pressure"].apply(lambda x: x > 5)
        & candidates["Solvent_1"].apply(lambda x: x in list(mock_substances)[-2:])
    ).sum()
    assert num_entries == 0

    # Number of entries with pressure below 3 and temperature above 120
    num_entries = (
        candidates["Pressure"].apply(lambda x: x < 3)
        & candidates["Temperature"].apply(lambda x: x > 120)
    ).sum()
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_8"]])
def test_prodsum1(campaign):
    """Tests sum constraint."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number of entries with 1,2-sum above 150
    num_entries = (candidates[["Fraction_1", "Fraction_2"]].sum(axis=1) > 150.0).sum()
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_9"]])
def test_prodsum2(campaign):
    """Tests product constrain."""
    candidates = campaign.searchspace.discrete.get_candidates()

    # Number of entries with product under 30
    num_entries = (candidates[["Fraction_1", "Fraction_2"]].prod(axis=1) < 30).sum()
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_10"]])
def test_prodsum3(campaign):
    """Tests exact sum constraint."""
    candidates = campaign.searchspace.discrete.get_candidates()
    # Number of entries with sum unequal to 100
    num_entries = (
        candidates[["Fraction_1", "Fraction_2"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum()
    )
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
    ("coefficients", "threshold", "operator", "n_invalid"),
    [
        param(None, 1.0, "<=", 3, id="default"),
        param((1.0, 1.0), 1.0, "<=", 3, id="all-ones"),
        param((2.0, 1.0), 1.0, "<=", 5, id="scaled"),
        param((1.0, -1.0), 0.5, "<=", 1, id="negative"),
        param((1.0, 1.0), 1.0, "=", 6, id="equality"),
    ],
)
def test_sum_constraint_coefficients(coefficients, threshold, operator, n_invalid):
    """DiscreteSumConstraint filters correctly with default and custom coefficients."""
    kwargs = {} if coefficients is None else {"coefficients": coefficients}
    constraint = DiscreteSumConstraint(
        parameters=["A", "B"],
        condition=ThresholdCondition(threshold=threshold, operator=operator),
        **kwargs,
    )
    df = pd.DataFrame(
        list(itertools.product([0.0, 0.5, 1.0], repeat=2)), columns=["A", "B"]
    )
    coeffs = coefficients or (1.0, 1.0)
    weighted = df["A"] * coeffs[0] + df["B"] * coeffs[1]
    expected = df.index[~ThresholdCondition(threshold, operator).evaluate(weighted)]
    assert list(constraint.get_invalid(df)) == list(expected)
    assert len(constraint.get_invalid(df)) == n_invalid
