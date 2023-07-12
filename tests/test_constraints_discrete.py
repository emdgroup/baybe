"""
Test for imposing dependency constraints
"""
import math

import pytest


@pytest.mark.parametrize(
    "parameter_names",
    [["Switch_1", "Switch_2", "Fraction_1", "Solvent_1", "Frame_A", "Frame_B"]],
)
@pytest.mark.parametrize("constraint_names", [["Constraint_1"]])
def test_simple_dependency_variant1(
    baybe, n_grid_points, mock_substances, mock_categories
):
    """
    Test declaring dependencies by declaring them in a single constraints entry.
    """
    # Number entries with both switches on
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "on")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "right")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances) * len(
        mock_categories
    ) * len(mock_categories)

    # Number entries with Switch_1 off
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "off")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "right")
    ).sum()
    assert num_entries == len(mock_categories) * len(mock_categories)

    # Number entries with both switches on
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "on")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "left")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances)

    # Number entries with both switches on
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "off")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "left")
    ).sum()
    assert num_entries == 1


@pytest.mark.parametrize(
    "parameter_names",
    [["Switch_1", "Switch_2", "Fraction_1", "Solvent_1", "Frame_A", "Frame_B"]],
)
@pytest.mark.parametrize("constraint_names", [["Constraint_2", "Constraint_3"]])
def test_simple_dependency_variant2(
    baybe, n_grid_points, mock_substances, mock_categories
):
    """
    Test declaring dependencies by declaring them in separate constraint entries.
    """
    # Number entries with both switches on
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "on")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "right")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances) * len(
        mock_categories
    ) * len(mock_categories)

    # Number entries with Switch 1 off
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "off")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "right")
    ).sum()
    assert num_entries == len(mock_categories) * len(mock_categories)

    # Number entries with both switches on
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "on")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "left")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances)

    # Number entries with both switches on
    num_entries = (
        (baybe.searchspace.discrete.exp_rep["Switch_1"] == "off")
        & (baybe.searchspace.discrete.exp_rep["Switch_2"] == "left")
    ).sum()
    assert num_entries == 1


@pytest.mark.parametrize(
    "parameter_names",
    [["Solvent_1", "SomeSetting", "Temperature", "Pressure"]],
)
@pytest.mark.parametrize(
    "constraint_names", [["Constraint_4", "Constraint_5", "Constraint_6"]]
)
def test_exclusion(baybe, mock_substances):
    """
    Tests exclusion constraint
    """
    # Number of entries with either first/second substance and a temperature above 151
    num_entries = (
        baybe.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 151)
        & baybe.searchspace.discrete.exp_rep["Solvent_1"].apply(
            lambda x: x in list(mock_substances)[:2]
        )
    ).sum()
    assert num_entries == 0

    # Number of entries with either last / second last substance and a pressure above 5
    num_entries = (
        baybe.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 5)
        & baybe.searchspace.discrete.exp_rep["Solvent_1"].apply(
            lambda x: x in list(mock_substances)[-2:]
        )
    ).sum()
    assert num_entries == 0

    # Number of entries with pressure below 3 and temperature above 120
    num_entries = (
        baybe.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x < 3)
        & baybe.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 120)
    ).sum()
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_8"]])
def test_prodsum1(baybe):
    """
    Tests sum constraint.
    """
    # Number of entries with 1,2-sum above 150
    num_entries = (
        baybe.searchspace.discrete.exp_rep[["Fraction_1", "Fraction_2"]].sum(axis=1)
        > 150.0
    ).sum()
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_9"]])
def test_prodsum2(baybe):
    """
    Tests product constrain.
    """
    # Number of entries with product under 30
    num_entries = (
        baybe.searchspace.discrete.exp_rep[["Fraction_1", "Fraction_2"]].prod(axis=1)
        < 30
    ).sum()
    assert num_entries == 0


@pytest.mark.parametrize("parameter_names", [["Fraction_1", "Fraction_2"]])
@pytest.mark.parametrize("constraint_names", [["Constraint_10"]])
def test_prodsum3(baybe):
    """
    Tests exact sum constraint.
    """
    # Number of entries with sum unequal to 100
    num_entries = (
        baybe.searchspace.discrete.exp_rep[["Fraction_1", "Fraction_2"]]
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
def test_mixture(baybe, n_grid_points, mock_substances):
    """
    Tests various constraints in a mixture use case.
    """
    # Number of searchspace entries where fractions do not sum to 100.0
    num_entries = (
        baybe.searchspace.discrete.exp_rep[["Fraction_1", "Fraction_2", "Fraction_3"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum()
    )
    assert num_entries == 0

    # Number of searchspace entries that have duplicate solvent labels
    num_entries = (
        baybe.searchspace.discrete.exp_rep[["Solvent_1", "Solvent_2", "Solvent_3"]]
        .nunique(axis=1)
        .ne(3)
        .sum()
    )
    assert num_entries == 0

    # Number of searchspace entries with permutation-invariant combinations
    num_entries = (
        baybe.searchspace.discrete.exp_rep[["Solvent_1", "Solvent_2", "Solvent_3"]]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(
            baybe.searchspace.discrete.exp_rep[
                ["Fraction_1", "Fraction_2", "Fraction_3"]
            ]
        )
        .duplicated()
        .sum()
    )
    assert num_entries == 0

    # Number of unique 1-solvent entries
    num_entries = (
        (
            baybe.searchspace.discrete.exp_rep[
                ["Fraction_1", "Fraction_2", "Fraction_3"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(2)
        .sum()
    )
    assert num_entries == math.comb(len(mock_substances), 1) * 1

    # Number of unique 2-solvent entries
    num_entries = (
        (
            baybe.searchspace.discrete.exp_rep[
                ["Fraction_1", "Fraction_2", "Fraction_3"]
            ]
            == 0.0
        )
        .sum(axis=1)
        .eq(1)
        .sum()
    )
    assert num_entries == math.comb(len(mock_substances), 2) * (n_grid_points - 2)

    # Number of unique 3-solvent entries
    num_entries = (
        (
            baybe.searchspace.discrete.exp_rep[
                ["Fraction_1", "Fraction_2", "Fraction_3"]
            ]
            == 0.0
        )
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
    [["Solvent_1", "SomeSetting", "Temperature", "Pressure"]],
)
@pytest.mark.parametrize("constraint_names", [["Constraint_13"]])
def test_custom(baybe):
    """
    Tests custom constraint (uses config from exclude test)
    """
    num_entries = (
        baybe.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 5)
        & baybe.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 120)
        & baybe.searchspace.discrete.exp_rep["Solvent_1"].eq("water")
    ).sum()
    assert num_entries == 0

    (
        baybe.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 3)
        & baybe.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 180)
        & baybe.searchspace.discrete.exp_rep["Solvent_1"].eq("water")
    ).sum()
    assert num_entries == 0

    (
        baybe.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 3)
        & baybe.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x < 150)
        & baybe.searchspace.discrete.exp_rep["Solvent_1"].eq("C3")
    ).sum()
    assert num_entries == 0
