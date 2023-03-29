"""
Test for imposing dependency constraints
"""
import math

import numpy as np
import pandas as pd
import pytest

from baybe.core import BayBE


@pytest.fixture(name="config_discrete_1target")
def fixture_config_discrete_1target():
    """
    Config for a basic test using all basic parameter types and 1 target.
    """
    config_dict = {
        "project_name": "Discrete Space 1 Target",
        "random_seed": 1337,
        "allow_repeated_recommendations": False,
        "allow_recommending_already_measured": False,
        "numerical_measurements_must_be_within_tolerance": True,
        "parameters": [
            {
                "name": "Categorical_1",
                "type": "CAT",
                "values": ["A", "B", "C"],
                "encoding": "OHE",
            },
            {
                "name": "Categorical_2",
                "type": "CAT",
                "values": ["bad", "OK", "good"],
                "encoding": "INT",
            },
            {
                "name": "Num_disc_1",
                "type": "NUM_DISCRETE",
                "values": [1, 2, 7],
                "tolerance": 0.3,
            },
        ],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
            ],
        },
        "strategy": {
            "surrogate_model_cls": "GP",
            "recommender_cls": "UNRESTRICTED_RANKING",
        },
    }

    return config_dict


@pytest.fixture(name="config_constraints_dependency")
def fixture_config_constraints_dependency(
    n_grid_points, mock_substances, mock_categories
):
    """
    Config for a use case with dependency constraints.
    """
    config_dict = {
        "project_name": "Project with switches and dependencies",
        "allow_repeated_recommendations": False,
        "allow_recommending_already_measured": False,
        "numerical_measurements_must_be_within_tolerance": True,
        "parameters": [
            {
                "name": "Switch1",
                "type": "CAT",
                "values": ["on", "off"],
            },
            {
                "name": "Switch2",
                "type": "CAT",
                "values": ["left", "right"],
            },
            {
                "name": "Fraction1",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 100, n_grid_points)),
                "tolerance": 0.2,
            },
            {
                "name": "Solvent1",
                "type": "SUBSTANCE",
                "data": mock_substances,
            },
            {
                "name": "FrameA",
                "type": "CAT",
                "values": mock_categories,
            },
            {
                "name": "FrameB",
                "type": "CAT",
                "values": mock_categories,
            },
        ],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
            ],
        },
    }
    return config_dict


@pytest.fixture(name="config_constraints_exclude")
def fixture_config_constraints_exclude(n_grid_points, mock_substances, mock_categories):
    """
    Config for a use case with exclusion constraints.
    """
    config_dict = {
        "project_name": "Project with substances and exclusion constraints",
        "allow_repeated_recommendations": False,
        "allow_recommending_already_measured": True,
        "numerical_measurements_must_be_within_tolerance": True,
        "parameters": [
            {
                "name": "Solvent",
                "type": "SUBSTANCE",
                "data": mock_substances,
            },
            {
                "name": "SomeSetting",
                "type": "CAT",
                "values": mock_categories,
                "encoding": "INT",
            },
            {
                "name": "Temperature",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(100, 200, n_grid_points)),
            },
            {
                "name": "Pressure",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 6, n_grid_points)),
            },
        ],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
            ],
        },
    }
    return config_dict


@pytest.fixture(name="config_constraints_prodsum")
def fixture_config_constraints_prodsum(n_grid_points):
    """
    Config with some numerical parameters for a use case with product and sum
    constraints.
    """
    config_dict = {
        "project_name": "Project with several numerical parameters",
        "allow_repeated_recommendations": False,
        "allow_recommending_already_measured": True,
        "numerical_measurements_must_be_within_tolerance": True,
        "parameters": [
            {
                "name": "Solvent",
                "type": "SUBSTANCE",
                "data": {
                    "water": "O",
                    "C1": "C",
                    "C2": "CC",
                    "C3": "CCC",
                },
                "encoding": "RDKIT",
            },
            {
                "name": "SomeSetting",
                "type": "CAT",
                "values": ["slow", "normal", "fast"],
                "encoding": "INT",
            },
            {
                "name": "NumParameter1",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 100, n_grid_points)),
                "tolerance": 0.5,
            },
            {
                "name": "NumParameter2",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 100, n_grid_points)),
                "tolerance": 0.5,
            },
        ],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
            ],
        },
    }
    return config_dict


@pytest.fixture(name="config_constraints_mixture")
def fixture_config_constraints_mixture(n_grid_points, mock_substances):
    """
    Config for a mixture use case (3 solvents).
    """
    config_dict = {
        "project_name": "Exclusion Constraints Test (Discrete)",
        "allow_repeated_recommendations": False,
        "allow_recommending_already_measured": True,
        "numerical_measurements_must_be_within_tolerance": True,
        "parameters": [
            {
                "name": "Solvent1",
                "type": "SUBSTANCE",
                "data": mock_substances,
                "encoding": "MORDRED",
            },
            {
                "name": "Solvent2",
                "type": "SUBSTANCE",
                "data": mock_substances,
                "encoding": "MORDRED",
            },
            {
                "name": "Solvent3",
                "type": "SUBSTANCE",
                "data": mock_substances,
                "encoding": "MORDRED",
            },
            {
                "name": "Fraction1",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 100, n_grid_points)),
                "tolerance": 0.2,
            },
            {
                "name": "Fraction2",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 100, n_grid_points)),
                "tolerance": 0.2,
            },
            {
                "name": "Fraction3",
                "type": "NUM_DISCRETE",
                "values": list(np.linspace(0, 100, n_grid_points)),
                "tolerance": 0.2,
            },
        ],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MAX",
                },
            ],
        },
    }

    return config_dict


def test_simple_dependency_variant1(
    config_constraints_dependency, n_grid_points, mock_substances, mock_categories
):
    """
    Test declaring dependencies by declaring them in a single constraints entry.
    """
    config_constraints_dependency["constraints"] = [
        {
            "type": "DEPENDENCIES",
            "parameters": ["Switch1", "Switch2"],
            "conditions": [
                {
                    "type": "SUBSELECTION",
                    "selection": ["on"],
                },
                {
                    "type": "SUBSELECTION",
                    "selection": ["right"],
                },
            ],
            "affected_parameters": [
                ["Solvent1", "Fraction1"],
                ["FrameA", "FrameB"],
            ],
        },
    ]
    baybe_obj = BayBE.from_dict(config_constraints_dependency)

    # Number entries with both switches on
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "on")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "right")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances) * len(
        mock_categories
    ) * len(mock_categories)

    # Number entries with Switch1 off
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "off")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "right")
    ).sum()
    assert num_entries == len(mock_categories) * len(mock_categories)

    # Number entries with both switches on
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "on")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "left")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances)

    # Number entries with both switches on
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "off")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "left")
    ).sum()
    assert num_entries == 1


def test_simple_dependency_variant2(
    config_constraints_dependency, n_grid_points, mock_substances, mock_categories
):
    """
    Test declaring dependencies by declaring them in separate constraint entries.
    """
    config_constraints_dependency["constraints"] = [
        {
            "type": "DEPENDENCIES",
            "parameters": ["Switch1"],
            "conditions": [
                {
                    "type": "SUBSELECTION",
                    "selection": ["on"],
                },
            ],
            "affected_parameters": [
                ["Solvent1", "Fraction1"],
            ],
        },
        {
            "type": "DEPENDENCIES",
            "parameters": ["Switch2"],
            "conditions": [
                {
                    "type": "SUBSELECTION",
                    "selection": ["right"],
                },
            ],
            "affected_parameters": [
                ["FrameA", "FrameB"],
            ],
        },
    ]
    baybe_obj = BayBE.from_dict(config_constraints_dependency)

    # Number entries with both switches on
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "on")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "right")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances) * len(
        mock_categories
    ) * len(mock_categories)

    # Number entries with Switch1 off
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "off")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "right")
    ).sum()
    assert num_entries == len(mock_categories) * len(mock_categories)

    # Number entries with both switches on
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "on")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "left")
    ).sum()
    assert num_entries == n_grid_points * len(mock_substances)

    # Number entries with both switches on
    num_entries = (
        (baybe_obj.searchspace.discrete.exp_rep["Switch1"] == "off")
        & (baybe_obj.searchspace.discrete.exp_rep["Switch2"] == "left")
    ).sum()
    assert num_entries == 1


def test_exclusion(config_constraints_exclude, mock_substances):
    """
    Tests exclusion constraint
    """
    config_constraints_exclude["constraints"] = [
        # This constraint simulates a situation where substances 1 and 2 are not
        # compatible with temperatures > 154 and should thus be excluded for those temps
        {
            "type": "EXCLUDE",
            "parameters": ["Temperature", "Solvent"],
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 151,
                    "operator": ">",
                },
                {
                    "type": "SUBSELECTION",
                    "selection": list(mock_substances)[:2],
                },
            ],
        },
        # This constraint simulates a situation where last and second last substances
        # are not compatible with pressures >= 5 and should thus be excluded
        {
            "type": "EXCLUDE",
            "parameters": ["Pressure", "Solvent"],
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 5,
                    "operator": ">",
                },
                {
                    "type": "SUBSELECTION",
                    "selection": list(mock_substances)[-2:],
                },
            ],
        },
        # This constraint simulates a situation where pressures below 3 should never be
        # combined with temperatures above 120
        {
            "type": "EXCLUDE",
            "parameters": ["Pressure", "Temperature"],
            "combiner": "AND",
            "conditions": [
                {
                    "type": "THRESHOLD",
                    "threshold": 3.0,
                    "operator": "<",
                },
                {
                    "type": "THRESHOLD",
                    "threshold": 120.0,
                    "operator": ">",
                },
            ],
        },
    ]
    baybe_obj = BayBE.from_dict(config_constraints_exclude)

    # Number of entries with either first/second substance and a temperature above 151
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 151)
        & baybe_obj.searchspace.discrete.exp_rep["Solvent"].apply(
            lambda x: x in list(mock_substances)[:2]
        )
    ).sum()
    assert num_entries == 0

    # Number of entries with either last / second last substance and a pressure above 5
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 5)
        & baybe_obj.searchspace.discrete.exp_rep["Solvent"].apply(
            lambda x: x in list(mock_substances)[-2:]
        )
    ).sum()
    assert num_entries == 0

    # Number of entries with pressure below 3 and temperature above 120
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x < 3)
        & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 120)
    ).sum()
    assert num_entries == 0


def test_prodsum1(config_constraints_prodsum):
    """
    Tests sum constraint.
    """
    config_constraints_prodsum["constraints"] = [
        {
            "type": "SUM",
            "parameters": ["NumParameter1", "NumParameter2"],
            "condition": {
                "threshold": 150.0,
                "operator": "<=",
            },
        }
    ]

    baybe_obj = BayBE.from_dict(config_constraints_prodsum)

    # Number of entries with 1,2-sum above 150
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep[["NumParameter1", "NumParameter2"]].sum(
            axis=1
        )
        > 150.0
    ).sum()
    assert num_entries == 0


def test_prodsum2(config_constraints_prodsum):
    """
    Tests product constrain.
    """
    config_constraints_prodsum["constraints"] = [
        {
            "type": "PRODUCT",
            "parameters": ["NumParameter1", "NumParameter2"],
            "condition": {
                "threshold": 30.0,
                "operator": ">=",
            },
        }
    ]

    baybe_obj = BayBE.from_dict(config_constraints_prodsum)

    # Number of entries with product under 30
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep[["NumParameter1", "NumParameter2"]].prod(
            axis=1
        )
        < 30
    ).sum()
    assert num_entries == 0


def test_prodsum3(config_constraints_prodsum):
    """
    Tests exact sum constraint.
    """
    config_constraints_prodsum["constraints"] = [
        {
            "type": "SUM",
            "parameters": ["NumParameter1", "NumParameter2"],
            "condition": {
                "threshold": 100.0,
                "operator": "=",
                "tolerance": 1.0,
            },
        }
    ]

    baybe_obj = BayBE.from_dict(config_constraints_prodsum)

    # Number of entries with sum unequal to 100
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep[["NumParameter1", "NumParameter2"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum()
    )
    assert num_entries == 0


def test_mixture(config_constraints_mixture, n_grid_points, mock_substances):
    """
    Tests various constraints in a mixture use case.
    """
    config_constraints_mixture["constraints"] = [
        {
            "type": "PERMUTATION_INVARIANCE",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
            "dependencies": {
                "parameters": ["Fraction1", "Fraction2", "Fraction3"],
                "conditions": [
                    {
                        "type": "THRESHOLD",
                        "threshold": 0.0,
                        "operator": ">",
                    },
                    {
                        "type": "THRESHOLD",
                        "threshold": 0.0,
                        "operator": ">",
                    },
                    {
                        # This is just to test whether the specification via
                        # subselection condition also works
                        "type": "SUBSELECTION",
                        "selection": list(np.linspace(0, 100, n_grid_points)[1:]),
                    },
                ],
                "affected_parameters": [
                    ["Solvent1"],
                    ["Solvent2"],
                    ["Solvent3"],
                ],
            },
        },
        {
            "type": "SUM",
            "parameters": ["Fraction1", "Fraction2", "Fraction3"],
            "condition": {
                "threshold": 100.0,
                "operator": "=",
                "tolerance": 0.01,
            },
        },
        {
            "type": "NO_LABEL_DUPLICATES",
            "parameters": ["Solvent1", "Solvent2", "Solvent3"],
        },
    ]

    baybe_obj = BayBE.from_dict(config_constraints_mixture)

    # Number of searchspace entries where fractions do not sum to 100.0
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep[["Fraction1", "Fraction2", "Fraction3"]]
        .sum(axis=1)
        .apply(lambda x: x - 100.0)
        .abs()
        .gt(0.01)
        .sum()
    )
    assert num_entries == 0

    # Number of searchspace entries that have duplicate solvent labels
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep[["Solvent1", "Solvent2", "Solvent3"]]
        .nunique(axis=1)
        .ne(3)
        .sum()
    )
    assert num_entries == 0

    # Number of searchspace entries with permutation-invariant combinations
    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep[["Solvent1", "Solvent2", "Solvent3"]]
        .apply(frozenset, axis=1)
        .to_frame()
        .join(
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
            ]
        )
        .duplicated()
        .sum()
    )
    assert num_entries == 0

    # Number of unique 1-solvent entries
    num_entries = (
        (
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
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
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
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
            baybe_obj.searchspace.discrete.exp_rep[
                ["Fraction1", "Fraction2", "Fraction3"]
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


def test_custom(config_constraints_exclude):
    """
    Tests custom constraint (uses config from exclude test)
    """

    def custom_function(ser: pd.Series) -> bool:
        if ser.Solvent == "water":
            if ser.Temperature > 120 and ser.Pressure > 5:
                return False
            if ser.Temperature > 180 and ser.Pressure > 3:
                return False
        if ser.Solvent == "C3":
            if ser.Temperature < 150 and ser.Pressure > 3:
                return False
        return True

    config_constraints_exclude["constraints"] = [
        # This constraint uses the user-defined function as a valdiator/filter
        {
            "type": "CUSTOM",
            "parameters": ["Pressure", "Solvent", "Temperature"],
            "validator": custom_function,
        },
    ]

    baybe_obj = BayBE.from_dict(config_constraints_exclude)

    num_entries = (
        baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 5)
        & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 120)
        & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("water")
    ).sum()
    assert num_entries == 0

    (
        baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 3)
        & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x > 180)
        & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("water")
    ).sum()
    assert num_entries == 0

    (
        baybe_obj.searchspace.discrete.exp_rep["Pressure"].apply(lambda x: x > 3)
        & baybe_obj.searchspace.discrete.exp_rep["Temperature"].apply(lambda x: x < 150)
        & baybe_obj.searchspace.discrete.exp_rep["Solvent"].eq("C3")
    ).sum()
    assert num_entries == 0
