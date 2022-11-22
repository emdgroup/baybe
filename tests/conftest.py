"""
PyTest configuration
"""
import pytest


@pytest.fixture(scope="function")
def config_basic_1target():
    """
    Config for a basic test using all basic parameter types and 1 target.
    """
    config_dict = {
        "project_name": "Basic 1 Target",
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
                "values": [1, 2, 3, 7],
                "tolerance": 0.3,
            },
        ],
        "objective": {
            "mode": "SINGLE",
            "targets": [
                {
                    "name": "Target_1",
                    "type": "NUM",
                    "mode": "MIN",
                },
            ],
        },
        "strategy": {
            "surrogate_model_cls": "GP",
            "recommender_cls": "UNRESTRICTED_RANKING",
        },
    }

    return config_dict


@pytest.fixture
def mock_substances():
    """
    A set of test substances.
    """
    substances = {
        "Water": "O",
        "THF": "C1CCOC1",
        "DMF": "CN(C)C=O",
        "Hexane": "CCCCCC",
        "Ethanol": "CCO",
    }

    return substances


@pytest.fixture
def n_iterations():
    """
    Number if iterations ran in tests.
    """
    return 3


@pytest.mark.parametrize("grid_points", [5, 8])
@pytest.fixture()
def n_grid_points(grid_points):
    """
    Number of grid points used in e.g. the mixture tests. Test an even number
    (5 grid points will cause 4 sections) and a number that causes division into
    numbers that have no perfect floating point representation
    (8 grid points will cause 7 sections).
    """
    return grid_points


@pytest.fixture(scope="session")
def good_reference_values():
    """
    Define some good reference values which are used by the utility function to
    generate fake good results.
    """
    return {"Categorical_1": ["B"], "Categorical_2": ["OK"]}
