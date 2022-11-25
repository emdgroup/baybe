"""
PyTest configuration
"""
import numpy as np
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

# All fixture functions have prefix 'fixture_' and explicitly declared name so they
# can be reused by other fixtures, see
# https://docs.pytest.org/en/stable/reference/reference.html#pytest-fixture


# Independent Fixtures
@pytest.fixture(scope="session", params=[2], name="n_iterations", ids=["iter2"])
def fixture_n_iterations(request):
    """
    Number of iterations ran in tests.
    """
    return request.param


@pytest.fixture(
    scope="session", params=[1, 3], name="batch_quantity", ids=["batch1", "batch3"]
)
def fixture_batch_quantity(request):
    """
    Number of recommendations requested per iteration. Testing 1 as edge case and 3
    as a case for >1.
    """
    return request.param


@pytest.fixture(
    scope="session",
    params=[5, 8],
    name="n_grid_points",
    ids=["grid5", "grid8"],
)
def fixture_n_grid_points(request):
    """
    Number of grid points used in e.g. the mixture tests. Test an even number
    (5 grid points will cause 4 sections) and a number that causes division into
    numbers that have no perfect floating point representation
    (8 grid points will cause 7 sections).
    """
    return request.param


@pytest.fixture(scope="session", name="good_reference_values")
def fixture_good_reference_values():
    """
    Define some good reference values which are used by the utility function to
    generate fake good results.
    """
    return {"Categorical_1": ["B"], "Categorical_2": ["OK"]}


@pytest.fixture(name="mock_substances")
def fixture_mock_substances():
    """
    A set of test substances.
    """
    substances = {
        "Water": "O",
        "THF": "C1CCOC1",
        "DMF": "CN(C)C=O",
        "Hexane": "CCCCCC",
    }

    return substances


@pytest.fixture(name="mock_categories")
def fixture_mock_categories():
    """
    A set of mock categories for categorical parameters.
    """
    return ["Type1", "Type2", "Type3"]


# Dependent Fixtures
@pytest.fixture(name="config_basic_1target")
def fixture_config_basic_1target():
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
        # The constrains simulate a situation where we want to mix up to three solvents,
        # but their respective fractions need to sum up to 100. Also, the solvents
        # should never be chosen twice.
        "constraints": [
            {
                # This constraint will affect searchspace creation
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
                # This constraint will only affect searchspace creation
                "type": "SUM",
                "parameters": ["Fraction1", "Fraction2", "Fraction3"],
                "condition": {
                    "threshold": 100.0,
                    "operator": "=",
                    "tolerance": 0.01,
                },
            },
            {
                # This constraint will only affect searchspace creation
                "type": "NO_LABEL_DUPLICATES",
                "parameters": ["Solvent1", "Solvent2", "Solvent3"],
            },
        ],
    }

    return config_dict


@pytest.fixture(name="baybe_object_batch3_iterations2")
def fixture_baybe_object_batch3_iterations2(
    config_basic_1target, good_reference_values
):
    """
    Returns BayBE object that has been run for 2 iterations with mock data.
    """
    config = BayBEConfig(**config_basic_1target)
    baybe_obj = BayBE(config)

    for _ in range(2):
        rec = baybe_obj.recommend(batch_quantity=3)
        add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
        add_parameter_noise(rec, baybe_obj, noise_level=0.1)
        baybe_obj.add_results(rec)

    return baybe_obj
