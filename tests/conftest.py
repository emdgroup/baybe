"""
PyTest configuration
"""
import pytest

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise

# All fixture functions have prefix 'fixture_' and explicitly declared name due to:
# https://docs.pytest.org/en/stable/reference/reference.html#pytest-fixture


@pytest.fixture(name="config_basic_1target", scope="function")
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


@pytest.fixture(scope="session", params=[1], name="n_iterations")
def fixture_n_iterations(request):
    """
    Number if iterations ran in tests.
    """
    return request.param


@pytest.fixture(scope="session", params=[3], name="batch_quantity")
def fixture_batch_quantity(request):
    """
    Number of recommendations requested per iteration. Testing 1 as edge case and >1
    as common case.
    """
    return request.param


@pytest.fixture(scope="session", params=[5], name="n_grid_points")
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
