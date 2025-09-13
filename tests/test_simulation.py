"""Tests for the simulation module."""

import os

import pandas as pd
import pytest

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.simulation import simulate_experiment, simulate_scenarios
from baybe.targets import NumericalTarget

pytestmark = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Only tested in FULLTEST environment.",
)


@pytest.fixture(scope="function")
def simulation_campaign() -> Campaign:
    """Fixture for a simple campaign used in simulation tests."""
    params = [
        NumericalDiscreteParameter("a", [0, 1, 2]),
        NumericalDiscreteParameter("b", [3, 4, 5]),
    ]
    searchspace = SearchSpace.from_product(parameters=params)
    objective = SingleTargetObjective(NumericalTarget("c"))
    return Campaign(searchspace=searchspace, objective=objective)


@pytest.fixture(scope="function")
def simulation_lookup() -> pd.DataFrame:
    """Fixture for a simple lookup DataFrame used in simulation tests."""
    return pd.DataFrame(
        {
            "a": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "b": [3, 4, 5, 3, 4, 5, 3, 4, 5],
            "c": [10, 20, 30, 40, 50, 60, 70, 80, 90],
        }
    )


@pytest.fixture(scope="function")
def simulation_random_seed() -> int:
    """Fixture for a random seed used in simulation tests."""
    return 1337


def test_simulate_experiment_dataframe_structure(
    simulation_campaign, simulation_lookup, simulation_random_seed
):
    """Test simulate_experiment result structure and correctness."""
    n_doe_iterations = 3

    result = simulate_experiment(
        simulation_campaign,
        simulation_lookup,
        n_doe_iterations=n_doe_iterations,
        batch_size=2,
        random_seed=simulation_random_seed,
    )

    expected_cols = [
        "Iteration",
        "Num_Experiments",
        "c_Measurements",
        "c_IterBest",
        "c_CumBest",
    ]

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == expected_cols
    assert len(result) == n_doe_iterations
    assert all(result["c_IterBest"] == result["c_Measurements"].apply(max))

    all_measurements = [
        m for measurements in result["c_Measurements"] for m in measurements
    ]
    cum_lens = result["c_Measurements"].apply(len).cumsum()
    expected_cum_best = [max(all_measurements[:cum_len]) for cum_len in cum_lens]

    assert list(result["c_CumBest"]) == expected_cum_best


def test_simulate_scenarios_structure(
    simulation_campaign, simulation_lookup, simulation_random_seed
):
    """Test simulate_scenarios output structure and correctness."""
    scenarios = {"Test Campaign": simulation_campaign}

    doe_iterations = 3
    n_mc_iterations = 3
    result = simulate_scenarios(
        scenarios,
        simulation_lookup,
        n_doe_iterations=doe_iterations,
        batch_size=2,
        random_seed=simulation_random_seed,
        n_mc_iterations=n_mc_iterations,
    )

    assert isinstance(result, pd.DataFrame)
    print(result.columns)
    assert len(result) == doe_iterations * n_mc_iterations * len(scenarios)
    expected_cols = [
        "Scenario",
        "Monte_Carlo_Run",
        "Iteration",
        "Num_Experiments",
        "c_Measurements",
        "c_IterBest",
        "c_CumBest",
        "Random_Seed",
    ]
    assert list(result.columns) == expected_cols
    assert set(result["Scenario"].unique()) == set(scenarios.keys())
    assert set(result["Monte_Carlo_Run"].unique()) == set(range(n_mc_iterations))
    assert set(result["Iteration"].unique()) == set(range(doe_iterations))

    expected_seed_values = list(
        range(simulation_random_seed, simulation_random_seed + n_mc_iterations)
    )
    assert set(result["Random_Seed"].unique()) == set(expected_seed_values)
    for mc_run in result["Monte_Carlo_Run"].unique():
        mc_data = result[result["Monte_Carlo_Run"] == mc_run]
        assert all(mc_data["c_IterBest"] == mc_data["c_Measurements"].apply(max))

        all_measurements = [
            m for measurements in mc_data["c_Measurements"] for m in measurements
        ]
        cum_lens = mc_data["c_Measurements"].apply(len).cumsum()
        expected_cum_best = [max(all_measurements[:cum_len]) for cum_len in cum_lens]

        assert list(mc_data["c_CumBest"]) == expected_cum_best
