"""Tests for the simulation module."""

import os
from collections.abc import Callable, Sequence
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.simulation import simulate_scenarios

pytestmark = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Only tested in FULLTEST environment.",
)


def _match_aggregator(x: Sequence, /, match_value):
    idx = np.abs(np.asarray(x) - match_value).argmin()
    return x[idx]


_aggregators: dict[str, Callable] = {
    "Target_max": max,
    "Target_max_bounded": max,
    "Target_min": min,
    "Target_min_bounded": min,
    "Target_match_bell": partial(_match_aggregator, match_value=50.0),
}


@pytest.mark.parametrize(
    "target_names",
    [
        param(["Target_max"], id="single_max"),
        param(["Target_min"], id="single_min"),
        param(["Target_match_bell"], id="single_match"),
        param(
            ["Target_max_bounded", "Target_min_bounded", "Target_match_bell"],
            id="multi",
        ),
    ],
)
def test_simulate_scenarios_structure(campaign, batch_size):
    """Test simulate_scenarios output structure and correctness."""
    doe_iterations = 2
    n_mc_iterations = 2
    simulation_random_seed = 1337
    scenarios = {"test": campaign}

    result = simulate_scenarios(
        scenarios,
        None,  # use random data for lookup
        n_doe_iterations=doe_iterations,
        batch_size=batch_size,
        random_seed=simulation_random_seed,
        n_mc_iterations=n_mc_iterations,
        parallel_runs=False,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == doe_iterations * n_mc_iterations * len(scenarios)
    expected_cols = [
        "Scenario",
        "Monte_Carlo_Run",
        "Iteration",
        "Num_Experiments",
        "Random_Seed",
    ]
    for t in campaign.targets:
        expected_cols += [
            f"{t.name}_Measurements",
            f"{t.name}_IterBest",
            f"{t.name}_CumBest",
        ]
    assert set(result.columns) == set(expected_cols), (result.columns, expected_cols)
    assert set(result["Scenario"].unique()) == set(scenarios.keys())
    assert set(result["Monte_Carlo_Run"].unique()) == set(range(n_mc_iterations))
    assert set(result["Iteration"].unique()) == set(range(doe_iterations))

    expected_seed_values = list(
        range(simulation_random_seed, simulation_random_seed + n_mc_iterations)
    )
    assert set(result["Random_Seed"].unique()) == set(expected_seed_values)
    for mc_run in result["Monte_Carlo_Run"].unique():
        mc_data = result[result["Monte_Carlo_Run"] == mc_run]

        for t in campaign.targets:
            aggregator = _aggregators[t.name]
            assert np.isclose(
                mc_data[f"{t.name}_IterBest"].values,
                mc_data[f"{t.name}_Measurements"].apply(aggregator).values,
            ).all()

            all_measurements = [
                m
                for measurements in mc_data[f"{t.name}_Measurements"]
                for m in measurements
            ]
            cum_lens = mc_data[f"{t.name}_Measurements"].apply(len).cumsum().values
            expected_cum_best = [
                aggregator(all_measurements[:cum_len]) for cum_len in cum_lens
            ]

            assert list(mc_data[f"{t.name}_CumBest"]) == expected_cum_best
