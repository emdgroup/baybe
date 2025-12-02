"""Tests for the simulation module."""

import os
from collections.abc import Callable, Collection, Sequence
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.simulation import simulate_scenarios
from baybe.simulation.scenarios import _Rollouts
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import create_fake_input

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


def _validate_target_data(
    df: pd.DataFrame, targets: Collection[NumericalTarget]
) -> None:
    """Validate that the target-related columns in the produced simulation dataframe."""
    for t in targets:
        aggregator = _aggregators[t.name]
        assert np.isclose(
            df[f"{t.name}_IterBest"].values,
            df[f"{t.name}_Measurements"].apply(aggregator).values,
        ).all()

        all_measurements = [
            m for measurements in df[f"{t.name}_Measurements"] for m in measurements
        ]
        cum_lens = df[f"{t.name}_Measurements"].apply(len).cumsum().values
        expected_cum_best = [
            aggregator(all_measurements[:cum_len]) for cum_len in cum_lens
        ]

        assert list(df[f"{t.name}_CumBest"]) == expected_cum_best


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
@pytest.mark.parametrize(
    "rollouts",
    [
        param(_Rollouts(n_mc_iterations=None, n_initial_data=None), id="one_run"),
        param(_Rollouts(n_mc_iterations=2, n_initial_data=None), id="some_mc"),
        param(_Rollouts(n_mc_iterations=None, n_initial_data=2), id="some_data"),
        param(_Rollouts(n_mc_iterations=2, n_initial_data=2), id="cartesian"),
    ],
)
def test_simulate_scenarios_structure(campaign, batch_size, rollouts: _Rollouts):
    """Test simulate_scenarios output structure and correctness."""
    doe_iterations = 2
    simulation_random_seed = 59234  # <-- uncommon number to avoid clash with default
    scenarios = {"test": campaign}

    n_mc_iterations = rollouts.n_mc_iterations
    if (n_data := rollouts.n_initial_data) is None:
        initial_data = None
    else:
        initial_data = [
            create_fake_input(campaign.parameters, campaign.targets)
            for _ in range(n_data)
        ]

    result = simulate_scenarios(
        scenarios,
        None,  # use random data for lookup
        n_doe_iterations=doe_iterations,
        batch_size=batch_size,
        random_seed=simulation_random_seed,
        n_mc_iterations=n_mc_iterations,
        initial_data=initial_data,
        parallel_runs=False,
    )

    expected_length = len(scenarios) * doe_iterations * len(rollouts)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_length

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
    if initial_data is not None:
        expected_cols.append("Initial_Data")
    assert set(result.columns) == set(expected_cols), (result.columns, expected_cols)
    assert set(result["Scenario"].unique()) == set(scenarios.keys())
    assert set(result["Monte_Carlo_Run"].unique()) == set(range(n_mc_iterations or 1))
    assert set(result["Iteration"].unique()) == set(range(doe_iterations))

    expected_seed_values = list(
        range(simulation_random_seed, simulation_random_seed + (n_mc_iterations or 1))
    )
    assert set(result["Random_Seed"].unique()) == set(expected_seed_values)

    groupby_cols = ["Scenario", "Monte_Carlo_Run"]
    if initial_data is not None:
        groupby_cols.append("Initial_Data")
    result.groupby(groupby_cols).apply(_validate_target_data, targets=campaign.targets)
