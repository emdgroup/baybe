"""Tests for the simulation module."""

import os
from collections.abc import Callable, Collection, Sequence
from contextlib import nullcontext
from functools import partial

import numpy as np
import pandas as pd
import pytest
from pytest import param

from baybe.settings import Settings
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
    """Validate the target-related columns in the produced simulation dataframe."""
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
    ("n_mc_iterations", "n_initial_data"),
    [
        param(None, None, id="invalid"),
        param(2, None, id="some_mc"),
        param(None, 2, id="some_data"),
        param(2, 2, id="cartesian"),
    ],
)
@Settings(parallelize_simulation_runs=False)
def test_simulate_scenarios_structure(
    campaign, batch_size, n_mc_iterations, n_initial_data
):
    """Test simulate_scenarios output structure and correctness."""
    doe_iterations = 2
    seed = 59234  # <-- uncommon number to avoid clash with default
    scenarios = {"test": campaign}

    if n_initial_data is None:
        initial_data = None
    else:
        initial_data = [
            create_fake_input(campaign.parameters, campaign.targets)
            for _ in range(n_initial_data)
        ]

    with (
        pytest.raises(ValueError, match="requires that initial data is specified")
        if (should_fail := n_mc_iterations is None and n_initial_data is None)
        else nullcontext()
    ):
        result = simulate_scenarios(
            scenarios,
            None,  # use random data for lookup
            n_doe_iterations=doe_iterations,
            batch_size=batch_size,
            random_seed=seed,
            n_mc_iterations=n_mc_iterations,
            initial_data=initial_data,
        )
    if should_fail:
        return

    expected_length = (
        len(scenarios)
        * doe_iterations
        * len(_Rollouts(n_mc_iterations, n_initial_data))
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_length

    expected_cols = [
        "Scenario",
        "Iteration",
        "Num_Experiments",
        "Random_Seed",
        "Initial_Data",
    ]
    for t in campaign.targets:
        expected_cols += [
            f"{t.name}_Measurements",
            f"{t.name}_IterBest",
            f"{t.name}_CumBest",
        ]
    assert set(result.columns) == set(expected_cols), (result.columns, expected_cols)
    assert set(result["Scenario"].unique()) == set(scenarios.keys())
    assert set(result["Iteration"].unique()) == set(range(doe_iterations))

    expected_seed_values = list(
        range(seed, seed + (n_mc_iterations or len(initial_data)))
    )
    assert set(result["Random_Seed"].unique()) == set(expected_seed_values)

    groupby_cols = ["Scenario", "Random_Seed", "Initial_Data"]
    result.groupby(groupby_cols).apply(_validate_target_data, targets=campaign.targets)
