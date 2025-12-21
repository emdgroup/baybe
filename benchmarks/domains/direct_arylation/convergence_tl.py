"""Benchmark on Direct Arylation data for transfer learning.

This benchmark uses one temperature as source and another temperature as target.
"""

from __future__ import annotations

import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import (
    NumericalDiscreteParameter,
    SubstanceParameter,
    TaskParameter,
)
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.categorical import TransferMode
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.base import RunMode
from benchmarks.definition.convergence import ConvergenceBenchmark


def load_data() -> pd.DataFrame:
    """Load data for benchmark."""
    data = pd.read_table(
        DATA_PATH / "direct_arylation" / "data.csv",
        sep=",",
        index_col=0,
        dtype={"Temp_C": str},
    )
    return data


def make_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
    transfer_mode: TransferMode | None = None,
) -> SearchSpace:
    """Create the search space for the benchmark."""
    params: list[DiscreteParameter] = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_SMILES"])),
            encoding="RDKIT2DDESCRIPTORS",
        )
        for substance in ["Solvent", "Base", "Ligand"]
    ] + [
        NumericalDiscreteParameter(
            name="Concentration",
            values=sorted(data["Concentration"].unique()),
        ),
    ]
    if use_task_parameter:
        params.append(
            TaskParameter(
                name="Temp_C",
                values=["90", "105", "120"],
                active_values=["105"],
                transfer_mode=transfer_mode or TransferMode.JOINT,
            )
        )
    return SearchSpace.from_product(parameters=params)


def make_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(NumericalTarget(name="yield"))


def make_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create the lookup for the benchmark.

    Note that we filter the data to only include the target tasks.
    Without the filtering, there would be multiple entries for the same parameter
    configuration. Since this might yield issues for the non-transfer learning
    campaigns, we filter the data to only include the target tasks.
    """
    return data[data["Temp_C"] == "105"]


def make_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create the initial data for the benchmark."""
    return data[data["Temp_C"] != "105"]


def direct_arylation_tl_temperature(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark on Direct Arylation data for transfer learning.

    Key characteristics:
    • Uses one temperature as source (90°C and 120°C) and another as target (105°C)
    • Compares transfer learning vs. non-transfer learning approaches
    • Tests varying amounts of source data:
      - 1% of source data
      - 10% of source data
      - 20% of source data
    • Includes baseline with no transfer learning (0% source data)
    • Parameters:
      - Solvent: Substance with RDKIT2DDESCRIPTORS encoding
      - Base: Substance with RDKIT2DDESCRIPTORS encoding
      - Ligand: Substance with RDKIT2DDESCRIPTORS encoding
      - Concentration: Numerical discrete
      - Temp_C: Task parameter (90°C, 105°C, 120°C)
    • Target: Reaction yield (continuous)
    • Objective: Maximization

    Args:
        settings: Configuration settings for the convergence benchmark

    Returns:
        DataFrame containing benchmark results for all test cases
    """
    data = load_data()

    # Create search spaces for each transfer mode
    searchspace_joint = make_searchspace(
        data=data,
        use_task_parameter=True,
        transfer_mode=TransferMode.JOINT,
    )
    searchspace_joint_pos = make_searchspace(
        data=data,
        use_task_parameter=True,
        transfer_mode=TransferMode.JOINT_POS,
    )
    searchspace_mean = make_searchspace(
        data=data,
        use_task_parameter=True,
        transfer_mode=TransferMode.MEAN,
    )
    searchspace_nontl = make_searchspace(
        data=data,
        use_task_parameter=False,
    )

    lookup = make_lookup(data)
    initial_data = make_initial_data(data)
    objective = make_objective()

    # Create campaigns for each transfer mode
    tl_campaign_joint = Campaign(searchspace=searchspace_joint, objective=objective)
    tl_campaign_joint_pos = Campaign(
        searchspace=searchspace_joint_pos, objective=objective
    )
    tl_campaign_mean = Campaign(searchspace=searchspace_mean, objective=objective)
    non_tl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)

    percentages = [0.01, 0.1, 0.2]

    initial_data_samples = {}
    with temporary_seed(settings.random_seed):
        for p in percentages:
            initial_data_samples[p] = [
                initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
            ]

    # Test all transfer modes with all source data percentages (full matrix testing)
    results = []
    for p in percentages:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}_joint": tl_campaign_joint,
                    f"{int(100 * p)}_joint_pos": tl_campaign_joint_pos,
                    f"{int(100 * p)}_mean": tl_campaign_mean,
                    f"{int(100 * p)}_naive": non_tl_campaign,
                },
                lookup,
                initial_data=initial_data_samples[p],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
                random_seed=settings.random_seed,
            )
        )
    results.append(
        simulate_scenarios(
            {
                "0_joint": tl_campaign_joint,
                "0_joint_pos": tl_campaign_joint_pos,
                "0_mean": tl_campaign_mean,
                "0_naive": non_tl_campaign,
            },
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
            random_seed=settings.random_seed,
        )
    )
    return pd.concat(results)


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size_settings={
        RunMode.DEFAULT: 2,
        RunMode.SMOKETEST: 2,
    },
    n_doe_iterations_settings={
        RunMode.DEFAULT: 20,
        RunMode.SMOKETEST: 2,
    },
    n_mc_iterations_settings={
        RunMode.DEFAULT: 55,
        RunMode.SMOKETEST: 2,
    },
)

direct_arylation_tl_temperature_benchmark = ConvergenceBenchmark(
    function=direct_arylation_tl_temperature,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
