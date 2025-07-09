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
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    ConvergenceBenchmarkSettings,
)
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

    searchspace = make_searchspace(
        data=data,
        use_task_parameter=True,
    )
    searchspace_nontl = make_searchspace(
        data=data,
        use_task_parameter=False,
    )

    lookup = make_lookup(data)
    initial_data = make_initial_data(data)
    objective = make_objective()

    tl_campaign = Campaign(searchspace=searchspace, objective=objective)
    non_tl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)

    results = []
    for p in [0.01, 0.1, 0.2]:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}": tl_campaign,
                    f"{int(100 * p)}_naive": non_tl_campaign,
                },
                lookup,
                initial_data=[
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    results.append(
        simulate_scenarios(
            {"0": tl_campaign, "0_naive": non_tl_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    return pd.concat(results)


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=20,
    n_mc_iterations=55,
)

direct_arylation_tl_temperature_benchmark = ConvergenceBenchmark(
    function=direct_arylation_tl_temperature,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
