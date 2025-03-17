"""Benchmark on Directl Arylation data for transfer learning.

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
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import (
    ConvergenceBenchmarkSettings,
)
from benchmarks.definition.convergence import ConvergenceBenchmark


def get_data() -> pd.DataFrame:
    """Load data for benchmark."""
    data_path = DATA_PATH / "DirectArylation"
    data = pd.read_table(data_path / "data.csv", sep=",", index_col=0)
    data["Temp_C"] = data["Temp_C"].astype(str)
    return data


def create_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
) -> SearchSpace:
    """Create the search space for the benchmark."""
    params = [
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
            tolerance=0.001,
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


def create_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))


def create_lookup(data: pd.DataFrame) -> pd.DataFrame:
    """Create the lookup for the benchmark."""
    return data[data["Temp_C"] == "105"]


def create_initial_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create the initial data for the benchmark."""
    return data[data["Temp_C"] != "105"]


def direct_arylation_tl_temperature(
    settings: ConvergenceBenchmarkSettings,
) -> pd.DataFrame:
    """Benchmark on Direct Arylation data for transfer learning.

    Inputs:
        Solvent         Substance with `RDKIT2DDESCRIPTORS` encoding
        Base            Substance with `RDKIT2DDESCRIPTORS` encoding
        Ligand          Substance with `RDKIT2DDESCRIPTORS` encoding
        Concentration   Numerical discrete
        Temp_C          Discrete task parameter
    Output: continuous
    Objective: Maximization
    Optimal Inputs: [
        {
            Base:           "Cesium acetate",
            Ligand:         "SCHEMBL15068049",
            Solvent:        "DMAc",
            Concentration:  0.153
        },
        {
            Base:           "Cesium pivalate",
            Ligand:         "SCHEMBL15068049",
            Solvent:        "DMAc",
            Concentration:  0.153
        },
    ]
    Optimal Output: 100.0
    """
    data = get_data()

    searchspace = create_searchspace(
        data=data,
        use_task_parameter=True,
    )
    searchspace_nontl = create_searchspace(
        data=data,
        use_task_parameter=False,
    )

    # Searchspace in which all initial data will be added
    searchspace_naive_tl = create_searchspace(
        data=data,
        use_task_parameter=False,
    )

    lookup = create_lookup(data)
    initial_data = create_initial_data(data)

    tl_campaign = Campaign(
        searchspace=searchspace,
        objective=create_objective(),
    )
    non_tl_campaign = Campaign(
        searchspace=searchspace_nontl, objective=create_objective()
    )
    naive_campaign = Campaign(
        searchspace=searchspace_naive_tl, objective=create_objective()
    )
    naive_campaign.add_measurements(initial_data)

    results = []
    for p in [0.01, 0.02, 0.05, 0.1, 0.2]:
        results.append(
            simulate_scenarios(
                {f"{int(100 * p)}": tl_campaign},
                lookup,
                initial_data=[
                    initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
                ],
                batch_size=settings.batch_size,
                n_doe_iterations=settings.n_doe_iterations,
                impute_mode="error",
            )
        )
    # No training data and non-TL campaign
    results.append(
        simulate_scenarios(
            {"0": tl_campaign, "non_TL": non_tl_campaign, "naive": naive_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    results = pd.concat(results)
    return results


benchmark_config = ConvergenceBenchmarkSettings(
    batch_size=2,
    n_doe_iterations=10,
    n_mc_iterations=50,
)

direct_arylation_tl_temperature_benchmark = ConvergenceBenchmark(
    function=direct_arylation_tl_temperature,
    optimal_target_values={"yield": 100},
    settings=benchmark_config,
)
