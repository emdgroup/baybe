"""Benchmark on ArylHalides data with two distinct arylhalides as TL tasks.

This file provides the basic structure such that one can easily create different
benchmarks by changing the source and target tasks. The benchmark compares TL and
non-TL campaigns.

By convention, the benchmarks are named in the format "SourceHalides-TargetHalides.py"
where SourceHalides and TargetHalides are abbreviations of the used source and target
tasks respectively.
"""

from __future__ import annotations

import pandas as pd

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import SubstanceParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from benchmarks.data.utils import DATA_PATH
from benchmarks.definition import ConvergenceBenchmarkSettings


def get_data() -> pd.DataFrame:
    """Load the data for the benchmark."""
    data = pd.read_table(DATA_PATH / "ArylHalides" / "data.csv", sep=",").dropna(
        subset=["base", "ligand", "additive", "aryl_halide"]
    )
    # Only keep relevant columns
    data = data[
        [
            "base",
            "ligand",
            "additive",
            "ligand_smiles",
            "base_smiles",
            "additive_smiles",
            "aryl_halide",
            "yield",
        ]
    ]
    return data


def create_searchspace(
    data: pd.DataFrame,
    use_task_parameter: bool,
    target_tasks: list[str],
    source_tasks: list[str],
) -> SearchSpace:
    """Create the search space for the benchmark."""
    params: list[DiscreteParameter] = [
        SubstanceParameter(
            name=substance,
            data=dict(zip(data[substance], data[f"{substance}_smiles"])),
            encoding="MORDRED",
        )
        for substance in ["base", "ligand", "additive"]
    ]
    if use_task_parameter:
        params.append(
            TaskParameter(
                name="aryl_halide",
                values=target_tasks + source_tasks,
                active_values=target_tasks,
            )
        )
    return SearchSpace.from_product(parameters=params)


def create_objective() -> SingleTargetObjective:
    """Create the objective for the benchmark."""
    return SingleTargetObjective(NumericalTarget(name="yield", mode="MAX"))


def create_lookup(data: pd.DataFrame, target_tasks: list[str]) -> pd.DataFrame:
    """Create the lookup for the benchmark."""
    return data[data["aryl_halide"].isin(target_tasks)]


def create_initial_data(data: pd.DataFrame, source_tasks: list[str]) -> pd.DataFrame:
    """Create the initial data for the benchmark."""
    return data[data["aryl_halide"].isin(source_tasks)]


def abstract_arylhalides_tl_substance_benchmark(
    settings: ConvergenceBenchmarkSettings,
    source_tasks: list[str],
    target_tasks: list[str],
    percentages: list[float],
) -> pd.DataFrame:
    """Abstract benchmark function comparing TL and non-TL campaigns.

    Inputs:
        base:           Substance parameter
        ligand:         Substance parameter
        additive:       Substance parameter
        aryl_halide:    Task parameter
    Output:             Continuous (yield)
    Objective:          Maximization
    """
    data = get_data()

    searchspace = create_searchspace(
        data=data,
        use_task_parameter=True,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
    )
    searchspace_nontl = create_searchspace(
        data=data,
        use_task_parameter=False,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
    )

    lookup = create_lookup(data, target_tasks)
    initial_data = create_initial_data(data, source_tasks)

    tl_campaign = Campaign(
        searchspace=searchspace,
        objective=create_objective(),
    )
    nontl_campaign = Campaign(
        searchspace=searchspace_nontl, objective=create_objective()
    )

    results = []
    for p in percentages:
        results.append(
            simulate_scenarios(
                {
                    f"{int(100 * p)}": tl_campaign,
                    f"{int(100 * p)}_naive": nontl_campaign,
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
    # No training data and non-TL campaign
    results.append(
        simulate_scenarios(
            {"0": tl_campaign, "non_TL": nontl_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    return pd.concat(results)
