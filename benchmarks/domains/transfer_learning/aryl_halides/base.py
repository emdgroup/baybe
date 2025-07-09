"""Benchmark on aryl halides data with two distinct aryl halides as TL tasks.

This module provides the basic structure for creating different
benchmarks by changing the source and target tasks. The benchmark compares TL and
non-TL campaigns.

By convention, the benchmarks name use the format
"sou_<Source tasks>_tar_<Target task>".
"""

from __future__ import annotations

from collections.abc import Sequence

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


def load_data() -> pd.DataFrame:
    """Load the data for the benchmark."""
    relevant_columns = [
        "base",
        "ligand",
        "additive",
        "ligand_smiles",
        "base_smiles",
        "additive_smiles",
        "aryl_halide",
        "yield",
    ]
    data = pd.read_table(
        DATA_PATH / "aryl_halide" / "data.csv", sep=",", usecols=relevant_columns
    ).dropna(subset=["base", "ligand", "additive", "aryl_halide"])
    return data


def make_searchspace(
    data: pd.DataFrame,
    target_tasks: Sequence[str] | None = None,
    source_tasks: Sequence[str] | None = None,
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
    if target_tasks is not None and source_tasks is not None:
        all_tasks = [*source_tasks, *target_tasks]
        all_tasks = [*source_tasks, *target_tasks]
        params.append(
            TaskParameter(
                name="aryl_halide",
                values=all_tasks,
                active_values=target_tasks,
            )
        )
    return SearchSpace.from_product(parameters=params)


def make_lookup(data: pd.DataFrame, target_tasks: Sequence[str]) -> pd.DataFrame:
    """Create the lookup for the benchmark.

    Without the filtering, there would be multiple entries for the same parameter
    configuration. Since this might yield issues for the non-transfer learning
    campaigns, we filter the data to only include the target tasks.
    """
    return data[data["aryl_halide"].isin(target_tasks)]


def make_initial_data(data: pd.DataFrame, source_tasks: Sequence[str]) -> pd.DataFrame:
    """Create the initial data for the benchmark."""
    return data[data["aryl_halide"].isin(source_tasks)]


def aryl_halide_tl_substance_benchmark(
    settings: ConvergenceBenchmarkSettings,
    source_tasks: Sequence[str],
    target_tasks: Sequence[str],
    percentages: Sequence[float],
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
    data = load_data()

    searchspace = make_searchspace(
        data=data,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
    )
    searchspace_nontl = make_searchspace(data=data)

    lookup = make_lookup(data, target_tasks)
    initial_data = make_initial_data(data, source_tasks)
    objective = SingleTargetObjective(NumericalTarget(name="yield"))

    tl_campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
    )
    nontl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)

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
    results.append(
        simulate_scenarios(
            {"0": tl_campaign, "0_naive": nontl_campaign},
            lookup,
            batch_size=settings.batch_size,
            n_doe_iterations=settings.n_doe_iterations,
            n_mc_iterations=settings.n_mc_iterations,
            impute_mode="error",
        )
    )
    return pd.concat(results)
