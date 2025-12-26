"""Benchmark on aryl halides data with two distinct aryl halides as TL tasks.

This module provides the basic structure for creating different
benchmarks by changing the source and target tasks. The benchmark compares TL and
non-TL campaigns.

By convention, the benchmarks name use the format
"sou_<Source tasks>_tar_<Target task>".
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from distutils.util import strtobool

import pandas as pd
from joblib import Parallel, delayed

from baybe.campaign import Campaign
from baybe.objectives import SingleTargetObjective
from baybe.parameters import SubstanceParameter, TaskParameter
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.categorical import TransferMode
from baybe.searchspace import SearchSpace
from baybe.simulation import simulate_scenarios
from baybe.targets import NumericalTarget
from baybe.utils.random import temporary_seed
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
    transfer_mode: TransferMode | None = None,
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
                transfer_mode=transfer_mode or TransferMode.JOINT,
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


def make_objective() -> SingleTargetObjective:
    """Create the objective for aryl halides benchmarks."""
    return NumericalTarget(name="yield").to_objective()


def _run_percentage_scenario(
    p: float,
    campaigns: dict[str, Campaign],
    lookup: pd.DataFrame,
    initial_data_samples: dict[float, list[pd.DataFrame]],
    batch_size: int,
    n_doe_iterations: int,
    random_seed: int,
) -> pd.DataFrame:
    """Helper function to run scenarios for a single percentage.

    Args:
        p: The percentage of initial data to use
        campaigns: Dictionary of campaign configurations
        lookup: The lookup DataFrame
        initial_data_samples: Dictionary mapping percentages to lists of initial data
        batch_size: Batch size for DOE iterations
        n_doe_iterations: Number of DOE iterations
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with simulation results for this percentage
    """
    scenarios = {
        f"{int(100 * p)}_joint": campaigns["joint"],
        f"{int(100 * p)}_joint_pos": campaigns["joint_pos"],
        f"{int(100 * p)}_mean": campaigns["mean"],
        f"{int(100 * p)}_naive": campaigns["naive"],
    }

    return simulate_scenarios(
        scenarios,
        lookup,
        initial_data=initial_data_samples[p],
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        impute_mode="error",
        random_seed=random_seed,
    )


def _run_percentage_scenario_wrapper(args: tuple) -> pd.DataFrame:
    """Wrapper function for concurrent.futures that unpacks arguments."""
    return _run_percentage_scenario(*args)


def aryl_halide_tl_substance_benchmark(
    settings: ConvergenceBenchmarkSettings,
    source_tasks: Sequence[str],
    target_tasks: Sequence[str],
    percentages: Sequence[float],
    parallel_percentages: bool | None = None,
    n_percentage_jobs: int = -1,
) -> pd.DataFrame:
    """Abstract benchmark function comparing TL and non-TL campaigns.

    Args:
        settings: Configuration settings for the convergence benchmark
        source_tasks: List of source task names for transfer learning
        target_tasks: List of target task names to evaluate
        percentages: List of percentages of source data to test
        parallel_percentages: Whether to parallelize over percentages.
            If None, reads from BAYBE_PARALLEL_PERCENTAGE_RUNS environment variable
        n_percentage_jobs: Number of parallel jobs for percentage parallelization.
            -1 uses all available cores

    Returns:
        DataFrame containing benchmark results

    Inputs:
        base:           Substance parameter
        ligand:         Substance parameter
        additive:       Substance parameter
        aryl_halide:    Task parameter
    Output:             Continuous (yield)
    Objective:          Maximization
    """
    # Handle parallelization configuration
    if parallel_percentages is None:
        parallel_percentages = strtobool(
            os.environ.get("BAYBE_PARALLEL_PERCENTAGE_RUNS", "True")
        )

    data = load_data()

    # Create search spaces for each transfer mode
    searchspace_joint = make_searchspace(
        data=data,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
        transfer_mode=TransferMode.JOINT,
    )
    searchspace_joint_pos = make_searchspace(
        data=data,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
        transfer_mode=TransferMode.JOINT_POS,
    )
    searchspace_mean = make_searchspace(
        data=data,
        source_tasks=source_tasks,
        target_tasks=target_tasks,
        transfer_mode=TransferMode.MEAN,
    )
    searchspace_nontl = make_searchspace(data=data)

    lookup = make_lookup(data, target_tasks)
    initial_data = make_initial_data(data, source_tasks)
    objective = make_objective()

    # Create campaigns for each transfer mode
    tl_campaign_joint = Campaign(
        searchspace=searchspace_joint,
        objective=objective,
    )
    tl_campaign_joint_pos = Campaign(
        searchspace=searchspace_joint_pos,
        objective=objective,
    )
    tl_campaign_mean = Campaign(
        searchspace=searchspace_mean,
        objective=objective,
    )
    nontl_campaign = Campaign(searchspace=searchspace_nontl, objective=objective)

    initial_data_samples = {}
    with temporary_seed(settings.random_seed):
        for p in percentages:
            initial_data_samples[p] = [
                initial_data.sample(frac=p) for _ in range(settings.n_mc_iterations)
            ]

    # Test all transfer modes with all source data percentages (full matrix testing)
    # Create campaigns dictionary for helper function
    campaigns = {
        "joint": tl_campaign_joint,
        "joint_pos": tl_campaign_joint_pos,
        "mean": tl_campaign_mean,
        "naive": nontl_campaign,
    }

    if parallel_percentages and len(percentages) > 1:
        # Parallel execution over percentages
        try:
            # Try joblib first (preferred for HPC compatibility)
            results = Parallel(n_jobs=n_percentage_jobs, verbose=1, backend='loky')(
                delayed(_run_percentage_scenario)(
                    p, campaigns, lookup, initial_data_samples,
                    settings.batch_size, settings.n_doe_iterations, settings.random_seed
                )
                for p in percentages
            )
        except Exception as e:
            print(f"JobLib failed ({e}), falling back to concurrent.futures...")
            # Fallback to concurrent.futures
            max_workers = None if n_percentage_jobs == -1 else n_percentage_jobs
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(
                    _run_percentage_scenario_wrapper,
                    [(p, campaigns, lookup, initial_data_samples,
                      settings.batch_size, settings.n_doe_iterations, settings.random_seed)
                     for p in percentages]
                ))
    else:
        # Sequential execution (fallback)
        results = []
        for p in percentages:
            results.append(
                _run_percentage_scenario(
                    p, campaigns, lookup, initial_data_samples,
                    settings.batch_size, settings.n_doe_iterations, settings.random_seed
                )
            )
    results.append(
        simulate_scenarios(
            {
                "0_joint": tl_campaign_joint,
                "0_joint_pos": tl_campaign_joint_pos,
                "0_mean": tl_campaign_mean,
                "0_naive": nontl_campaign,
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
