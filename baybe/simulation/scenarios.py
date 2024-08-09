"""Batch simulation of multiple campaigns."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.exceptions import NothingToSimulateError, UnusedObjectWarning
from baybe.simulation.core import simulate_experiment

if TYPE_CHECKING:
    from xarray import DataArray

_DEFAULT_SEED = 1337


def simulate_scenarios(
    scenarios: dict[Any, Campaign],
    lookup: pd.DataFrame | Callable | None = None,
    /,
    *,
    batch_size: int = 1,
    n_doe_iterations: int | None = None,
    initial_data: list[pd.DataFrame] | None = None,
    groupby: list[str] | None = None,
    n_mc_iterations: int = 1,
    random_seed: int | None = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: float | None = None,
) -> pd.DataFrame:
    """Simulate multiple Bayesian optimization scenarios.

    A wrapper function around :func:`baybe.simulation.core.simulate_experiment` that
    allows to specify multiple simulation settings at once.

    Args:
        scenarios: A dictionary mapping scenario identifiers to DOE specifications.
        lookup: See :func:`baybe.simulation.core.simulate_experiment`.
        batch_size: See :func:`baybe.simulation.core.simulate_experiment`.
        n_doe_iterations: See :func:`baybe.simulation.core.simulate_experiment`.
        initial_data: A list of initial data sets for which the scenarios should be
            simulated.
        groupby: The names of the parameters to be used to partition the search space.
            A separate simulation will be conducted for each partition, with the search
            restricted to that partition.
        n_mc_iterations: The number of Monte Carlo simulations to be used.
        random_seed: An optional integer specifying the random seed for the first Monte
            Carlo run. Each subsequent runs will increase this value by 1. If omitted,
            the current random seed is used.
        impute_mode: See :func:`baybe.simulation.core.simulate_experiment`.
        noise_percent: See :func:`baybe.simulation.core.simulate_experiment`.

    Returns:
        A dataframe like returned from :func:`baybe.simulation.core.simulate_experiment`
        but with additional columns. See the ``Note`` for details.

    Note:
        The following additional columns are contained in the dataframe returned by this
        function:

        * ``Scenario``: Specifies the scenario identifier of the respective simulation.
        * ``Monte_Carlo_Run``: Specifies the Monte Carlo repetition of the
          respective simulation.
        * Optional, if ``random_seed`` is provided: A column ``Random_Seed`` that
          specifies the random seed used for the respective simulation.
        * Optional, if ``initial_data`` is provided: A column ``Initial_Data`` that
          specifies the index of the initial data set used for the respective
          simulation.
        * Optional, if ``groupby`` is provided: A column for each ``groupby`` parameter
          that specifies the search space partition considered for the respective
          simulation.
    """

    @dataclass
    class SimulationResult:
        """A thin wrapper to enable dataframe-valued return values with xyzpy.

        Args:
            result: The result of the simulation.
        """

        result: pd.DataFrame

    def make_xyzpy_callable(result_variable: str) -> Callable:
        """Make a batch simulator that allows running campaigns in parallel."""
        from baybe._optional.simulation import xyzpy

        @xyzpy.label(var_names=[result_variable])
        def simulate(
            Scenario: str,
            Monte_Carlo_Run: int,
            Initial_Data=None,
        ):
            """Callable for xyzpy simulation."""
            data = None if initial_data is None else initial_data[Initial_Data]
            seed = None if random_seed is None else Monte_Carlo_Run + _DEFAULT_SEED
            result = _simulate_groupby(
                scenarios[Scenario],
                lookup,
                batch_size=batch_size,
                n_doe_iterations=n_doe_iterations,
                initial_data=data,
                groupby=groupby,
                random_seed=seed,
                impute_mode=impute_mode,
                noise_percent=noise_percent,
            )
            if random_seed is not None:
                result["Random_Seed"] = seed
            return SimulationResult(result)

        return simulate

    def unpack_simulation_results(array: DataArray) -> pd.DataFrame:
        """Turn the xyzpy simulation results into a flat dataframe."""
        # Convert to dataframe and remove the wrapper layer
        series = array.to_series()
        series = series.apply(lambda x: x.result)

        # Un-nest all simulation results
        dfs = []
        for setting, df_result in series.items():
            df_setting = pd.DataFrame(
                [setting], columns=series.index.names, index=df_result.index
            )
            dfs.append(pd.concat([df_setting, df_result], axis=1))

        # Concatenate all results into a single dataframe
        return pd.concat(dfs, ignore_index=True)

    # Collect the settings to be simulated
    combos = {"Scenario": scenarios.keys()}
    combos["Monte_Carlo_Run"] = range(n_mc_iterations)
    if initial_data:
        combos["Initial_Data"] = range(len(initial_data))

    # Simulate and unpack
    result_variable = "simulation_result"
    batch_simulator = make_xyzpy_callable(result_variable)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UnusedObjectWarning,
            module="baybe.recommenders.pure.nonpredictive.base",
        )
        da_results = batch_simulator.run_combos(combos)[result_variable]

    df_results = unpack_simulation_results(da_results)

    return df_results


def _simulate_groupby(
    campaign: Campaign,
    lookup: pd.DataFrame | Callable[..., tuple[float, ...]] | None = None,
    /,
    *,
    batch_size: int = 1,
    n_doe_iterations: int | None = None,
    initial_data: pd.DataFrame | None = None,
    groupby: list[str] | None = None,
    random_seed: int = _DEFAULT_SEED,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: float | None = None,
) -> pd.DataFrame:
    """Scenario simulation for different search space partitions.

    A wrapper around :func:`baybe.simulation.core.simulate_experiment` that allows to
    partition the search space into different groups and run separate simulations for
    all groups where the search is restricted to the corresponding partition.

    Args:
        campaign: See :func:`baybe.simulation.core.simulate_experiment`.
        lookup: See :func:`baybe.simulation.core.simulate_experiment`.
        batch_size: See :func:`baybe.simulation.core.simulate_experiment`.
        n_doe_iterations: See :func:`baybe.simulation.core.simulate_experiment`.
        initial_data: See :func:`baybe.simulation.core.simulate_experiment`.
        groupby: See :func:`baybe.simulation.scenarios.simulate_scenarios`.
        random_seed: See :func:`baybe.simulation.core.simulate_experiment`.
        impute_mode: See :func:`baybe.simulation.core.simulate_experiment`.
        noise_percent: See :func:`baybe.simulation.core.simulate_experiment`.

    Returns:
        A dataframe like returned from
        :func:`baybe.simulation.core.simulate_experiment`, but with additional
        ``groupby columns`` (named according to the specified groupby parameters) that
        subdivide the results into the different simulations.

    Raises:
        NothingToSimulateError: If there is nothing to simulate.
    """
    # Create the groups. If no grouping is specified, use a single group containing
    # all parameter configurations.
    # NOTE: In the following, we intentionally work with *integer* indexing (iloc)
    #   instead of pandas indexes (loc), because the latter would yield wrong
    #   results in cases where the search space dataframe contains duplicate
    #   index entries (i.e., controlling the recommendable entries would affect
    #   all duplicates). While duplicate entries should be prevented by the search
    #   space constructor, the integer-based indexing provides a second safety net.
    #   Hence, the "reset_index" call.
    if groupby is None:
        groups = ((None, campaign.searchspace.discrete.exp_rep.reset_index()),)
    else:
        groups = campaign.searchspace.discrete.exp_rep.reset_index().groupby(groupby)

    # Simulate all subgroups
    dfs = []
    for group_id, group in groups:
        # Create a campaign that focuses only on the current group by excluding
        # off-group configurations from the candidates list
        # TODO: Reconsider if deepcopies are required once [16605] is resolved
        campaign_group = deepcopy(campaign)
        # TODO: Implement SubspaceDiscrete.__len__
        off_group_idx = np.full(
            len(campaign.searchspace.discrete.exp_rep), fill_value=True, dtype=bool
        )
        off_group_idx[group.index.values] = False
        # TODO [16605]: Avoid direct manipulation of metadata
        campaign_group.searchspace.discrete.metadata.loc[
            off_group_idx, "dont_recommend"
        ] = True

        # Run the group simulation
        try:
            df_group = simulate_experiment(
                campaign_group,
                lookup,
                batch_size=batch_size,
                n_doe_iterations=n_doe_iterations,
                initial_data=initial_data,
                random_seed=random_seed,
                impute_mode=impute_mode,
                noise_percent=noise_percent,
            )
        except NothingToSimulateError:
            continue

        # Add the group columns
        if groupby is not None:
            group_tuple = group_id if isinstance(group_id, tuple) else (group_id,)
            context = pd.DataFrame([group_tuple], columns=groupby, index=df_group.index)
            df_group = pd.concat([context, df_group], axis=1)

        dfs.append(df_group)

    # Collect all results
    if len(dfs) == 0:
        raise NothingToSimulateError
    df = pd.concat(dfs, ignore_index=True)

    return df
