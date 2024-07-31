"""Functionality for transfer learning backtesting."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from baybe.campaign import Campaign
from baybe.parameters import TaskParameter
from baybe.searchspace import SearchSpaceType
from baybe.simulation.scenarios import simulate_scenarios


def simulate_transfer_learning(
    campaign: Campaign,
    lookup: pd.DataFrame,
    /,
    *,
    batch_size: int = 1,
    n_doe_iterations: int | None = None,
    groupby: list[str] | None = None,
    n_mc_iterations: int = 1,
) -> pd.DataFrame:
    """Simulate Bayesian optimization with transfer learning.

    A wrapper around :func:`baybe.simulation.scenarios.simulate_scenarios` that
    partitions the search space into its tasks and simulates each task with the training
    data from the remaining tasks.

    **NOTE:**
    Currently, the simulation only supports purely discrete search spaces. This is
    because ``lookup`` serves both as the loop-closing element **and** as the source
    for off-task training data. For continuous (or mixed) spaces, the lookup mechanism
    would need to be either implemented as a callable (in which case the training data
    must be provided separately) or the continuous parameters need to be effectively
    restricted to the finite number of provided lookup configurations. Neither is
    implemented at the moment.

    Args:
        campaign: See :func:`baybe.simulation.core.simulate_experiment`.
        lookup: See :func:`baybe.simulation.scenarios.simulate_scenarios`.
        batch_size: See :func:`baybe.simulation.scenarios.simulate_scenarios`.
        n_doe_iterations: See :func:`baybe.simulation.scenarios.simulate_scenarios`.
        groupby: See :func:`baybe.simulation.scenarios.simulate_scenarios`.
        n_mc_iterations: See :func:`baybe.simulation.scenarios.simulate_scenarios`.

    Returns:
        A dataframe as returned by :func:`baybe.simulation.scenarios.simulate_scenarios`
        where the different tasks are represented in the ``Scenario`` column.

    Raises:
        NotImplementedError: If a non-discrete search space is chosen.
    """
    # TODO: Currently, we assume a purely discrete search space
    if campaign.searchspace.type != SearchSpaceType.DISCRETE:
        raise NotImplementedError(
            "Currently, only purely discrete search spaces are supported. "
            "For details, see NOTE in the function docstring."
        )

    # TODO [16932]: Currently, we assume exactly one task parameter exists
    # Extract the single task parameter
    task_params = [p for p in campaign.parameters if isinstance(p, TaskParameter)]
    if len(task_params) > 1:
        raise NotImplementedError(
            "Currently, transfer learning supports only a single task parameter."
        )
    task_param = task_params[0]

    # Create simulation objects for all tasks
    scenarios: dict[Any, Campaign] = {}
    for task in task_param.values:
        # Create a campaign that focuses only on the current task by excluding
        # off-task configurations from the candidates list
        # TODO: Reconsider if deepcopies are required once [16605] is resolved
        campaign_task = deepcopy(campaign)
        off_task_mask = campaign.searchspace.discrete.exp_rep[task_param.name] != task
        # TODO [16605]: Avoid direct manipulation of metadata
        campaign_task.searchspace.discrete.metadata.loc[
            off_task_mask.values, "dont_recommend"
        ] = True

        # Use all off-task data as training data
        df_train = lookup[lookup[task_param.name] != task]
        campaign_task.add_measurements(df_train)

        # Add the task scenario
        scenarios[task] = campaign_task

    # Simulate all tasks
    return simulate_scenarios(
        scenarios,
        lookup,
        batch_size=batch_size,
        n_doe_iterations=n_doe_iterations,
        groupby=groupby,
        n_mc_iterations=n_mc_iterations,
        impute_mode="ignore",
    )
