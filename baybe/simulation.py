"""Functionality to "simulate" Bayesian DOE given a lookup mechanism.

The term "simulation" can have two slightly different interpretations, depending on the
applied context:

*   It can refer to "backtesting" a particular DOE strategy on a fixed (finite)
    dataset. In this context, "simulation" means investigating what experimental
    trajectory we would have observed if we had applied the strategy in a certain
    defined context and restricted the possible parameter configurations to those
    contained in the dataset.

*   It can refer to the simulation of an actual DOE loop (i.e. recommending experiments
    and retrieving the corresponding measurements) where the loop closure is realized
    in the form of a callable (black-box) function that can be queried during the
    optimization to provide target values.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

try:
    import xyzpy as xyz
    from xarray import DataArray
except ImportError as ex:
    # This is just to augment the error message and provide a suggestion
    raise ModuleNotFoundError(
        "xyzpy is not installed, the simulation module is unavailable. Consider "
        "installing baybe with `simulation` dependency, e.g.`pip install "
        "baybe[simulation]`"
    ) from ex

from baybe.campaign import Campaign
from baybe.exceptions import NotEnoughPointsLeftError, NothingToSimulateError
from baybe.parameters import TaskParameter
from baybe.searchspace import SearchSpaceType
from baybe.utils import (
    add_fake_results,
    add_parameter_noise,
    closer_element,
    closest_element,
    set_random_seed,
)

if TYPE_CHECKING:
    from baybe.targets import NumericalTarget

_logger = logging.getLogger(__name__)
_DEFAULT_SEED = 1337


def simulate_transfer_learning(
    campaign: Campaign,
    lookup: pd.DataFrame,
    /,
    *,
    batch_quantity: int = 1,
    n_doe_iterations: Optional[int] = None,
    groupby: Optional[List[str]] = None,
    n_mc_iterations: int = 1,
) -> pd.DataFrame:
    """Simulate Bayesian optimization with transfer learning.

    A wrapper around :func:`baybe.simulation.simulate_scenarios` that partitions the
    search space into its tasks and simulates each task with the training data from the
    remaining tasks.

    **NOTE:**
    Currently, the simulation only supports purely discrete search spaces. This is
    because ``lookup`` serves both as the loop-closing element **and** as the source
    for off-task training data. For continuous (or mixed) spaces, the lookup mechanism
    would need to be either implemented as a callable (in which case the training data
    must be provided separately) or the continuous parameters need to be effectively
    restricted to the finite number of provided lookup configurations. Neither is
    implemented at the moment.

    Args:
        campaign: See :func:`baybe.simulation.simulate_experiment`.
        lookup: See :func:`baybe.simulation.simulate_scenarios`.
        batch_quantity: See :func:`baybe.simulation.simulate_scenarios`.
        n_doe_iterations: See :func:`baybe.simulation.simulate_scenarios`.
        groupby: See :func:`baybe.simulation.simulate_scenarios`.
        n_mc_iterations: See :func:`baybe.simulation.simulate_scenarios`.

    Returns:
        A dataframe as returned by :func:`baybe.simulation.simulate_scenarios` where
        the different tasks are represented in the ``Scenario`` column.

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
    scenarios: Dict[Any, Campaign] = {}
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
        batch_quantity=batch_quantity,
        n_doe_iterations=n_doe_iterations,
        groupby=groupby,
        n_mc_iterations=n_mc_iterations,
        impute_mode="ignore",
    )


def simulate_scenarios(
    scenarios: Dict[Any, Campaign],
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    /,
    *,
    batch_quantity: int = 1,
    n_doe_iterations: Optional[int] = None,
    initial_data: Optional[List[pd.DataFrame]] = None,
    groupby: Optional[List[str]] = None,
    n_mc_iterations: int = 1,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate multiple Bayesian optimization scenarios.

    A wrapper function around :func:`baybe.simulation.simulate_experiment` that
    allows to specify multiple simulation settings at once.

    Args:
        scenarios: A dictionary mapping scenario identifiers to DOE specifications.
        lookup: See :func:`baybe.simulation.simulate_experiment`.
        batch_quantity: See :func:`baybe.simulation.simulate_experiment`.
        n_doe_iterations: See :func:`baybe.simulation.simulate_experiment`.
        initial_data: A list of initial data sets for which the scenarios should be
            simulated.
        groupby: The names of the parameters to be used to partition the search space.
            A separate simulation will be conducted for each partition, with the search
            restricted to that partition.
        n_mc_iterations: The number of Monte Carlo simulations to be used.
        impute_mode: See :func:`baybe.simulation.simulate_experiment`.
        noise_percent: See :func:`baybe.simulation.simulate_experiment`.

    Returns:
        A dataframe like returned from :func:`baybe.simulation.simulate_experiment` but
        with additional columns. See the ``Note`` for details.

    Note:
        The following additional columns are contained in the dataframe returned by this
        function:

        * ``Scenario``: Specifies the scenario identifier of the respective simulation.
        * ``Random_Seed``: Specifies the random seed used for the respective simulation.
        * Optional, if ``initial_data`` is provided: A column ``Initial_Data`` that
          specifies the index of the initial data set used for the respective
          simulation.
        * Optional, if ``groupby`` is provided: A column for each ``groupby`` parameter
          that specifies the search space partition considered for the respective
          simulation.
    """
    _RESULT_VARIABLE = "simulation_result"

    @dataclass
    class SimulationResult:
        """A thin wrapper to enable dataframe-valued return values with xyzpy.

        Args:
            result: The result of the simulation.
        """

        result: pd.DataFrame

    @xyz.label(var_names=[_RESULT_VARIABLE])
    def simulate(
        Scenario: str,
        Random_Seed=None,
        Initial_Data=None,
    ):
        """Callable for xyzpy simulation."""
        data = None if initial_data is None else initial_data[Initial_Data]
        return SimulationResult(
            _simulate_groupby(
                scenarios[Scenario],
                lookup,
                batch_quantity=batch_quantity,
                n_doe_iterations=n_doe_iterations,
                initial_data=data,
                groupby=groupby,
                random_seed=Random_Seed,
                impute_mode=impute_mode,
                noise_percent=noise_percent,
            )
        )

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
    combos["Random_Seed"] = range(_DEFAULT_SEED, _DEFAULT_SEED + n_mc_iterations)
    if initial_data:
        combos["Initial_Data"] = range(len(initial_data))

    # Simulate and unpack
    da_results = simulate.run_combos(combos)[_RESULT_VARIABLE]
    df_results = unpack_simulation_results(da_results)

    return df_results


def _simulate_groupby(
    campaign: Campaign,
    lookup: Optional[Union[pd.DataFrame, Callable[..., Tuple[float, ...]]]] = None,
    /,
    *,
    batch_quantity: int = 1,
    n_doe_iterations: Optional[int] = None,
    initial_data: Optional[pd.DataFrame] = None,
    groupby: Optional[List[str]] = None,
    random_seed: int = _DEFAULT_SEED,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """Scenario simulation for different search space partitions.

    A wrapper around :func:`baybe.simulation.simulate_experiment` that allows to
    partition the search space into different groups and run separate simulations for
    all groups where the search is restricted to the corresponding partition.

    Args:
        campaign: See :func:`baybe.simulation.simulate_experiment`.
        lookup: See :func:`baybe.simulation.simulate_experiment`.
        batch_quantity: See :func:`baybe.simulation.simulate_experiment`.
        n_doe_iterations: See :func:`baybe.simulation.simulate_experiment`.
        initial_data: See :func:`baybe.simulation.simulate_experiment`.
        groupby: See :func:`baybe.simulation.simulate_scenarios`.
        random_seed: See :func:`baybe.simulation.simulate_experiment`.
        impute_mode: See :func:`baybe.simulation.simulate_experiment`.
        noise_percent: See :func:`baybe.simulation.simulate_experiment`.

    Returns:
        A dataframe like returned from :func:`baybe.simulation.simulate_experiments`,
        but with additional ``groupby columns`` (named according to the specified
        groupby parameters) that subdivide the results into the different simulations.

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
                batch_quantity=batch_quantity,
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


def simulate_experiment(
    campaign: Campaign,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    /,
    *,
    batch_quantity: int = 1,
    n_doe_iterations: Optional[int] = None,
    initial_data: Optional[pd.DataFrame] = None,
    random_seed: int = _DEFAULT_SEED,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """Simulate a Bayesian optimization loop.

    The most basic type of simulation. Runs a single execution of the loop either
    for a specified number of steps or until there are no more configurations left
    to be tested.

    Args:
        campaign: The DOE setting to be simulated.
        lookup: The lookup used to close the loop, provided in the form of a dataframe
            or callable that define the targets for the queried parameter settings:
            First, a dataframe containing experimental settings and their target
            results can be chosen.
            Second, A callable, providing target values for the given parameter
            settings. can be chosen. The callable is assumed to return either a float
            or a tuple of floats and to accept an arbitrary number of floats as input.
            Finally,``None`` can be chosen, producing fake results.
        batch_quantity: The number of recommendations to be queried per iteration.
        n_doe_iterations:  The number of iterations to run the design-of-experiments
            loop. If not specified, the simulation proceeds until there are no more
            testable configurations left.
        initial_data: The initial measurement data to be ingested before starting the
            loop.
        random_seed: The random seed used for the simulation.
        impute_mode: Specifies how a missing lookup will be handled.
            There are six different options available.

            - ``"error"``: An error will be thrown.
            - ``"worst"``: Imputation uses the worst available value for each target.
            - ``"best"``: Imputation uses the best available value for each target.
            - ``"mean"``: Imputation uses the mean value for each target.
            - ``"random"``: A random row will be used as lookup.
            - ``"ignore"``: The search space is stripped before recommendations are made
              so that unmeasured experiments will not be recommended.
        noise_percent: If not ``None``, relative noise in percent of
            ``noise_percent`` will be applied to the parameter measurements.

    Returns:
        A dataframe ready for plotting, see the ``Note`` for details.

    Raises:
        TypeError: If a non-suitable lookup is chosen.
        ValueError: If the impute mode ``ignore`` is chosen for non-dataframe lookup.
        ValueError: If a setup is provided that would run indefinitely.

    Note:
        The returned dataframe contains the following columns:

        * ``Iteration``:
          Corresponds to the DOE iteration (starting at 0)
        * ``Num_Experiments``:
          Corresponds to the running number of experiments performed (usually x-axis)
        * for each target a column ``{targetname}_IterBest``:
          Corresponds to the best result for that target at the respective iteration
        * for each target a column ``{targetname}_CumBest``:
          Corresponds to the best result for that target up to including
          respective iteration
        * for each target a column ``{targetname}_Measurements``:
          The individual measurements obtained for the respective target and iteration
    """
    # TODO: Due to the "..." operator, sphinx does not render this properly. Might
    # want to investigate in the future.
    # TODO: In the markdown variant, bullet points with more levels are not rendered
    # properly. We thus might want to refactor this when using html based documentation
    # Validate the lookup mechanism
    if not (isinstance(lookup, (pd.DataFrame, Callable)) or (lookup is None)):
        raise TypeError(
            "The lookup can either be 'None', a pandas dataframe or a callable."
        )

    # Validate the data imputation mode
    if (impute_mode == "ignore") and (not isinstance(lookup, pd.DataFrame)):
        raise ValueError(
            "Impute mode 'ignore' is only available for dataframe lookups."
        )

    # Validate the number of experimental steps
    # TODO: Probably, we should add this as a property to Campaign
    will_terminate = (campaign.searchspace.type == SearchSpaceType.DISCRETE) and (
        not campaign.strategy.allow_recommending_already_measured
    )
    if (n_doe_iterations is None) and (not will_terminate):
        raise ValueError(
            "For the specified setting, the experimentation loop can be continued "
            "indefinitely. Hence, `n_doe_iterations` must be explicitly provided."
        )

    # Create a fresh campaign and set the corresponding random seed
    # TODO: Reconsider if deepcopies are required once [16605] is resolved
    campaign = deepcopy(campaign)
    set_random_seed(random_seed)

    # Add the initial data
    if initial_data is not None:
        campaign.add_measurements(initial_data)

    # For impute_mode 'ignore', do not recommend space entries that are not
    # available in the lookup
    # TODO [16605]: Avoid direct manipulation of metadata
    if impute_mode == "ignore":
        searchspace = campaign.searchspace.discrete.exp_rep
        missing_inds = searchspace.index[
            searchspace.merge(lookup, how="left", indicator=True)["_merge"]
            == "left_only"
        ]
        campaign.searchspace.discrete.metadata.loc[
            missing_inds, "dont_recommend"
        ] = True

    # Run the DOE loop
    limit = n_doe_iterations or np.inf
    k_iteration = 0
    n_experiments = 0
    dfs = []
    while k_iteration < limit:
        # Get the next recommendations and corresponding measurements
        try:
            measured = campaign.recommend(batch_quantity=batch_quantity)
        except NotEnoughPointsLeftError:
            # TODO: This except block requires a more elegant solution
            strategy = campaign.strategy
            allow_repeated = strategy.allow_repeated_recommendations
            allow_measured = strategy.allow_recommending_already_measured

            # Sanity check: if the variable was True, the except block should have
            # been impossible to reach in the first place
            # TODO: Currently, this is still possible due to bug [15917] though.
            assert not allow_measured

            measured, _ = campaign.searchspace.discrete.get_candidates(
                allow_repeated_recommendations=allow_repeated,
                allow_recommending_already_measured=allow_measured,
            )

            if len(measured) == 0:
                break

        n_experiments += len(measured)
        _look_up_target_values(measured, campaign, lookup, impute_mode)

        # Create the summary for the current iteration and store it
        result = pd.DataFrame(
            [  # <-- this ensures that the internal lists to not get expanded
                {
                    "Iteration": k_iteration,
                    "Num_Experiments": n_experiments,
                    **{
                        f"{target.name}_Measurements": measured[target.name].to_list()
                        for target in campaign.targets
                    },
                }
            ]
        )
        dfs.append(result)

        # Apply optional noise to the parameter measurements
        if noise_percent:
            add_parameter_noise(
                measured,
                campaign.parameters,
                noise_type="relative_percent",
                noise_level=noise_percent,
            )

        # Update the campaign
        campaign.add_measurements(measured)

        # Update the iteration counter
        k_iteration += 1

    # Collect the iteration results
    if len(dfs) == 0:
        raise NothingToSimulateError()
    results = pd.concat(dfs, ignore_index=True)

    # Add the instantaneous and running best values for all targets
    for target in campaign.targets:
        # Define the summary functions for the current target
        if target.mode == "MAX":
            agg_fun = np.max
            cum_fun = np.maximum.accumulate
        elif target.mode == "MIN":
            agg_fun = np.min
            cum_fun = np.minimum.accumulate
        elif target.mode == "MATCH":
            match_val = np.mean(target.bounds)
            agg_fun = partial(closest_element, target=match_val)
            cum_fun = lambda x: np.array(  # noqa: E731
                np.frompyfunc(
                    partial(closer_element, target=match_val),
                    2,
                    1,
                ).accumulate(x),
                dtype=float,
            )

        # Add the summary columns
        measurement_col = f"{target.name}_Measurements"
        iterbest_col = f"{target.name}_IterBest"
        cumbest_cols = f"{target.name}_CumBest"
        results[iterbest_col] = results[measurement_col].apply(agg_fun)
        results[cumbest_cols] = cum_fun(results[iterbest_col])

    return results


def _look_up_target_values(
    queries: pd.DataFrame,
    campaign: Campaign,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
):
    """Fill the target values in the query dataframe using the lookup mechanism.

    Note that this does not create a new dataframe but modifies ``queries`` in-place.

    Args:
        queries: A dataframe containing points to be queried.
        campaign: The campaign for which the experiments should be simulated.
        lookup: The lookup mechanism. See :func:`baybe.simulation.simulate_scenarios`
            for details.
        impute_mode: The used impute mode. See
            :func:`baybe.simulation.simulate_scenarios` for details.

    Raises:
        AssertionError: If an analytical function is used and an incorrect number of
            targets was specified.
    """
    # TODO: This function needs another code cleanup and refactoring. In particular,
    #   the different lookup modes should be implemented via multiple dispatch.

    # Extract all target names
    target_names = [t.name for t in campaign.targets]

    # If no lookup is provided, invent some fake results
    if lookup is None:
        add_fake_results(queries, campaign)

    # Compute the target values via a callable
    elif isinstance(lookup, Callable):
        # TODO: Currently, the alignment of return values to targets is based on the
        #   column ordering, which is not robust. Instead, the callable should return
        #   a dataframe with properly labeled columns.

        # Since the return of a lookup function is a a tuple, the following code stores
        # tuples of floats in a single column with label 0:
        measured_targets = queries.apply(lambda x: lookup(*x.values), axis=1).to_frame()
        # We transform this column to a DataFrame in which there is an individual
        # column for each of the targets....
        split_target_columns = pd.DataFrame(
            measured_targets[0].to_list(), index=measured_targets.index
        )
        # ... and assign this to measured_targets in order to have one column per target
        measured_targets[split_target_columns.columns] = split_target_columns
        if measured_targets.shape[1] != len(campaign.targets):
            raise AssertionError(
                "If you use an analytical function as lookup, make sure "
                "the configuration has the right amount of targets "
                "specified."
            )
        for k_target, target in enumerate(campaign.targets):
            queries[target.name] = measured_targets.iloc[:, k_target]

    # Get results via dataframe lookup (works only for exact matches)
    # IMPROVE: Although its not too important for a simulation, this
    #  could also be implemented for approximate matches
    elif isinstance(lookup, pd.DataFrame):
        all_match_vals = []
        for _, row in queries.iterrows():
            # IMPROVE: to the entire matching at once via a merge
            ind = lookup[
                (lookup.loc[:, row.index] == row).all(axis=1, skipna=False)
            ].index.values

            if len(ind) > 1:
                # More than two instances of this parameter combination
                # have been measured
                _logger.warning(
                    "The lookup rows with indexes %s seem to be "
                    "duplicates regarding parameter values. Choosing a "
                    "random one.",
                    ind,
                )
                match_vals = lookup.loc[np.random.choice(ind), target_names].values

            elif len(ind) < 1:
                # Parameter combination cannot be looked up and needs to be
                # imputed.
                if impute_mode == "ignore":
                    raise AssertionError(
                        "Something went wrong for impute_mode 'ignore'. "
                        "It seems the search space was not correctly "
                        "reduced before recommendations were generated."
                    )
                match_vals = _impute_lookup(row, lookup, campaign.targets, impute_mode)

            else:
                # Exactly one match has been found
                match_vals = lookup.loc[ind[0], target_names].values

            # Collect the matches
            all_match_vals.append(match_vals)

        # Add the lookup values
        queries.loc[:, target_names] = np.asarray(all_match_vals)


def _impute_lookup(
    row: pd.Series,
    lookup: pd.DataFrame,
    targets: List[NumericalTarget],
    mode: Literal["error", "best", "worst", "mean", "random"] = "error",
) -> np.ndarray:
    """Perform data imputation for missing lookup values.

    Depending on the chosen mode, this might raise errors instead.

    Args:
        row: The data that should be matched with the lookup data frame.
        lookup: The lookup data frame.
        targets: The campaign targets, providing the required mode information.
        mode: The used impute mode. See :func:`baybe.simulation.simulate_scenarios`
            for details.

    Returns:
        The filled-in lookup results.

    Raises:
        IndexError: If the mode ``"error"`` is chosen and at least one of the targets
            could not be found.
    """
    # TODO: this function needs another code cleanup and refactoring

    target_names = [t.name for t in targets]
    if mode == "mean":
        match_vals = lookup.loc[:, target_names].mean(axis=0).values
    elif mode == "worst":
        worst_vals = []
        for target in targets:
            if target.mode == "MAX":
                worst_vals.append(lookup.loc[:, target.name].min().flatten()[0])
            elif target.mode == "MIN":
                worst_vals.append(lookup.loc[:, target.name].max().flatten()[0])
            if target.mode == "MATCH":
                worst_vals.append(
                    lookup.loc[
                        lookup.loc[
                            (lookup[target.name] - target.bounds.center).abs().idxmax(),
                        ],
                        target.name,
                    ].flatten()[0]
                )
        match_vals = np.array(worst_vals)
    elif mode == "best":
        best_vals = []
        for target in targets:
            if target.mode == "MAX":
                best_vals.append(lookup.loc[:, target.name].max().flatten()[0])
            elif target.mode == "MIN":
                best_vals.append(lookup.loc[:, target.name].min().flatten()[0])
            if target.mode == "MATCH":
                best_vals.append(
                    lookup.loc[
                        lookup.loc[
                            (lookup[target.name] - target.bounds.center).abs().idxmin(),
                        ],
                        target.name,
                    ].flatten()[0]
                )
        match_vals = np.array(best_vals)
    elif mode == "random":
        vals = []
        randindex = np.random.choice(lookup.index)
        for target in targets:
            vals.append(lookup.loc[randindex, target.name].flatten()[0])
        match_vals = np.array(vals)
    else:
        raise IndexError(
            f"Cannot match the recommended row {row} to any of "
            f"the rows in the lookup."
        )

    return match_vals
