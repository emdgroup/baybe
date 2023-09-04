"""
Functionality to "simulate" Bayesian DOE given a lookup mechanism.

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
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import pandas as pd
import xyzpy as xyz
from xarray import DataArray

from baybe.core import BayBE
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
    baybe: BayBE,
    lookup: pd.DataFrame,
    /,
    *,
    batch_quantity: int = 1,
    n_exp_iterations: Optional[int] = None,
    groupby: Optional[List[str]] = None,
    n_mc_iterations: int = 1,
) -> pd.DataFrame:
    """
    Simulate Bayesian optimization with transfer learning.

    A wrapper around `simulate_scenarios` that partitions the search space into its
    tasks and simulates each task with the training data from the remaining tasks.

    Parameters
    ----------
    baybe:
        See `simulate_experiment`.
    lookup
        See `simulate_scenarios`.
    batch_quantity
        See `simulate_scenarios`.
    n_exp_iterations
        See `simulate_scenarios`.
    groupby
        See `simulate_scenarios`.
    n_mc_iterations
        See `simulate_scenarios`.

    Returns
    -------
    A dataframe as returned by `simulate_scenarios` where the different tasks are
    represented in the 'Scenario' column.
    """

    # TODO: Currently, we assume a purely discrete search space
    assert baybe.searchspace.type == SearchSpaceType.DISCRETE

    # TODO: Currently, we assume no prior measurements
    assert len(baybe.measurements_exp) == 0

    # TODO: Currently, we assume that everything can be recommended
    assert not baybe.searchspace.discrete.metadata.any().any()

    # TODO [16932]: Currently, we assume exactly one task parameter exists
    # Extract the single task parameter
    task_params = [p for p in baybe.parameters if isinstance(p, TaskParameter)]
    assert len(task_params) == 1
    task_param = task_params[0]

    # Create simulation objects for all tasks
    scenarios: Dict[Any, BayBE] = {}
    for task in task_param.values:

        # Create a baybe object that focuses only on the current task by excluding
        # off-task configurations from the candidates list
        # TODO: Reconsider if deepcopies are required once [16605] is resolved
        baybe_task = deepcopy(baybe)
        off_task_mask = baybe.searchspace.discrete.exp_rep[task_param.name] != task
        # TODO [16605]: Avoid direct manipulation of metadata
        baybe_task.searchspace.discrete.metadata.loc[
            off_task_mask.values, "dont_recommend"
        ] = True

        # Use all off-task data as training data
        df_train = lookup[lookup[task_param.name] != task]
        baybe_task.add_measurements(df_train)

        # Add the task scenario
        scenarios[task] = baybe_task

    # Simulate all tasks
    return simulate_scenarios(
        scenarios,
        lookup,
        batch_quantity=batch_quantity,
        n_exp_iterations=n_exp_iterations,
        groupby=groupby,
        n_mc_iterations=n_mc_iterations,
        impute_mode="ignore",
    )


def simulate_scenarios(
    scenarios: Dict[Any, BayBE],
    lookup: Optional[
        Union[pd.DataFrame, Callable[[float, ...], Union[float, Tuple[float, ...]]]]
    ] = None,
    /,
    *,
    batch_quantity: int = 1,
    n_exp_iterations: Optional[int] = None,
    initial_data: Optional[List[pd.DataFrame]] = None,
    groupby: Optional[List[str]] = None,
    n_mc_iterations: int = 1,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulation of multiple Bayesian optimization scenarios.

    A wrapper function around `simulate_experiment` that allows to specify multiple
    simulation settings at once.

    Parameters
    ----------
    scenarios
        A dictionary mapping scenario identifiers to DOE specifications.
    lookup
        See `simulate_experiment`.
    batch_quantity
        See `simulate_experiment`.
    n_exp_iterations
        See `simulate_experiment`.
    initial_data
        A list of initial data sets for which the scenarios should be simulated.
    groupby
        See `_simulate_groupby`.
    n_mc_iterations
        The number of Monte Carlo simulations to be used.
    impute_mode
        See `simulate_experiment`.
    noise_percent
        See `simulate_experiment`.

    Returns
    -------
    A dataframe like returned from `simulate_experiments` but with the following
    additional columns:
        * 'Scenario': Specifies the scenario identifier of the respective simulation.
        * 'Random_Seed': Specifies the random seed used for the respective simulation.
        * Optional, if `initial_data` is provided:
            A column 'Initial_Data' that pecifies the index of the initial data set
            used for the respective simulation.
        * Optional, if `groupby` is provided: A column for each "groupby" parameter
            that specifies the search space partition considered for the respective
            simulation.
    """

    _RESULT_VARIABLE = "simulation_result"  # pylint: disable=invalid-name

    @dataclass
    class SimulationResult:
        """A thin wrapper to enable dataframe-valued return values with xyzpy."""

        result: pd.DataFrame

    @xyz.label(var_names=[_RESULT_VARIABLE])
    def simulate(  # pylint: disable=invalid-name
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
                n_exp_iterations=n_exp_iterations,
                initial_data=data,
                groupby=groupby,
                random_seed=Random_Seed,
                impute_mode=impute_mode,
                noise_percent=noise_percent,
            )
        )

    def unpack_simulation_results(array: DataArray) -> pd.DataFrame:
        """Turns the xyzpy simulation results into a flat dataframe."""
        # Convert to dataframe and remove the wrapper layer
        series = array.to_series()
        series = series.apply(lambda x: x.result)

        # Un-nest all simulation results
        dfs = []
        for setting, df_result in series.items():
            df_setting = pd.DataFrame(
                [setting], columns=series.index.names, index=df_result.index
            )
            df_result = pd.concat([df_setting, df_result], axis=1)
            dfs.append(df_result)

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
    baybe_obj: BayBE,
    lookup: Optional[Union[pd.DataFrame, Callable[..., Tuple[float, ...]]]] = None,
    /,
    *,
    batch_quantity: int = 1,
    n_exp_iterations: Optional[int] = None,
    initial_data: Optional[pd.DataFrame] = None,
    groupby: Optional[List[str]] = None,
    random_seed: int = _DEFAULT_SEED,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Scenario simulation for different search space partitions.

    A wrapper around `simulate_experiment` that allows to partition the search space
    into different groups and run separate simulations for all groups where the search
    is restricted to the corresponding partition.

    Parameters
    ----------
    baybe_obj
        See `simulate_experiment`.
    lookup
        See `simulate_experiment`.
    batch_quantity
        See `simulate_experiment`.
    n_exp_iterations
        See `simulate_experiment`.
    initial_data
        See `simulate_experiment`.
    groupby
        The names of the parameters to be used to partition the search space.
        A separate simulation will be conducted for each partition, with the search
        restricted to that partition.
    random_seed
        See `simulate_experiment`.
    impute_mode
        See `simulate_experiment`.
    noise_percent
        See `simulate_experiment`.

    Returns
    -------
    A dataframe like returned from `simulate_experiments`, but with additional
    "groupby columns" (named according to the specified groupby parameters) that
    subdivide the results into the different simulations.
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
        groups = ((None, baybe_obj.searchspace.discrete.exp_rep.reset_index()),)
    else:
        groups = baybe_obj.searchspace.discrete.exp_rep.reset_index().groupby(groupby)

    # Simulate all subgroups
    dfs = []
    for group_id, group in groups:

        # Create a baybe object that focuses only on the current group by excluding
        # off-group configurations from the candidates list
        # TODO: Reconsider if deepcopies are required once [16605] is resolved
        baybe_group = deepcopy(baybe_obj)
        # TODO: Implement SubspaceDiscrete.__len__
        off_group_idx = np.full(
            len(baybe_obj.searchspace.discrete.exp_rep), fill_value=True, dtype=bool
        )
        off_group_idx[group.index.values] = False
        # TODO [16605]: Avoid direct manipulation of metadata
        baybe_group.searchspace.discrete.metadata.loc[
            off_group_idx, "dont_recommend"
        ] = True

        # Run the group simulation
        try:
            df_group = simulate_experiment(
                baybe_group,
                lookup,
                batch_quantity=batch_quantity,
                n_exp_iterations=n_exp_iterations,
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
    baybe_obj: BayBE,
    lookup: Optional[Union[pd.DataFrame, Callable[..., Tuple[float, ...]]]] = None,
    /,
    *,
    batch_quantity: int = 1,
    n_exp_iterations: Optional[int] = None,
    initial_data: Optional[pd.DataFrame] = None,
    random_seed: int = _DEFAULT_SEED,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulates a Bayesian optimization loop.

    The most basic type of simulation. Runs a single execution of the loop either
    for a specified number of steps or until there are no more configurations left
    to be tested.

    Parameters
    ----------
    baybe_obj
        The DOE setting to be simulated.
    lookup
        # TODO: needs refactoring
        The lookup used to close the loop,provided in the form of a dataframe or
        callable that define the targets for the queried parameter settings:
            * A dataframe containing experimental settings and their target results.
            * A callable, providing target values for the given parameter settings.
                The callable is assumed to return either a float or a tuple of floats
                and to accept an arbitrary number of floats as input.
            * 'None' (produces fake results).
    batch_quantity
        The number of recommendations to be queried per iteration.
    n_exp_iterations
        The number of iterations to run the loop. If not specified, the simulation
        proceeds until there are no more testable configurations left.
    initial_data
        The initial measurement data to be ingested before starting the loop.
    random_seed
        The random seed used for the simulation.
    impute_mode
        Specifies how a missing lookup will be handled:
            * 'error': an error will be thrown
            * 'worst': imputation using the worst available value for each target
            * 'best': imputation using the best available value for each target
            * 'mean': imputation using mean value for each target
            * 'random': a random row will be used as lookup
            * 'ignore': the search space is stripped before recommendations are made
                so that unmeasured experiments will not be recommended
    noise_percent
        If not 'None', relative noise in percent of `noise_percent` will be
        applied to the parameter measurements.

    Returns
    -------
    A dataframe ready for plotting, containing the following columns:
        * 'Iteration': corresponds to the DOE iteration (starting at 0)
        * 'Num_Experiments': corresponds to the running number of experiments
            performed (usually x-axis)
        * for each target a column '{targetname}_IterBest': corresponds to the best
            result for that target at the respective iteration
        * for each target a column '{targetname}_CumBest': corresponds to the best
            result for that target up to including respective iteration
        * for each target a column '{targetname}_Measurements': the individual
            measurements obtained for the respective target and iteration
    """
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
    # TODO: Probably, we should add this as a property to BayBE
    will_terminate = (baybe_obj.searchspace.type == SearchSpaceType.DISCRETE) and (
        not baybe_obj.strategy.allow_recommending_already_measured
    )
    if (n_exp_iterations is None) and (not will_terminate):
        raise ValueError(
            "For the specified setting, the experimentation loop can be continued "
            "indefinitely. Hence, `n_exp_iterations` must be explicitly provided."
        )

    # Create a fresh BayBE object and set the corresponding random seed
    # TODO: Reconsider if deepcopies are required once [16605] is resolved
    baybe_obj = deepcopy(baybe_obj)
    set_random_seed(random_seed)

    # Add the initial data
    if initial_data is not None:
        baybe_obj.add_measurements(initial_data)

    # For impute_mode 'ignore', do not recommend space entries that are not
    # available in the lookup
    # TODO [16605]: Avoid direct manipulation of metadata
    if impute_mode == "ignore":
        searchspace = baybe_obj.searchspace.discrete.exp_rep
        missing_inds = searchspace.index[
            searchspace.merge(lookup, how="left", indicator=True)["_merge"]
            == "left_only"
        ]
        baybe_obj.searchspace.discrete.metadata.loc[
            missing_inds, "dont_recommend"
        ] = True

    # Run the DOE loop
    limit = n_exp_iterations or np.inf
    k_iteration = 0
    n_experiments = 0
    dfs = []
    while k_iteration < limit:

        # Get the next recommendations and corresponding measurements
        try:
            measured = baybe_obj.recommend(batch_quantity=batch_quantity)
        except NotEnoughPointsLeftError:
            # TODO: This except block requires a more elegant solution
            strategy = baybe_obj.strategy
            allow_repeated = strategy.allow_repeated_recommendations
            allow_measured = strategy.allow_recommending_already_measured

            # Sanity check: if the variable was True, the except block should have
            # been impossible to reach in the first place
            # TODO: Currently, this is still possible due to bug [15917] though.
            assert not allow_measured

            measured, _ = baybe_obj.searchspace.discrete.get_candidates(
                allow_repeated_recommendations=allow_repeated,
                allow_recommending_already_measured=allow_measured,
            )

            if len(measured) == 0:
                break

        n_experiments += len(measured)
        _look_up_target_values(measured, baybe_obj, lookup, impute_mode)

        # Create the summary for the current iteration and store it
        result = pd.DataFrame(
            [  # <-- this ensures that the internal lists to not get expanded
                {
                    "Iteration": k_iteration,
                    "Num_Experiments": n_experiments,
                    **{
                        f"{target.name}_Measurements": measured[target.name].to_list()
                        for target in baybe_obj.targets
                    },
                }
            ]
        )
        dfs.append(result)

        # Apply optional noise to the parameter measurements
        if noise_percent:
            add_parameter_noise(
                measured,
                baybe_obj.parameters,
                noise_type="relative_percent",
                noise_level=noise_percent,
            )

        # Update the BayBE object
        baybe_obj.add_measurements(measured)

        # Update the iteration counter
        k_iteration += 1

    # Collect the iteration results
    if len(dfs) == 0:
        raise NothingToSimulateError()
    results = pd.concat(dfs, ignore_index=True)

    # Add the instantaneous and running best values for all targets
    for target in baybe_obj.targets:
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
            cum_fun = lambda x: np.array(  # noqa: E731, pylint: disable=C3001
                np.frompyfunc(
                    partial(closer_element, target=match_val),  # pylint: disable=W0640
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
    baybe_obj: BayBE,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
) -> None:
    """
    Fills the target values in the given query dataframe using the provided lookup
    mechanism. See `simulate_from_configs` for details.

    Returns
    -------
    Nothing (the given dataframe is modified in-place).
    """
    # TODO: This function needs another code cleanup and refactoring. In particular,
    #   the different lookup modes should be implemented via multiple dispatch.

    # Extract all target names
    target_names = [t.name for t in baybe_obj.targets]

    # If no lookup is provided, invent some fake results
    if lookup is None:
        add_fake_results(queries, baybe_obj)

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
        if measured_targets.shape[1] != len(baybe_obj.targets):
            raise AssertionError(
                "If you use an analytical function as lookup, make sure "
                "the configuration has the right amount of targets "
                "specified."
            )
        for k_target, target in enumerate(baybe_obj.targets):
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
                match_vals = _impute_lookup(row, lookup, baybe_obj.targets, impute_mode)

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
    """
    Performs data imputation (or raises errors, depending on the mode) for missing
    lookup values.

    Parameters
    ----------
    row : pd.Series
        The data that should be matched with the lookup table.
    lookup: pd.DataFrame
        The lookup table.
    targets: List[NumericalTarget]
        Targets from the BayBE object, providing the required mode information.
    mode : "error" | "worst" | "best" | "mean" | "random" | "ignore"
        See `simulate_from_configs`.

    Returns
    -------
    np.ndarray
        The filled-in lookup results.
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
