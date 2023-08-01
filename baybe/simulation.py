"""
Provides functions to simulate a Bayesian DOE with BayBE given a lookup.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from baybe.core import BayBE
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


def simulate_scenarios(
    scenarios: Dict[str, BayBE],
    batch_quantity: int,
    n_exp_iterations: int,
    n_mc_iterations: Optional[int] = None,
    initial_data: Optional[List[pd.DataFrame]] = None,
    lookup: Optional[
        Union[pd.DataFrame, Callable[[float, ...], Union[float, Tuple[float, ...]]]]
    ] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulates Monte Carlo runs of experiments via lookup for different configurations.

    Parameters
    ----------
    scenarios : Dict[str, BayBE]
        BayBE objects (dict-values) and corresponding scenario names (dict-keys) to be
        simulated.
    batch_quantity : int
        Number of recommendations returned per experimental DOE iteration.
    n_exp_iterations : int
        Number of experimental DOE iterations that should be simulated.
    n_mc_iterations : int (optional)
        Number of Monte Carlo runs that should be used in the simulation. Each run will
        have a different random seed. Must be 'None' if `initial_data` is specified.
    initial_data : List[pd.DataFrame] (optional)
        A collection of initial data sets. The experiment is repeated once with each
        data set in the collection for each configuration. Must be 'None' if
        `n_mc_iterations` is specified.
    lookup : pd.DataFrame or callable that takes one or more floats and returns one
        or more floats (optional)
        Defines the targets for the queried parameter settings. Can be:
            * A dataframe containing experimental settings and their target results.
            * A callable, providing target values for the given parameter settings.
                This callable is assumed to return either a float or a tuple of floats
                and to accept an arbitrary number of floats as input.
            * 'None' (produces fake results).
    impute_mode : "error" | "worst" | "best" | "mean" | "random" | "ignore"
        Specifies how a missing lookup will be handled:
        * 'error': an error will be thrown
        * 'worst': imputation using the worst available value for each target
        * 'best': imputation using the best available value for each target
        * 'mean': imputation using mean value for each target
        * 'random': a random row will be used as lookup
        * 'ignore': the search space is stripped before recommendations are made
            so that unmeasured experiments will not be recommended
    noise_percent : float (optional)
        If this is not 'None', relative noise in percent of `noise_percent` will be
        applied to the parameter measurements.

    Returns
    -------
    pd.DataFrame
        A dataframe ready for plotting, containing the following columns:
            * 'Variant': corresponds to the dict keys used in `config_variants`
            * 'Random_Seed': the random seed used for the respective simulation
            * 'Num_Experiments': corresponds to the running number of experiments
                performed (usually x-axis)
            * 'Iteration': corresponds to the DOE iteration (starting at 0)
            *  for each target a column '{targetname}_IterBest': corresponds to the best
                result for that target at the respective iteration
            *  for each target a column '{targetname}_CumBest': corresponds to the best
                result for that target up to including respective iteration
            * '{targetname}_Measurements': the individual measurements obtained for the
                respective target and iteration

    Examples
    --------
    results = simulate_from_configs(
        config_base=config_dict_base,
        batch_quantity=3,
        n_exp_iterations=20,
        n_mc_iterations=5,
        lookup=lookup,
        impute_mode="ignore",
        config_variants={
            "GP | Mordred": config_dict_v1,
            "GP | RDKit": config_dict_v2,
            "GP | FP": config_dict_v3,
            "GP | OHE": config_dict_v4,
            "RANDOM": config_dict_v5,
        },
    )
    sns.lineplot(data=results, x="Num_Experiments", y="Target_CumBest", hue="Variant")
    """
    # Validate the iteration specification
    if not (n_mc_iterations is None) ^ (initial_data is None):
        raise ValueError(
            "Exactly one of 'n_mc_iterations' and 'initial_data' can take a value."
        )

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

    # Create a dataframe to store the simulation results
    results = pd.DataFrame()

    # Simulate all configuration variants
    for scenario_name, baybe_template in scenarios.items():

        # Create a dataframe to store the results for the current variant
        results_var = pd.DataFrame()

        # Create an iterator for repeating the experiment
        pbar = trange(n_mc_iterations) if initial_data is None else tqdm(initial_data)

        # Run all experiment repetitions
        for k_mc, data in enumerate(pbar):
            # Show the simulation progress
            pbar.set_description(scenario_name)

            # Create a fresh BayBE object and set the corresponding random seed
            baybe = deepcopy(baybe_template)
            random_seed = 1337 + k_mc
            set_random_seed(random_seed)

            # Add the initial data
            if initial_data is not None:
                baybe.add_measurements(data)

            # For impute_mode 'ignore', do not recommend space entries that are not
            # available in the lookup
            # IMPROVE: Avoid direct manipulation of the searchspace members
            if impute_mode == "ignore":
                searchspace = baybe.searchspace.discrete.exp_rep
                missing_inds = searchspace.index[
                    searchspace.merge(lookup, how="left", indicator=True)["_merge"]
                    == "left_only"
                ]
                baybe.searchspace.discrete.metadata.loc[
                    missing_inds, "dont_recommend"
                ] = True

            # Run all experimental iterations
            results_mc = _simulate_experiment(
                baybe,
                batch_quantity,
                n_exp_iterations,
                lookup,
                impute_mode,
                noise_percent,
            )

            # Add the random seed information and append the results
            results_mc.insert(0, "Random_Seed", random_seed)
            results_var = pd.concat([results_var, results_mc])

        # Add the variant information and append the results
        results_var.insert(0, "Variant", scenario_name)
        results = pd.concat([results, results_var])

    return results.reset_index(drop=True)


def _simulate_experiment(
    baybe_obj: BayBE,
    batch_quantity: int,
    n_exp_iterations: int,
    lookup: Optional[Union[pd.DataFrame, Callable[..., Tuple[float, ...]]]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulates a single experimental DOE loop. See `simulate_from_configs` for details.
    Note that the type hint Callable[..., Tuple[float, ...]] means that the callable
    accepts any number of arguments and returns either a single or a tuple of floats.
    The inputs however also need to be floats!
    """
    # Create a dataframe to store the simulation results
    results = pd.DataFrame()

    # Run the DOE loop
    for k_iteration in range(n_exp_iterations):
        # Get the next recommendations and corresponding measurements
        measured = baybe_obj.recommend(batch_quantity=batch_quantity)
        _look_up_target_values(measured, baybe_obj, lookup, impute_mode)

        # Create the summary for the current iteration and store it
        results_iter = pd.DataFrame(
            [  # <-- this ensures that the internal lists to not get expanded
                {
                    "Iteration": k_iteration,
                    "Num_Experiments": (k_iteration + 1) * batch_quantity,
                    **{
                        f"{target.name}_Measurements": measured[target.name].to_list()
                        for target in baybe_obj.targets
                    },
                }
            ]
        )
        results = pd.concat([results, results_iter])

        # Apply optional noise to the parameter measurements
        if noise_percent:
            add_parameter_noise(
                measured,
                baybe_obj,
                noise_type="relative_percent",
                noise_level=noise_percent,
            )

        # Update the BayBE object
        baybe_obj.add_measurements(measured)

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

    return results.reset_index(drop=True)


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
