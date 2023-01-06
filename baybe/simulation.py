"""
Provides functions to simulate a Bayesian DOE with BayBE given a lookup.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from functools import partial
from typing import Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tqdm import trange

from baybe.core import BayBE, BayBEConfig
from baybe.utils import (
    add_fake_results,
    add_parameter_noise,
    closer_element,
    closest_element,
    name_to_smiles,
)

if TYPE_CHECKING:
    from .targets import NumericalTarget

log = logging.getLogger(__name__)


def simulate_from_configs(
    config_base: dict,
    batch_quantity: int,
    n_exp_iterations: int,
    n_mc_iterations: int,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    config_variants: Optional[Dict[str, dict]] = None,
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulates Monte Carlo runs of experiments via lookup for different configurations.

    Parameters
    ----------
    config_base : dict
        Contains a base configuration that is shared between all configuration variants.
    batch_quantity : int
        Number of recommendations returned per experimental DOE iteration.
    n_exp_iterations : int
        Number of experimental DOE iterations that should be simulated.
    n_mc_iterations : int
        Number of Monte Carlo runs that should be used in the simulation. Each run will
        have a different random seed.
    lookup : Union[pd.DataFrame, Callable] (optional)
        Defines the targets for the queried parameter settings. Can be:
            * A dataframe containing experimental settings and their target results.
            * A callable, providing target values for the given parameter settings.
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
    config_variants : dict
        A dict whose keys are the names of the different configuration variants and
        whose items are configurations that specify the variant. For instance, a
        variant can define a different strategy or different parameter encoding.
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

    # If no configuration variants are provided, use the base configuration as the
    # only "variant" to be simulated
    if config_variants is None:
        config_variants = {"Simulation": config_base}

    # Create a dataframe to store the simulation results
    results = pd.DataFrame()

    # Simulate all configuration variants
    for variant_name, variant_config in config_variants.items():
        # Create a dataframe to store the results for the current variant
        results_var = pd.DataFrame()

        # Use the configuration of the current variant
        config_dict = deepcopy(config_base)
        config_dict.update(variant_config)

        # Run all Monte Carlo repetitions
        for k_mc in (pbar := trange(n_mc_iterations)):
            # Show the simulation progress
            pbar.set_description(variant_name)

            # Create a BayBE object with a new random seed
            # IMPROVE: Potential speedup by copying the BayBE object + overwriting seed.
            #   Requires a clean way to change the seed of the object without accessing
            #   its members directly.
            config_dict["random_seed"] = 1337 + k_mc
            config = BayBEConfig(**config_dict)
            baybe_obj = BayBE(config)

            # For impute_mode 'ignore', do not recommend space entries that are not
            # available in the lookup
            # IMPROVE: Avoid direct manipulation of the searchspace members
            if impute_mode == "ignore":
                searchspace = baybe_obj.searchspace.exp_rep
                missing_inds = searchspace.index[
                    searchspace.merge(lookup, how="left", indicator=True)["_merge"]
                    == "left_only"
                ]
                baybe_obj.searchspace.metadata.loc[
                    missing_inds, "dont_recommend"
                ] = True

            # Run all experimental iterations
            results_mc = _simulate_experiment(
                baybe_obj,
                batch_quantity,
                n_exp_iterations,
                lookup,
                impute_mode,
                noise_percent,
            )

            # Add the random seed information and append the results
            results_mc.insert(0, "Random_Seed", config_dict["random_seed"])
            results_var = pd.concat([results_var, results_mc])

        # Add the variant information and append the results
        results_var.insert(0, "Variant", variant_name)
        results = pd.concat([results, results_var])

    return results.reset_index(drop=True)


def _simulate_experiment(
    baybe_obj: BayBE,
    batch_quantity: int,
    n_exp_iterations: int,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
) -> pd.DataFrame:
    """
    Simulates a single experimental DOE loop. See `simulate_from_configs` for details.
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
        baybe_obj.add_results(measured)

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
            cum_fun = lambda x: np.array(  # noqa: E731
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
        measured_targets = (
            queries.drop(columns=target_names)
            .apply(lambda x: lookup(*x.values), axis=1)
            .to_frame()
        )
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
        for _, row in queries.drop(columns=target_names).iterrows():
            # IMPROVE: to the entire matching at once via a merge
            ind = lookup[
                (lookup.loc[:, row.index] == row).all(axis=1, skipna=False)
            ].index.values

            if len(ind) > 1:
                # More than two instances of this parameter combination
                # have been measured
                log.warning(
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
                            (lookup[target.name] - np.mean(target.bounds))
                            .abs()
                            .idxmax(),
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
                            (lookup[target.name] - np.mean(target.bounds))
                            .abs()
                            .idxmin(),
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


def simulate_from_data(
    config_base: dict,
    n_exp_iterations: int,
    n_mc_iterations: int,
    parameter_types: Dict[str, List[dict]],
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "ignore",
    noise_percent: Optional[float] = None,
    batch_quantity: int = 5,
) -> pd.DataFrame:
    """
    Wrapper around `simulate_from_configs` that allows to vary parameter types.

    Parameters
    ----------
    config_base : dict
        See `simulate_from_configs`.
    n_exp_iterations : int
        See `simulate_from_configs`.
    n_mc_iterations : int
        See `simulate_from_configs`.
    parameter_types : dict
        Dictionary where keys are variant names. Entries contain lists which have
        dictionaries describing the parameter configurations except their values.
    lookup : Union[pd.DataFrame, Callable] (optional)
        See `simulate_from_configs`.
    impute_mode : "error" | "worst" | "best" | "mean" | "random" | "ignore"
        See `simulate_from_configs`.
    noise_percent : float (optional)
        See `simulate_from_configs`.
    batch_quantity : int
        See `simulate_from_configs`.

    Returns
    -------
    pd.DataFrame
        See `simulate_from_configs`.

    Example
    -------
    parameter_types = {
        "Variant1": [
            {"name": "Param1", "type": "CAT", "encoding": "INT"},
            {"name": "Param2", "type": "CAT"},
        ],
        "Variant2": [
            {"name": "Param1", "type": "NUM_DISCRETE"},
            {"name": "Param2", "type": "SUBSTANCE", "encoding": "RDKIT"},
        ],
    }
    """
    # TODO: this function needs another code cleanup and refactoring

    # Create the parameter configs from lookup data
    config_variants = {}
    for variant_name, parameter_list in parameter_types.items():
        variant_parameter_configs = []
        for param_dict in parameter_list:
            assert "name" in param_dict, (
                f"Parameter dictionary must contain the key 'name'. "
                f"Parsed dictionary: {param_dict}"
            )
            assert "type" in param_dict, (
                f"Parameter dictionary must contain the key 'type'. "
                f"Parsed dictionary: {param_dict}"
            )

            param_vals = list(lookup[param_dict["name"]].unique())
            parameter_config = deepcopy(param_dict)

            if param_dict["type"] == "SUBSTANCE":
                smiles = [name_to_smiles(itm) for itm in param_vals]
                if any(s == "" for s in smiles):
                    raise ValueError(
                        f"For the parameter {param_dict['name']} in 'SUBSTANCE' type "
                        f"not all SMILES could be retrieved from the NCI. The "
                        f"problematic substances are "
                        f"{[name for k,name in enumerate(param_vals) if smiles[k]=='']}"
                    )
                dat = dict(zip(param_vals, smiles))
                parameter_config["data"] = dat
            elif param_dict["type"] == "CUSTOM":
                raise ValueError(
                    f"Custom parameter types are not supported for "
                    f"simulation with automatic parameter value inference "
                    f"(encountered in parameter {param_dict})."
                )
            else:
                parameter_config["values"] = param_vals

            variant_parameter_configs.append(parameter_config)

        config_variants[variant_name] = {"parameters": variant_parameter_configs}

    print("### Inferred configurations:")
    print(config_variants)

    results = simulate_from_configs(
        config_base=config_base,
        batch_quantity=batch_quantity,
        n_exp_iterations=n_exp_iterations,
        n_mc_iterations=n_mc_iterations,
        lookup=lookup,
        impute_mode=impute_mode,
        config_variants=config_variants,
        noise_percent=noise_percent,
    )

    return results
