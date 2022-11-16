"""
Provides functions to simulate a Bayesian DOE with BayBE given a lookup.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tqdm import trange

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise, name_to_smiles

if TYPE_CHECKING:
    from .targets import NumericalTarget

log = logging.getLogger(__name__)


def simulate_from_configs(
    config_base: dict,
    n_exp_iterations: int,
    n_mc_iterations: int,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
    noise_percent: Optional[float] = None,
    batch_quantity: int = 5,
    config_variants: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Simulates Monte Carlo runs of experiments via lookup. Can include different
    configuration variants.

    Parameters
    ----------
    config_base : dict
        Contains a base configuration that is shared between all variants.
    n_exp_iterations : int
        Number of DOE iterations that should be simulated.
    n_mc_iterations : int
        Number of Monte Carlo runs that should be used in the simulation. Each run will
        have a different random seed.
    lookup : Union[pd.DataFrame, Callable] (optional)
        Defines the targets for the recommended parameter settings. Can be:
            * A dataframe with all possible experiments and their target results.
            * A callable, providing the target values for the given parameters.
            * 'None' (produces fake results).
    impute_mode : "error" | "worst" | "best" | "mean" | "random" | "ignore"
        Specifies how a missing lookup will be handled:
        * 'error': an error will be thrown.
        * 'worst': imputation using the worst available value for each target.
        * 'best': imputation using the best available value for each target.
        * 'mean': imputation using mean value for each target.
        * 'random': a random row will be used as lookup.
        * 'ignore': the search space is stripped before recommendations are made
            so that unmeasured experiments cannot be recommended.
    noise_percent : float (optional)
        If this is not 'None', relative noise in percent of `noise_percent` will be
        applied to the parameter measurements.
    batch_quantity : int
        Number of recommendation returned per iteration.
    config_variants : dict
        A dict whose keys are the names of the different configuration variants and
        whose items are configurations that specify the variant. For instance, a
        variant can define a different strategy or different parameter encoding.

    Returns
    -------
    pd.DataFrame
        A dataframe supposed to be plotted with seaborn. It will contain a column
        'Variant' which derives from the dict keys used in config_variants, a column
        'Num_Experiments' which corresponds to the number of experiments performed
        (usually x-axis), a column 'Iteration' corresponding to the DOE iteration
        (starting at 0), for each target a column '{targetname}_IterBest' corresponding
        to the best result for that target at that iteration, and for each target a
        column '{targetname}_CumBest' corresponding to the best result for that target
        up to including that iteration. The individual measurements are stored in
        'Measurements_{targetname}'.

    Examples
    --------
    results = simulate_from_configs(
        config_base=config_dict_base,
        lookup=lookup,
        impute_mode="ignore",
        n_exp_iterations=20,
        n_mc_iterations=5,
        batch_quantity=3,
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
    # TODO: this function needs another code cleanup and refactoring

    # validate input
    if not (isinstance(lookup, (pd.DataFrame, Callable)) or (lookup is None)):
        raise TypeError(
            "The lookup can either be 'None', a pandas dataframe or a callable."
        )

    if config_variants is None:
        config_variants = {"Simulation": config_base}

    results = []

    # Simulate all configuration variants
    for variant_name, variant_config in config_variants.items():
        config_dict = deepcopy(config_base)
        config_dict.update(variant_config)

        # Run all Monte Carlo repetitions
        for k_mc in (pbar := trange(n_mc_iterations)):
            # IMPROVE: potential speedup by copying the BayBE object + overwriting seed
            pbar.set_description(variant_name)
            config_dict["random_seed"] = 1337 + k_mc
            config = BayBEConfig(**config_dict)
            baybe_obj = BayBE(config)
            target_names = [t.name for t in baybe_obj.targets]

            # Initialize variables to store cumulative results
            best_mc_results = {}
            for target in baybe_obj.targets:
                if target.mode == "MAX":
                    best_mc_results[target.name] = -np.inf
                elif target.mode == "MIN":
                    best_mc_results[target.name] = np.inf
                elif target.mode == "MATCH":
                    best_mc_results[target.name] = np.nan

            # For impute_mode 'ignore', mark search space entries that are not
            # available in the lookup.
            if impute_mode == "ignore":
                if not isinstance(lookup, pd.DataFrame):
                    raise ValueError(
                        "Impute mode 'ignore' is only available for dataframe lookups."
                    )
                searchspace = baybe_obj.searchspace.exp_rep
                missing_inds = searchspace.index[
                    searchspace.merge(lookup, how="left", indicator=True)["_merge"]
                    == "left_only"
                ]
                baybe_obj.searchspace.metadata.loc[
                    missing_inds, "dont_recommend"
                ] = True

            # Run all experimental iterations
            for k_iteration in range(n_exp_iterations):

                # Get recommendations
                measured = baybe_obj.recommend(batch_quantity=batch_quantity)

                # Get the target values from the lookup
                if lookup is None:
                    # Add fake results
                    add_fake_results(measured, baybe_obj)

                elif isinstance(lookup, Callable):
                    # Get results via callable
                    measured_targets = (
                        measured.drop(columns=target_names)
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
                        measured[target.name] = measured_targets.iloc[:, k_target]

                elif isinstance(lookup, pd.DataFrame):
                    # Get results via dataframe lookup (works only for exact matches)
                    # IMPROVE: Although its not too important for a simulation, this
                    #  could also be implemented for approximate matches

                    all_match_vals = []
                    for _, row in measured.drop(columns=target_names).iterrows():
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
                            match_vals = lookup.loc[
                                np.random.choice(ind), target_names
                            ].values

                        elif len(ind) < 1:
                            # Parameter combination cannot be looked up and needs to be
                            # imputed.
                            if impute_mode == "ignore":
                                raise AssertionError(
                                    "Something went wrong for impute_mode 'ignore'. "
                                    "It seems the search space was not correctly "
                                    "reduced before recommendations were generated."
                                )
                            match_vals = _impute_lookup(
                                row, lookup, baybe_obj.targets, impute_mode
                            )

                        else:
                            # Exactly one match has been found
                            match_vals = lookup.loc[ind[0], target_names].values

                        # Collect the matches
                        all_match_vals.append(match_vals)

                    # Add the lookup values
                    measured.loc[:, target_names] = np.asarray(all_match_vals)

                # Prepare the results summary for the current iteration
                tempres = {
                    "Variant": variant_name,
                    "Iteration": k_iteration,
                    "Num_Experiments": (k_iteration + 1) * batch_quantity,
                }

                # Store iteration and cumulative best target values
                for target in baybe_obj.targets:
                    tempres[f"Measurements_{target.name}"] = measured[
                        target.name
                    ].to_list()

                    if target.mode == "MAX":
                        best_iter = measured[target.name].max()
                        if best_iter > best_mc_results[target.name]:
                            best_mc_results[target.name] = best_iter
                    elif target.mode == "MIN":
                        best_iter = measured[target.name].min()
                        if best_iter < best_mc_results[target.name]:
                            best_mc_results[target.name] = best_iter
                    elif target.mode == "MATCH":
                        matchval = np.mean(target.bounds)
                        best_iter = measured.loc[
                            (measured[target.name] - matchval).abs().idxmin(),
                            target.name,
                        ]
                        if np.isnan(best_mc_results[target.name]):
                            best_mc_results[target.name] = best_iter
                        else:
                            if np.abs(best_iter - matchval) < np.abs(
                                best_mc_results[target.name] - matchval
                            ):
                                best_mc_results[target.name] = best_iter

                    tempres[f"{target.name}_CumBest"] = best_mc_results[target.name]
                    tempres[f"{target.name}_IterBest"] = best_iter

                # Append the results of the current iteration
                results.append(tempres)

                # Apply optional noise to the parameter measurements
                if noise_percent:
                    add_parameter_noise(
                        measured,
                        baybe_obj,
                        noise_type="relative_percent",
                        noise_level=noise_percent,
                    )

                # Add the results of the current iteration to the BayBE object
                baybe_obj.add_results(measured)

    # Convert the results to a dataframe
    results = pd.DataFrame(results)

    return results


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
        lookup=lookup,
        n_exp_iterations=n_exp_iterations,
        n_mc_iterations=n_mc_iterations,
        batch_quantity=batch_quantity,
        noise_percent=noise_percent,
        impute_mode=impute_mode,
        config_variants=config_variants,
    )

    return results
