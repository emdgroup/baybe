"""
Provides functions to simulate a Bayesian DOE with BayBE given a lookup
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, Dict, List, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_noise

if TYPE_CHECKING:
    from .targets import NumericalTarget

log = logging.getLogger(__name__)


def simulate_from_configs(
    config_base: dict,
    n_exp_iterations: int,
    n_mc_iterations: int,
    lookup: Optional[Union[pd.DataFrame, Callable]] = None,
    impute_lookup: Literal["error", "worst", "best", "mean", "random"] = "error",
    noise_percent: Optional[float] = None,
    batch_quantity: int = 5,
    config_variants: Optional[Dict[str, dict]] = None,
) -> pd.DataFrame:
    """
    Simulates Monte-Carlo runs of experiments via lookup. Can include different
    configuration variants

    Parameters
    ----------
    config_base : dict
        Containing a base configuration that is the same between all variants
    n_exp_iterations : int
        Number of DOE iterations that should be simulated
    n_mc_iterations : int
        Number of Monte-Carlo runs that should be used in the simulation. Each run will
        have a different random seed
    lookup : pd.DataFrame
        A dataframe with all possible experiments and their target results
    impute_lookup : str
        If this is 'error' a missing looup value will result in an error. In case of
        'worst' the missing lookups will be the worst available value for that target.
        In case of 'best' the missing lookups will be the best available value for that
        target. In case of 'mean' the missing lookup is the mean of all available
        values for that target. In case of 'random' a random row will be used as lookup.
    noise_percent : None or float
        If this is not None relative noise in percent of noise_percent will be added
        to measurements
    batch_quantity : int
        How many recommendations should be drawn per iteration
    config_variants : dict
        The keys are the names of the different configuration variants. The items are
        configurations that specify the variant. For instance they can define a
        different strategy or different parameter encodings.

    Returns
    -------
    pd.DataFrame
        A dataframe supposed to be plotted with seaborn. It will contain a column
        'Variant' which derives from the dict keys suedd in config_variants, a column
        'Num_Experiments' which corresponds to the number of experiments performed
        (usually x-axis), a columns 'Iteration' corresponding to the DOE Iteration
        (starting at 0), for each target a column '{targetname}_IterBest' corresponding
        to the best result for that target of that iteration; and for each target a
        column '{targetname}_CumBest' corresponding to the best result for that target
        up to including that iteration.

    Examples
    --------
    results = simulate_variants(
        config_base=config_dict_base,
        lookup=None,
        n_exp_iterations=15,
        n_mc_iterations=50,
        batch_quantity=5,
        config_variants={
            "GP | Mordred": config_dict_v1,
            "GP | RDKit": config_dict_v2,
            "GP | FP": config_dict_v3,
            "GP | OHE": config_dict_v4,
            "RANDOM": config_dict_v5,
        },
    )
    sns.lineplot(data=results, x="Num_Experiments", y="Target_Best", hue="Variant")
    """

    if config_variants is None:
        config_variants = {"Simulation": config_base}

    results = []

    for variant_name, variant_config in config_variants.items():
        print(variant_name, end="\n")
        config_dict = deepcopy(config_base)
        config_dict.update(variant_config)

        for k_mc in tqdm(range(n_mc_iterations)):
            config_dict["random_seed"] = 1337 + k_mc
            config = BayBEConfig(**config_dict)
            baybe_obj = BayBE(config)
            target_names = [t.name for t in baybe_obj.targets]

            best_mc_results = {}
            for target in baybe_obj.targets:
                if target.mode == "MAX":
                    best_mc_results[target.name] = -np.inf
                elif target.mode == "MIN":
                    best_mc_results[target.name] = np.inf
                elif target.mode == "MATCH":
                    best_mc_results[target.name] = np.inf

            for k_iteration in range(n_exp_iterations):
                # Get recommendation
                measured = baybe_obj.recommend(batch_quantity=batch_quantity)

                # Lookup
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
                            "If you look up from an analytical function make sure the "
                            "configuration has the right amount of targets configured"
                        )
                    for k_target, target in enumerate(baybe_obj.targets):
                        measured[target.name] = measured_targets.iloc[:, k_target]
                elif isinstance(lookup, pd.DataFrame):
                    # Get results via dataframe lookup
                    # This only works for exact match here
                    # IMPROVE Although its not too important for a simulation, this
                    #  could also be implemented for approximate matches

                    all_match_vals = []
                    for _, row in measured.drop(columns=target_names).iterrows():
                        ind = lookup[
                            (lookup.loc[:, row.index] == row).all(axis=1, skipna=False)
                        ].index.values

                        if len(ind) > 1:
                            # More than two instances of this parameter combination
                            # have been measured
                            log.warning(
                                "The lookup rows with indexes %s seem to be "
                                "duplicates regarding parameter values. Choosing a "
                                "random one",
                                ind,
                            )
                            match_vals = lookup.loc[
                                np.random.choice(ind), target_names
                            ].values
                        elif len(ind) < 1:
                            # Parameter combination cannot be looked up and needs to be
                            # imputed
                            match_vals = _impute_lookup(
                                row, lookup, baybe_obj.targets, impute_lookup
                            )
                        else:
                            # Exactly one match has been found
                            match_vals = lookup.loc[ind[0], target_names].values
                        all_match_vals.append(match_vals)

                    measured.loc[:, target_names] = np.asarray(all_match_vals)
                else:
                    raise TypeError(
                        "The lookup can either be None, a pandas dataframe or a "
                        "callable function"
                    )

                # Remember best results of the iteration and best results up to
                # (including) this iteration
                tempres = {
                    "Variant": variant_name,
                    "Iteration": k_iteration,
                    "Num_Experiments": (k_iteration + 1) * batch_quantity,
                }

                # Remember iteration and cumulative best target values
                for target in baybe_obj.targets:
                    if target.mode == "MAX":
                        best_iter = measured[target.name].max()
                        tempres[f"{target.name}_IterBest"] = best_iter

                        if best_iter > best_mc_results[target.name]:
                            best_mc_results[target.name] = best_iter
                        tempres[f"{target.name}_CumBest"] = best_mc_results[target.name]
                    elif target.mode == "MIN":
                        best_iter = measured[target.name].min()
                        tempres[f"{target.name}_IterBest"] = best_iter

                        if best_iter < best_mc_results[target.name]:
                            best_mc_results[target.name] = best_iter
                        tempres[f"{target.name}_CumBest"] = best_mc_results[target.name]
                    elif target.mode == "MATCH":
                        matchval = np.mean(target.bounds)
                        best_iter = measured.loc[
                            (measured[target.name] - matchval).abs().idxmin(),
                            target.name,
                        ]
                        tempres[f"{target.name}_IterBest"] = best_iter

                        if (
                            measured.loc[
                                measured[target.name] == best_iter, target.name
                            ].values[0]
                            < measured.loc[
                                measured[target.name] == best_mc_results[target.name],
                                target.name,
                            ].values[0]
                        ):
                            best_mc_results[target.name] = best_iter
                        tempres[f"{target.name}_CumBest"] = best_mc_results[target.name]

                results.append(tempres)

                # Add results to BayBE object
                if noise_percent:
                    add_noise(
                        measured,
                        baybe_obj,
                        noise_type="relative_percent",
                        noise_level=noise_percent,
                    )

                baybe_obj.add_results(measured)

    results = pd.DataFrame(results)

    return results


def _impute_lookup(
    row: pd.Series,
    lookup: pd.DataFrame,
    targets: List[NumericalTarget],
    mode: Literal["error", "best", "worst", "mean", "random"] = "error",
):
    """
    If no lookup value is found this function will find other values to take depending
     on the mode or raise an error.

    Parameters
    ----------
    row: pd.Series
        The row that should be matched in lookup
    lookup: pd. DataFrame
        A lookup dataframe
    targets: list of BayeBE Target's
        Targets from the BayBE object so their respective mdoes can be considered
    mode: literal
        Describing the fill-mode that should be applied

    Returns
    np.array
        The filled-in lookup results
    -------

    """
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
            f"Cannot match the recommended row {row} to any of"
            f" the rows in the lookup"
        )

    return match_vals
