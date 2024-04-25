"""Target lookup mechanisms."""


from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Literal, Optional, Union

import numpy as np
import pandas as pd

from baybe.campaign import Campaign
from baybe.targets.enum import TargetMode
from baybe.utils.dataframe import add_fake_results

if TYPE_CHECKING:
    from baybe.targets import NumericalTarget

_logger = logging.getLogger(__name__)


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
        lookup: The lookup mechanism.
            See :func:`baybe.simulation.scenarios.simulate_scenarios` for details.
        impute_mode: The used impute mode.
            See :func:`baybe.simulation.scenarios.simulate_scenarios` for details.

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
    targets: list[NumericalTarget],
    mode: Literal["error", "best", "worst", "mean", "random"] = "error",
) -> np.ndarray:
    """Perform data imputation for missing lookup values.

    Depending on the chosen mode, this might raise errors instead.

    Args:
        row: The data that should be matched with the lookup data frame.
        lookup: The lookup data frame.
        targets: The campaign targets, providing the required mode information.
        mode: The used impute mode.
            See :func:`baybe.simulation.scenarios.simulate_scenarios` for details.

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
            if target.mode is TargetMode.MAX:
                worst_vals.append(lookup.loc[:, target.name].min().flatten()[0])
            elif target.mode is TargetMode.MIN:
                worst_vals.append(lookup.loc[:, target.name].max().flatten()[0])
            if target.mode is TargetMode.MATCH:
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
            if target.mode is TargetMode.MAX:
                best_vals.append(lookup.loc[:, target.name].max().flatten()[0])
            elif target.mode is TargetMode.MIN:
                best_vals.append(lookup.loc[:, target.name].min().flatten()[0])
            if target.mode is TargetMode.MATCH:
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
