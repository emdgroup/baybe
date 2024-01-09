"""Target lookup mechanisms."""

import warnings
from typing import Callable, Collection, List, Literal, Union

import numpy as np
import pandas as pd

from baybe.targets import NumericalTarget, TargetMode
from baybe.targets.base import Target
from baybe.utils import add_fake_results

CallableLookup = Callable[[pd.DataFrame], pd.DataFrame]
"""A callable that can be used to retrieve target values for given parameter
configurations."""


def look_up_targets(
    queries: pd.DataFrame,
    targets: List[Target],
    lookup: Union[pd.DataFrame, CallableLookup, None] = None,
    impute_mode: Literal["error", "worst", "best", "mean", "random"] = "error",
) -> None:
    """Fill the target values for given parameter configurations via a lookup mechanism.

    Note that this does not create a new dataframe but modifies ``queries`` in-place.

    Args:
        queries: A dataframe containing the parameter configurations.
        targets: The targets whose values should be filled.
        lookup: An optional lookup mechanism, provided in the form of a dataframe
            or callable, that defines the targets for the queried parameter settings.
            If omitted, fake values will be used.

            - pd.DataFrame: Each row contains the one parameter configuration and the
              corresponding target values.
            - callable: A callable that can be used to retrieve target values for given
              parameter configurations. Expects a dataframe where each row corresponds
              to one parameter configuration (columns names being the parameter names)
              and must return a dataframe containing the corresponding target values
              (column names being the target names), where each row holds the targets
              for the corresponding parameter row.
        impute_mode: Specifies how a missing lookup will be handled (only relevant
            for dataframe lookups). See :func:`baybe.simulation.simulate_experiment`
            for details.

    Raises:
        AssertionError: If an analytical function is used and an incorrect number of
            targets was specified.
    """
    # If no lookup is provided, invent some fake values
    if lookup is None:
        add_fake_results(queries, targets)

    # Compute the target values via a callable
    elif isinstance(lookup, Callable):
        _lookup_targets_from_callable(queries, [t.name for t in targets], lookup)

    # Get the results via dataframe lookup (works only for exact matches)
    # IMPROVE: Although its not too important for a simulation, this
    #  could also be implemented for approximate matches
    elif isinstance(lookup, pd.DataFrame):
        _lookup_targets_from_dataframe(queries, targets, lookup, impute_mode)


def _lookup_targets_from_callable(
    queries: pd.DataFrame,
    target_names: Collection[str],
    lookup: CallableLookup,
) -> None:
    """Look up target values via a callable."""
    # Evaluate the callable
    responses = lookup(queries)

    # Assert that all targets are contained in the response
    if not (exp := set(target_names)).issubset(act := set(responses)):
        raise ValueError(
            f"The provided lookup callable yielded values for the labels {act} but "
            f"required are values for {exp}."
        )

    # Insert the target values in-place
    queries[responses.columns] = responses


def _lookup_targets_from_dataframe(
    queries: pd.DataFrame,
    targets: List[Target],
    lookup: CallableLookup,
    impute_mode: Literal["error", "worst", "best", "mean", "random"] = "error",
) -> None:
    """Look up target values from a dataframe."""
    target_names = [t.name for t in targets]
    all_match_vals = []

    # IMPROVE: speed up the matching using a join
    for _, row in queries.iterrows():
        ind = lookup[
            (lookup.loc[:, row.index] == row).all(axis=1, skipna=False)
        ].index.values

        if len(ind) > 1:
            # More than one instance of this parameter combination
            # have been measured
            warnings.warn(
                f"The lookup rows with indexes {ind} seem to be "
                f"duplicates regarding parameter values. Choosing a "
                f"random one.",
                UserWarning,
            )
            match_vals = lookup.loc[np.random.choice(ind), target_names]

        elif len(ind) < 1:
            # Parameter combination cannot be looked up and needs to be
            # imputed.
            if impute_mode == "ignore":
                raise AssertionError(
                    "Something went wrong for impute_mode 'ignore'. "
                    "It seems the search space was not correctly "
                    "reduced before recommendations were generated."
                )
            match_vals = _impute_lookup(row, lookup, targets, impute_mode)

        else:
            # Exactly one match has been found
            match_vals = lookup.loc[ind[0], target_names]

        # Collect the matches
        all_match_vals.append(match_vals)

    # Add the lookup values
    queries.loc[:, target_names] = pd.concat(all_match_vals, axis=1).T.values


def _impute_lookup(
    row: pd.Series,
    lookup: pd.DataFrame,
    targets: List[NumericalTarget],
    mode: Literal["error", "worst", "best", "mean", "random"] = "error",
) -> pd.Series:
    """Perform data imputation for missing lookup values.

    Args:
        row: The parameter configuration for which the target values should be imputed.
        lookup: The lookup dataframe.
        targets: The targets, providing the required mode information.
        mode: The used impute mode. See :func:`baybe.simulation.lookup.look_up_targets`
            for details.

    Returns:
        The imputed values for the targets.

    Raises:
        IndexError: If the mode ``"error"`` is chosen and at least one of the targets
            could not be found.
    """
    # TODO: this function needs another code cleanup and refactoring

    target_names = [t.name for t in targets]

    if mode == "mean":
        return lookup[target_names].mean(axis=0)

    elif mode == "worst":
        worst_vals = []
        for target in targets:
            if target.mode is TargetMode.MAX:
                worst_vals.append(lookup[target.name].min().flatten()[0])
            elif target.mode is TargetMode.MIN:
                worst_vals.append(lookup[target.name].max().flatten()[0])
            if target.mode is TargetMode.MATCH:
                worst_vals.append(
                    lookup.loc[
                        lookup.loc[
                            (lookup[target.name] - target.bounds.center).abs().idxmax(),
                        ],
                        target.name,
                    ].flatten()[0]
                )
        return pd.Series(worst_vals, index=target_names)

    elif mode == "best":
        best_vals = []
        for target in targets:
            if target.mode is TargetMode.MAX:
                best_vals.append(lookup[target.name].max().flatten()[0])
            elif target.mode is TargetMode.MIN:
                best_vals.append(lookup[target.name].min().flatten()[0])
            if target.mode is TargetMode.MATCH:
                best_vals.append(
                    lookup.loc[
                        lookup.loc[
                            (lookup[target.name] - target.bounds.center).abs().idxmin(),
                        ],
                        target.name,
                    ].flatten()[0]
                )
        return pd.Series(best_vals, index=target_names)

    elif mode == "random":
        vals = []
        randindex = np.random.choice(lookup.index)
        for target in targets:
            vals.append(lookup.loc[randindex, target.name].flatten()[0])
        return pd.Series(vals, index=target_names)

    raise IndexError(
        f"Cannot match the recommended row {row} to any of the rows in the lookup."
    )
