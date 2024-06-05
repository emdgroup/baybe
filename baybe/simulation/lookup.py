"""Target lookup mechanisms."""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection
from functools import singledispatch
from typing import Literal

import numpy as np
import pandas as pd

from baybe.simulation._imputation import _impute_lookup
from baybe.targets.base import Target
from baybe.utils.dataframe import add_fake_results

_logger = logging.getLogger(__name__)


@singledispatch
def look_up_targets(
    lookup: pd.DataFrame | Callable | None,
    queries: pd.DataFrame,
    targets: Collection[Target],
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
) -> None:
    """Add/fill target values in a dataframe using a lookup mechanism.

    Note:
        This does not create a new dataframe but modifies ``queries`` in-place.

    Args:
        lookup: The lookup mechanism. Can be one of the following choices:

            -   A dataframe mapping rows of ``queries`` to the corresponding target
                values. That is, it must contain the same columns as ``queries`` plus
                one additional column for each of the given target.
            -   A callable, providing target values for each row of ``queries``.
            -   ``None``. Produces fake values for all targets.
        queries: The dataframe to be modified. Its content must be compatible with the
            chosen lookup mechanism.
        targets: The targets whose values are to be looked up.
        impute_mode: Specifies how a missing lookup will be handled. Only relevant for
            dataframe lookups. Can be one of the following choices:

            - ``"error"``: An error will be thrown.
            - ``"worst"``: Imputes the worst available value for each target.
            - ``"best"``: Imputes the best available value for each target.
            - ``"mean"``: Imputes the mean value for each target.
            - ``"random"``: A random row will be used for the lookup.

    Example:
        >>> import pandas as pd
        >>> from baybe.targets.numerical import NumericalTarget
        >>> from baybe.simulation.lookup import look_up_targets
        >>>
        >>> targets = [NumericalTarget("target", "MAX")]
        >>> df = pd.DataFrame({"x": [1, 2, 3]})
        >>> lookup_df = pd.DataFrame({"x": [1, 2], "target": [10, 20]})
        >>> look_up_targets(lookup_df, df, targets, impute_mode="mean")
        >>> print(df)
           x  target
        0  1    10.0
        1  2    20.0
        2  3    15.0
    """


@look_up_targets.register
def _look_up_fake_targets(
    _: None, queries: pd.DataFrame, targets: Collection[Target]
) -> None:
    """Look up fake target values."""
    add_fake_results(queries, targets)


@look_up_targets.register
def _look_up_targets_from_callable(
    lookup: Callable,
    queries: pd.DataFrame,
    targets: Collection[Target],
) -> None:
    """Look up target values by querying a callable."""
    # TODO: Currently, the alignment of return values to targets is based on the
    #   column ordering, which is not robust. Instead, the callable should return
    #   a dataframe with properly labeled columns.

    # Since the return of a lookup function is a tuple, the following code stores
    # tuples of floats in a single column with label 0:
    measured_targets = queries.apply(lambda x: lookup(*x.values), axis=1).to_frame()
    # We transform this column to a DataFrame in which there is an individual
    # column for each of the targets....
    split_target_columns = pd.DataFrame(
        measured_targets[0].to_list(), index=measured_targets.index
    )
    # ... and assign this to measured_targets in order to have one column per target
    measured_targets[split_target_columns.columns] = split_target_columns
    if measured_targets.shape[1] != len(targets):
        raise AssertionError(
            "If you use an analytical function as lookup, make sure "
            "the configuration has the right amount of targets "
            "specified."
        )
    for k_target, target in enumerate(targets):
        queries[target.name] = measured_targets.iloc[:, k_target]


@look_up_targets.register
def _look_up_targets_from_dataframe(
    lookup: pd.DataFrame,
    queries: pd.DataFrame,
    targets: Collection[Target],
    impute_mode: Literal[
        "error", "worst", "best", "mean", "random", "ignore"
    ] = "error",
) -> None:
    """Look up target values from a dataframe."""
    # # IMPROVE: Although it's not too important for a simulation, this
    # #  could also be implemented for approximate matches"""

    target_names = [t.name for t in targets]

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
            match_vals = _impute_lookup(row, lookup, targets, impute_mode)

        else:
            # Exactly one match has been found
            match_vals = lookup.loc[ind[0], target_names].values

        # Collect the matches
        all_match_vals.append(match_vals)

    # Add the lookup values
    queries.loc[:, target_names] = np.asarray(all_match_vals)
