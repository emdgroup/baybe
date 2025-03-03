"""Constraint utilities."""

import numpy as np
import pandas as pd

from baybe.parameters.utils import is_inactive
from baybe.searchspace import SubspaceContinuous


def is_cardinality_fulfilled(
    df: pd.DataFrame,
    subspace_continuous: SubspaceContinuous,
    *,
    check_minimum: bool = True,
    check_maximum: bool = True,
) -> bool:
    """Validate cardinality constraints in a dataframe of parameter configurations.

    Args:
        df: The dataframe to be checked.
        subspace_continuous: The subspace spanned by the considered parameters.
        check_minimum: If ``True``, minimum cardinality constraints are validated.
        check_maximum: If ``True``, maximum cardinality constraints are validated.

    Returns:
        ``True`` if all cardinality constraints are fulfilled, ``False`` otherwise.
    """
    for c in subspace_continuous.constraints_cardinality:
        # Get the activity thresholds for all parameters
        cols = df[c.parameters]
        thresholds = {
            p.name: c.get_absolute_thresholds(p.bounds)
            for p in subspace_continuous.get_parameters_by_name(c.parameters)
        }
        lower_thresholds = [thresholds[p].lower for p in cols.columns]
        upper_thresholds = [thresholds[p].upper for p in cols.columns]

        # Count the number of active values per dataframe row
        inactives = is_inactive(cols, lower_thresholds, upper_thresholds)
        n_zeros = inactives.sum(axis=1)
        n_active = len(c.parameters) - n_zeros

        # Check if cardinality is violated
        if check_minimum and np.any(n_active < c.min_cardinality):
            return False
        if check_maximum and np.any(n_active > c.max_cardinality):
            return False

    return True
