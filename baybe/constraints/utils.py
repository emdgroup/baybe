"""Constraint utilities."""

import numpy as np
import pandas as pd

from baybe.searchspace import SubspaceContinuous
from baybe.utils.dataframe import is_between


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
    if len(subspace_continuous.constraints_cardinality) == 0:
        return True

    for c in subspace_continuous.constraints_cardinality:
        # Get the activity thresholds for all parameters
        thresholds = {
            p.name: c.get_absolute_thresholds(p.bounds)
            for p in subspace_continuous.get_parameters_by_name(c.parameters)
        }

        # Count the number of active values per dataframe row
        n_zeros = is_between(df[c.parameters], thresholds).sum(axis=1)
        n_active = len(c.parameters) - n_zeros

        # Check if cardinality is violated
        if check_minimum and np.any(n_active < c.min_cardinality):
            return False
        if check_maximum and np.any(n_active > c.max_cardinality):
            return False

    return True
