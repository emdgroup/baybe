"""Utilities related to cardinality constraints."""

from typing import Literal

import numpy as np
import pandas as pd

from baybe.searchspace import SubspaceContinuous
from baybe.utils.interval import Interval


def count_zeros(thresholds: tuple[Interval, ...], points: pd.DataFrame) -> np.ndarray:
    """Return the counts of zeros in the recommendations.

    Args:
        thresholds: A list of thresholds according to which the counts of zeros
            in the recommendations should be calculated.
        points: The recommendations of the parameter objects.

    Returns:
        The counts of zero parameters in the recommendations.

    Raises:
        ValueError: If the number of thresholds differs from the number of
            parameters in points.
    """
    if len(thresholds) != len(points.columns):
        raise ValueError(
            f"The size of thresholds ({len(thresholds)}) must be the same as the "
            f"number of parameters ({len(points.columns)}) in points."
        )
    # Get the lower/upper thresholds for determining zeros/non-zeros
    lower_thresholds = np.array([threshold.lower for threshold in thresholds])
    lower_thresholds = np.broadcast_to(lower_thresholds, points.shape)

    upper_thresholds = np.array([threshold.upper for threshold in thresholds])
    upper_thresholds = np.broadcast_to(upper_thresholds, points.shape)

    # Boolean values indicating whether the candidates are treated zeros: True for zero
    zero_flags = (points > lower_thresholds) & (points < upper_thresholds)

    # Correct the comparison on the special boundary: zero. This step is needed
    # because when the lower_threshold = 0, a value v with lower_threshold <= v <
    # upper_threshold should be treated zero.
    zero_flags = (points == 0.0) | zero_flags

    return np.sum(zero_flags, axis=1)


def is_cardinality_fulfilled(
    subspace_continuous: SubspaceContinuous,
    batch: pd.DataFrame,
    type_cardinality: Literal["min", "max"],
) -> bool:
    """Check whether all minimum (or maximum) cardinality constraints are fulfilled.

    Args:
        subspace_continuous: The continuous subspace from which candidates are
            generated.
        batch: The recommended batch
        type_cardinality: "min" or "max". "min" indicates all minimum cardinality
            constraints will be checked; "max" for all maximum cardinality constraints.

    Returns:
        Return "True" if all minimum (or maximum) cardinality constraints are
        fulfilled; "False" otherwise.

    Raises:
        ValueError: If type_cardinality is neither "min" nor "max".
    """
    if type_cardinality not in ["min", "max"]:
        raise ValueError(
            f"Unknown type of cardinality. Only support min or max but "
            f"{type_cardinality=} is given."
        )

    if len(subspace_continuous.constraints_cardinality) == 0:
        return True

    for c in subspace_continuous.constraints_cardinality:
        # No need to check the redundant cardinality constraints that are
        # - min_cardinality = 0
        # - max_cardinality = len(parameters)
        if (c.min_cardinality == 0) and type_cardinality == "min":
            continue

        if (c.max_cardinality == len(c.parameters)) and type_cardinality == "max":
            continue

        # Batch of parameters that are related to cardinality constraint
        batch_related_to_c = batch[c.parameters]

        # Parameters related to cardinality constraint
        parameters_in_c = subspace_continuous.get_parameters_by_name(c.parameters)

        # Thresholds of parameters that are related to the cardinality constraint
        thresholds = tuple(c.get_absolute_thresholds(p.bounds) for p in parameters_in_c)

        # Count the number of zeros
        n_zeros = count_zeros(thresholds, batch_related_to_c)

        # When any minimum cardinality is violated
        if type_cardinality == "min" and np.any(
            len(c.parameters) - n_zeros < c.min_cardinality
        ):
            return False

        # When any maximum cardinality is violated
        if type_cardinality == "max" and np.any(
            len(c.parameters) - n_zeros > c.max_cardinality
        ):
            return False
    return True
