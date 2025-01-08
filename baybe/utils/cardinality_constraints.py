"""Utilities related to cardinality constraints."""

from typing import Literal

import numpy as np
import pandas as pd

from baybe.parameters.utils import SMALLEST_FLOAT32
from baybe.searchspace import SubspaceContinuous
from baybe.utils.interval import Interval


def count_near_zeros(
    thresholds: tuple[Interval, ...], points: pd.DataFrame
) -> np.ndarray:
    """Return the counts of near-zeros in the recommendations.

    Args:
        thresholds: A list of thresholds for according to which the counts of
            near-zeros in the recommendations should be calculated.
        points: The recommendations of the parameter objects.

    Raises:
        ValueError: If number of thresholds differs from the number of
            parameters in points.

    Returns:
        The counts of near-zero values in the recommendations.


    """
    if len(thresholds) != len(points.columns):
        raise ValueError(
            f"The size of thresholds ({len(thresholds)}) must be the same as the "
            f"number of parameters ({len(points.columns)}) in points."
        )
    # Get the lower threshold for determining zeros/non-zeros. When the
    # lower_threshold is zero, we replace it with a very small negative value to have
    # the threshold being an open-support.
    lower_threshold = np.array(
        [min(threshold.lower, -SMALLEST_FLOAT32) for threshold in thresholds]
    )
    lower_threshold = np.broadcast_to(lower_threshold, points.shape)

    # Get the upper threshold for determining zeros/non-zeros. When the
    # upper_threshold is zero, we replace it with a very small positive value.
    upper_threshold = np.array(
        [max(threshold.upper, SMALLEST_FLOAT32) for threshold in thresholds]
    )
    upper_threshold = np.broadcast_to(upper_threshold, points.shape)

    # Boolean values indicating whether the candidates is near-zero: True for is
    # near-zero.
    near_zero_flags = (points > lower_threshold) & (points < upper_threshold)
    return np.sum(near_zero_flags, axis=1)


def is_cardinality_fulfilled(
    subspace_continuous: SubspaceContinuous,
    batch: pd.DataFrame,
    type_cardinality: Literal["min", "max"],
) -> bool:
    """Check whether all minimum cardinality constraints are fulfilled.

    Args:
        subspace_continuous:
            The continuous subspace from which candidates are generated.
        batch: The recommended batch
        type_cardinality:
            "min" or "max". "min" indicates all minimum cardinality constraints are
            checked; "max" for all maximum cardinality constraints.

    Returns:
        Return "True" if all minimum cardinality constraints are fulfilled; "False"
        otherwise.

    Raises:
        ValueError: If type_cardinality is neither "min" nor "max".
    """
    if type_cardinality not in ["min", "max"]:
        raise ValueError(
            f"Unknown type of cardinality. Only support min or max but "
            f"{type_cardinality=}."
        )

    if len(subspace_continuous.constraints_cardinality) == 0:
        return True

    for c in subspace_continuous.constraints_cardinality:
        # No need to check this redundant cardinality constraint
        if (c.min_cardinality == 0) and type_cardinality == "min":
            continue

        if (c.max_cardinality == len(c.parameters)) and type_cardinality == "max":
            continue

        # Batch of parameters that are related to cardinality constraint
        batch_related_to_c = batch[c.parameters]

        # Parameters related to cardinality constraint
        parameters_in_c = subspace_continuous.get_parameters_by_name(c.parameters)

        # Thresholds of parameters that are related to the cardinality constraint
        thresholds = tuple(c.get_threshold(p) for p in parameters_in_c)

        # Count the number of near-zero elements
        n_near_zeros = count_near_zeros(thresholds, batch_related_to_c)

        # When any minimum cardinality is violated
        if type_cardinality == "min" and np.any(
            len(c.parameters) - n_near_zeros < c.min_cardinality
        ):
            return False

        # When any maximum cardinality is violated
        if type_cardinality == "max" and np.any(
            len(c.parameters) - n_near_zeros > c.max_cardinality
        ):
            return False
    return True
