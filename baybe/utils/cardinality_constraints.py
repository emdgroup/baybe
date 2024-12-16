"""Utilities related to cardinality constraints."""

import numpy as np
import pandas as pd

from baybe.parameters import NumericalContinuousParameter
from baybe.searchspace import SubspaceContinuous


def count_near_zeros(
    parameters: tuple[NumericalContinuousParameter, ...], points: pd.DataFrame
) -> np.ndarray:
    """Return the counts of near-zeros in the recommendations.

    Args:
        parameters: A list of parameter objects according to which the counts of
            near-zeros in the recommendations should be calculated.
        points: The recommendations of the parameter objects.

    Raises:
        ValueError: If parameters does not cover all parameters present in points.

    Returns:
        The counts of near-zero values in the recommendations.


    """
    p_names = [p.name for p in parameters]
    if not set(points.columns).issubset(set(p_names)):
        raise ValueError(
            "Parameters must cover all parameters present in points: "
            "parameter names in parameters are: {p_name} and parameter "
            "names from points are: {points.columns}."
        )

    # Only keep parameters that are present in points; The order of parameters
    # aligns with that in points.
    parameters_filtered_sorted = (
        p for p_name in points.columns for p in parameters if p.name == p_name
    )

    # Boolean values indicating whether the candidate is near-zero: True for near-zero.
    p_thresholds = np.array([p.near_zero_threshold for p in parameters_filtered_sorted])
    p_thresholds_mask = np.broadcast_to(p_thresholds, points.shape)
    near_zero_flags = (points > -p_thresholds_mask) & (points < p_thresholds_mask)
    return np.sum(near_zero_flags, axis=1)


def is_min_cardinality_fulfilled(
    subspace_continuous: SubspaceContinuous, batch: pd.DataFrame
) -> bool:
    """Check whether any minimum cardinality constraints are fulfilled.

    Args:
        subspace_continuous: The continuous subspace from which candidates are
            generated.
        batch: The recommended batch

    Returns:
        Return "True" if all minimum cardinality constraints are fulfilled; "False"
        otherwise.
    """
    if len(subspace_continuous.constraints_cardinality) == 0:
        return True

    for c in subspace_continuous.constraints_cardinality:
        if c.min_cardinality == 0:
            continue

        # Counts the near-zero elements
        batch_related_to_c = batch[c.parameters]
        n_near_zeros = count_near_zeros(
            subspace_continuous.parameters, batch_related_to_c
        )

        # When the minimum cardinality is violated
        if np.any(len(c.parameters) - n_near_zeros < c.min_cardinality):
            return False
    return True
