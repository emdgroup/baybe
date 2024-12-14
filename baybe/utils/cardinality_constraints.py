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
        ValueError: If the dimensionality of parameters does not match that of points.

    Returns:
        The counts of near-zero values in the recommendations.


    """
    if len(parameters) != points.shape[1]:
        raise ValueError(
            "Dimensionality mismatch: number of parameters = {len("
            "parameters)}, parameters in recommendations "
            "= {points.shape[1]}."
        )

    # Boolean values indicating whether candidate is near-zero: True for near-zero.
    p_thresholds = np.array([p.near_zero_threshold for p in parameters])
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

        # TODO: Is the parameters in constraints sorted or not? Can we assume the
        #  order of parameters in constraints align with that in the subspace?

        # Counts the near-zero elements
        batch_related_to_c = batch[c.parameters]
        parameters_related_to_c = tuple(
            p for p in subspace_continuous.parameters if p.name in c.parameters
        )
        n_near_zeros = count_near_zeros(parameters_related_to_c, batch_related_to_c)

        # When the minimum cardinality is violated
        if np.any(len(c.parameters) - n_near_zeros < c.min_cardinality):
            return False
    return True
