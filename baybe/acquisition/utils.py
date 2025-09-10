"""Utilities for acquisition functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from baybe.acquisition.base import AcquisitionFunction

if TYPE_CHECKING:
    from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
        BoxDecomposition,
    )
    from torch import Tensor


def str_to_acqf(name: str, /) -> AcquisitionFunction:
    """Create an ACQF object from a given ACQF name."""
    return AcquisitionFunction.from_dict({"type": name})


def convert_acqf(acqf: AcquisitionFunction | str, /) -> AcquisitionFunction:
    """Convert an ACQF name into an ACQF object (with ACQF object passthrough)."""
    return acqf if isinstance(acqf, AcquisitionFunction) else str_to_acqf(acqf)


def make_partitioning(
    predictions: Tensor, ref_point: Tensor, alpha: float | None
) -> BoxDecomposition:
    """Create a :class:`~botorch.utils.multi_objective.box_decompositions.box_decomposition.BoxDecomposition` object for the given predictions and reference point.

    For details on the arguments, see
    :class:`~botorch.utils.multi_objective.box_decompositions.non_dominated.NondominatedPartitioning`.

    Args:
        predictions: The predictions tensor of shape (n_samples, n_outputs).
        ref_point: The reference point tensor of shape (n_outputs,).
        alpha: Optional threshold parameter controlling the partitioning generation.
            Hypercells with a volume fraction (relative to the total Pareto set
            hypervolume) less than the specified value will be dropped, leading to more
            approximation but faster computation.

    Raises:
        ValueError: If the predictions or reference point do not have the
            expected shapes.

    Returns:
        A partitioning object for hypervolume acquisition functions.
    """  # noqa: E501
    from botorch.acquisition.input_constructors import (
        get_default_partitioning_alpha,
    )
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
        NondominatedPartitioning,
    )

    if predictions.ndim != 2:
        raise ValueError(
            f"Predictions must be a 2-D tensor, got shape {predictions.shape}."
        )

    if ref_point.ndim != 1:
        raise ValueError(
            f"Reference point must be a 1-D tensor, got shape {ref_point.shape}."
        )

    if (n_p := predictions.shape[1]) != (n_r := len(ref_point)):
        raise ValueError(
            f"Predictions dimensionality {n_p} does not match reference point "
            f"dimensionality {n_r}."
        )

    alpha = (
        get_default_partitioning_alpha(num_objectives=len(ref_point))
        if alpha is None
        else alpha
    )

    # alpha=0 means requesting an exact partitioning, for which there is a specialized
    # faster algorithm available
    if alpha == 0:
        return FastNondominatedPartitioning(ref_point=ref_point, Y=predictions)

    return NondominatedPartitioning(ref_point=ref_point, Y=predictions, alpha=alpha)
