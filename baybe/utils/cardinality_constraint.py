"""Helpers for relaxation of continuous cardinality constraints."""


import torch
from torch import Tensor

SMALL_CONST = 1e-3


def count_nonzeros(x: Tensor, ell: float = SMALL_CONST) -> Tensor:
    """Count the number of non-zeros using an approximation.

    To count the number of zeros in x, we approximate
    ``delta(x) = {1 if x=0; 0 else}`` use a narrow Gaussian function.

    Args:
        x: continuous parameters.
        ell: a small number controlling the approximation. The smaller it is, the closer
            the narrow Gaussian function is to delta().

    Returns:
        An approximation of counts of non-zeros.
    """
    n_params = x.shape[-1]

    # Check whether x is close to zero using a narrow gaussian function.
    is_zeros = torch.exp(-0.5 * (x / ell) ** 2)
    return n_params - is_zeros.sum(dim=-1)


def max_cardinality_relaxed(cardinality_threshold: int, x: Tensor) -> Tensor:
    """Evaluate a nonlinear inequality constraint as a relaxation of the maximum
    cardinality constraint.

    The nonlinear inequality constraint has the form of nonlinear_ineq(x) >= 0.

    Args:
        cardinality_threshold: The cardinality threshold.
        x: Constraint parameters.

    Returns:
        Nonlinear inequality constraint evaluated at x. It has value >= 0, if the
        relaxed cardinality constraint is fulfilled; <0, otherwise.

    Raises:
        ValueError: If the cardinality threshold exceeds the number of parameters.
    """  # noqa D202

    if cardinality_threshold > x.shape[-1]:
        raise ValueError(
            f"The cardinality threshold ({cardinality_threshold}) cannot be larger "
            f"than the number of parameters (={x.shape[-1]}). Check your "
            f"cardinality threshold and ensure the dimensionality of x."
        )

    # Cardinality_threshold - count_nonzeros(x) >= 0
    return cardinality_threshold - count_nonzeros(x)


def min_cardinality_relaxed(cardinality_threshold: int, x: Tensor) -> Tensor:
    """Evaluate a nonlinear inequality constraint as a relaxation of the minimum
    cardinality constraint."""  # noqa D202
    # Count_nonzeros(x) - cardinality_threshold >= 0
    return -max_cardinality_relaxed(cardinality_threshold, x)
