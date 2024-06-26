"""Helpers for relaxation of continuous cardinality constraints."""


import torch
from torch import Tensor

SMALL_CONST = 1e-3

# Define operators that are eligible for cardinality constraint.
_valid_cardinality_threshold_operators = [">=", "<="]


def count_nonzeros(x: Tensor, ell: float = SMALL_CONST) -> Tensor:
    """Count the number of non-zeros using an approximation.

    One key step is checking whether a parameter is zero or not. We approximate
    ``delta(x) = {1 if x=0; 0 else}`` use a narrow Gaussian function.

    Args:
        x: continuous parameters.
        ell: a small number controlling the approximation. The smaller it is, the closer
            narrow_gaussian() is to delta().

    Returns:
        An approximation of counts of non-zeros.
    """
    n_params = x.shape[-1]

    # Check whether x is close to zero. Use a narrow gaussian function to approximate
    # delta(x) = {1 if x=0; 0 else}
    is_zeros = torch.exp(-0.5 * (x / ell) ** 2)
    return n_params - is_zeros.sum(dim=-1)


def cardinality_relaxed(cardinality_threshold: int, operator: str, x: Tensor) -> Tensor:
    """Relax a maximum/minimum cardinality to a nonlinear inequality constraint.

    The number of non-zero elements in x can be relaxed to non_zeros = n_total -
    narrow_gaussian(x). Due to the maximum cardinality constraint, we have non_zeros
    <= max_cardinality, which can be converted to the nonlinear inequality constraint in
    botorch.optim.optimize.optimize_acqf.
    https://botorch.org/api/_modules/botorch/optim/optimize.html#optimize_acqf

    Args:
        cardinality_threshold: The minimum or maximum cardinality threshold.
        operator: ">=" for minimum cardinality and "<=" for maximum cardinality.
        x: constraint parameters

    Returns:
        Nonlinear inequality constraint evaluated at x. If the relaxed cardinality
        constraint is fulfilled, it has value >= 0; otherwise, its value < 0.

    Raises:
        ValueError: If the operator is not eligible.
        ValueError: If the cardinality threshold exceed the number of parameters.
    """
    if operator not in _valid_cardinality_threshold_operators:
        raise ValueError(
            f"The cardinality threshold operator ({operator}) must be "
            f"within {_valid_cardinality_threshold_operators}."
        )

    if cardinality_threshold > x.shape[-1]:
        raise ValueError(
            f"The cardinality threshold ({cardinality_threshold}) cannot be larger "
            f"than the number of parameters (={x.shape[-1]}). Check your "
            f"min_cardinality and ensure the dimensionality of x."
        )

    if operator == ">=":
        # Apply the minimum cardinality condition
        return count_nonzeros(x) - cardinality_threshold
    elif operator == "<=":
        # Apply the maximum cardinality condition
        return cardinality_threshold - count_nonzeros(x)
    else:
        raise RuntimeError(
            f"The cardinality threshold operator ({operator}) must be "
            f"within {_valid_cardinality_threshold_operators}."
        )
