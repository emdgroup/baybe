"""Helpers for relaxation of continuous cardinality constraints."""


import torch
from torch import Tensor

SMALL_CONST = 1e-3


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


def max_cardinality_relaxed(max_cardinality: int, x: Tensor) -> Tensor:
    """Relax a maximum cardinality to a nonlinear inequality constraint.

    The number of non-zero elements in x can be relaxed to non_zeros = n_total -
    narrow_gaussian(x). Due to the maximum cardinality constraint, we have non_zeros
    <= max_cardinality, which can be converted to the nonlinear inequality constraint in
    botorch.optim.optimize.optimize_acqf.
    https://botorch.org/api/_modules/botorch/optim/optimize.html#optimize_acqf

    Args:
        max_cardinality: The maximum allowed cardinality.
        x: constraint parameters.

    Returns:
        Nonlinear inequality constraint evaluated at x. If the relaxed
        cardinality constraint is fulfilled, it has value >= 0; otherwise, its value
        < 0.
    """
    # TODO: remove duplicate
    # TODO: validate max_cardinality
    # if >= 0, constraint is satisfied
    return max_cardinality - count_nonzeros(x)


def min_cardinality_relaxed(min_cardinality: int, x: Tensor) -> Tensor:
    """Relax a maximum cardinality to a nonlinear inequality constraint.

    See `max_cardinality_relaxed` for more details.
    """
    # TODO: remove duplicate
    # TODO: validate max_cardinality
    # if >= 0, constraint is satisfied
    return count_nonzeros(x) - min_cardinality
