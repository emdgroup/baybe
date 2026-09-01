"""Kernels for Gaussian process surrogate models.

The kernel classes mimic classes from GPyTorch. For details on specification and
arguments see https://docs.gpytorch.ai/en/stable/kernels.html.
"""

from baybe.kernels.basic import (
    IndexKernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
)
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel

__all__ = [
    "AdditiveKernel",
    "IndexKernel",
    "LinearKernel",
    "MaternKernel",
    "PeriodicKernel",
    "PiecewisePolynomialKernel",
    "PolynomialKernel",
    "ProductKernel",
    "RBFKernel",
    "RFFKernel",
    "RQKernel",
    "ScaleKernel",
]
