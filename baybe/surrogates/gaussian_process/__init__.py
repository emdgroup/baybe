"""Gaussian process surrogates."""

from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.surrogates.gaussian_process.multi_fidelity import (
    GaussianProcessSurrogateSTMF,
)

__all__ = [
    "GaussianProcessSurrogate",
    "GaussianProcessSurrogateSTMF",
]
