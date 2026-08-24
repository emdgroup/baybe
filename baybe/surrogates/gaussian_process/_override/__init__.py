"""Kernel override resolution for Gaussian process surrogates."""

from baybe.surrogates.gaussian_process._override.core import (
    raise_incompatible_override,
    reduce_kernel_spec,
)
from baybe.surrogates.gaussian_process._override.parameter import (
    extract_parameter_overrides,
)
from baybe.surrogates.gaussian_process._override.tl import (
    extract_transfer_learning_overrides,
)

__all__ = [
    "extract_parameter_overrides",
    "extract_transfer_learning_overrides",
    "raise_incompatible_override",
    "reduce_kernel_spec",
]
