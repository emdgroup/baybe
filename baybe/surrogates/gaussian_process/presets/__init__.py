"""Gaussian process surrogate presets."""

from baybe.surrogates.gaussian_process.presets.core import (
    GaussianProcessPreset,
    make_gp_from_preset,
)

__all__ = [
    "make_gp_from_preset",
    "GaussianProcessPreset",
]
