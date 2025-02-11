"""Gaussian process surrogate presets."""

from baybe.surrogates.gaussian_process.presets.core import (
    GaussianProcessPreset,
    make_gp_from_preset,
)
from baybe.surrogates.gaussian_process.presets.default import DefaultKernelFactory
from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory
from baybe.surrogates.gaussian_process.presets.botorch import BotorchKernelFactory

__all__ = [
    "DefaultKernelFactory",
    "EDBOKernelFactory",
    "BotorchKernelFactory",
    "make_gp_from_preset",
    "GaussianProcessPreset",
]
