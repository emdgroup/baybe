"""Gaussian process surrogate presets."""

from baybe.surrogates.gaussian_process.presets.core import (
    GaussianProcessPreset,
    make_gp_from_preset,
)
from baybe.surrogates.gaussian_process.presets.default import DefaultKernelFactory
from baybe.surrogates.gaussian_process.presets.edbo import EDBOKernelFactory
from baybe.surrogates.gaussian_process.presets.fidelity import (
    DefaultFidelityKernelFactory,
    IndependentFidelityKernelFactory,
    IndexFidelityKernelFactory,
)

__all__ = [
    "DefaultKernelFactory",
    "DefaultFidelityKernelFactory",
    "EDBOKernelFactory",
    "GaussianProcessPreset",
    "IndependentFidelityKernelFactory",
    "IndexFidelityKernelFactory",
    "make_gp_from_preset",
]
