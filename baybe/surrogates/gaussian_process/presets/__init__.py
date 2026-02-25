"""Gaussian process surrogate presets."""

# Core
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset

# Default preset
from baybe.surrogates.gaussian_process.presets.default import (
    DefaultKernelFactory,
    DefaultLikelihoodFactory,
    DefaultMeanFactory,
)

# EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo import (
    EDBOKernelFactory,
    EDBOLikelihoodFactory,
    EDBOMeanFactory,
)

# Smoothed EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
    SmoothedEDBOMeanFactory,
)

__all__ = [
    # Core
    "GaussianProcessPreset",
    # Default preset
    "DefaultKernelFactory",
    "DefaultLikelihoodFactory",
    "DefaultMeanFactory",
    # EDBO preset
    "EDBOKernelFactory",
    "EDBOLikelihoodFactory",
    "EDBOMeanFactory",
    # Smoothed EDBO preset
    "SmoothedEDBOKernelFactory",
    "SmoothedEDBOLikelihoodFactory",
    "SmoothedEDBOMeanFactory",
]
