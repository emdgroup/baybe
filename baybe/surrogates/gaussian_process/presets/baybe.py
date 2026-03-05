"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

from baybe.surrogates.gaussian_process.components.mean import LazyConstantMeanFactory
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
)

DefaultKernelFactory = SmoothedEDBOKernelFactory
"""The factory providing the default kernel for Gaussian process surrogates."""

DefaultMeanFactory = LazyConstantMeanFactory
"""The factory providing the default mean function for Gaussian process surrogates."""

DefaultLikelihoodFactory = SmoothedEDBOLikelihoodFactory
"""The factory providing the default likelihood for Gaussian process surrogates."""

# Aliases for generic preset imports
PresetKernelFactory = DefaultKernelFactory
PresetMeanFactory = DefaultMeanFactory
PresetLikelihoodFactory = DefaultLikelihoodFactory
