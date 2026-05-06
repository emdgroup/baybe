"""Gaussian process surrogate presets."""

# Criterion
from baybe.surrogates.gaussian_process.components.criterion import Criterion

# Default preset
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBECriterionFactory,
    BayBEKernelFactory,
    BayBELikelihoodFactory,
    BayBEMeanFactory,
)

# Chen preset
from baybe.surrogates.gaussian_process.presets.chen import (
    CHENCriterionFactory,
    CHENKernelFactory,
)

# Core
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset

# EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo import (
    EDBOCriterionFactory,
    EDBOKernelFactory,
    EDBOLikelihoodFactory,
    EDBOMeanFactory,
)

# Smoothed EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOCriterionFactory,
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
    SmoothedEDBOMeanFactory,
)

__all__ = [
    # Core
    "Criterion",
    "GaussianProcessPreset",
    # Default BayBE preset
    "BayBECriterionFactory",
    "BayBEKernelFactory",
    "BayBELikelihoodFactory",
    "BayBEMeanFactory",
    # Chen preset
    "CHENCriterionFactory",
    "CHENKernelFactory",
    # EDBO preset
    "EDBOCriterionFactory",
    "EDBOKernelFactory",
    "EDBOLikelihoodFactory",
    "EDBOMeanFactory",
    # Smoothed EDBO preset
    "SmoothedEDBOCriterionFactory",
    "SmoothedEDBOKernelFactory",
    "SmoothedEDBOLikelihoodFactory",
    "SmoothedEDBOMeanFactory",
]
