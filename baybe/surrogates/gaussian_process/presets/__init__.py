"""Gaussian process surrogate presets."""

# Criterion
from baybe.surrogates.gaussian_process.components.fit_criterion import FitCriterion

# Default preset
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBEFitCriterionFactory,
    BayBEKernelFactory,
    BayBELikelihoodFactory,
    BayBEMeanFactory,
)

# Chen preset
from baybe.surrogates.gaussian_process.presets.chen import (
    CHENFitCriterionFactory,
    CHENKernelFactory,
)

# Core
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset

# EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo import (
    EDBOFitCriterionFactory,
    EDBOKernelFactory,
    EDBOLikelihoodFactory,
    EDBOMeanFactory,
)

# Smoothed EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOFitCriterionFactory,
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
    SmoothedEDBOMeanFactory,
)

__all__ = [
    # Core
    "FitCriterion",
    "GaussianProcessPreset",
    # Default BayBE preset
    "BayBEFitCriterionFactory",
    "BayBEKernelFactory",
    "BayBELikelihoodFactory",
    "BayBEMeanFactory",
    # Chen preset
    "CHENFitCriterionFactory",
    "CHENKernelFactory",
    # EDBO preset
    "EDBOFitCriterionFactory",
    "EDBOKernelFactory",
    "EDBOLikelihoodFactory",
    "EDBOMeanFactory",
    # Smoothed EDBO preset
    "SmoothedEDBOFitCriterionFactory",
    "SmoothedEDBOKernelFactory",
    "SmoothedEDBOLikelihoodFactory",
    "SmoothedEDBOMeanFactory",
]
