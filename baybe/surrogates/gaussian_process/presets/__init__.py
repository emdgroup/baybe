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

# BoTorch preset
from baybe.surrogates.gaussian_process.presets.botorch import (
    BotorchKernelFactory,
    BotorchLikelihoodFactory,
    BotorchMeanFactory,
)

# Chen preset
from baybe.surrogates.gaussian_process.presets.chen import (
    CHEN_FIT_CRITERION_FACTORY,
    CHENKernelFactory,
)

# Core
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset

# EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo import (
    EDBO_FIT_CRITERION_FACTORY,
    EDBOKernelFactory,
    EDBOLikelihoodFactory,
    EDBOMeanFactory,
)

# Smoothed EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SMOOTHED_EDBO_FIT_CRITERION_FACTORY,
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
    # BoTorch preset
    "BotorchKernelFactory",
    "BotorchLikelihoodFactory",
    "BotorchMeanFactory",
    # Chen preset
    "CHEN_FIT_CRITERION_FACTORY",
    "CHENKernelFactory",
    # EDBO preset
    "EDBO_FIT_CRITERION_FACTORY",
    "EDBOKernelFactory",
    "EDBOLikelihoodFactory",
    "EDBOMeanFactory",
    # Smoothed EDBO preset
    "SMOOTHED_EDBO_FIT_CRITERION_FACTORY",
    "SmoothedEDBOKernelFactory",
    "SmoothedEDBOLikelihoodFactory",
    "SmoothedEDBOMeanFactory",
]
