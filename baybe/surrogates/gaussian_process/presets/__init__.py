"""Gaussian process surrogate presets."""

# Core
from baybe.surrogates.gaussian_process.components.fit_criterion import FitCriterion

# BayBE preset
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
from baybe.surrogates.gaussian_process.presets.chen import ChenKernelFactory
from baybe.surrogates.gaussian_process.presets.core import GaussianProcessPreset

# EDBO preset
from baybe.surrogates.gaussian_process.presets.edbo import (
    EDBOKernelFactory,
    EDBOLikelihoodFactory,
    EDBOMeanFactory,
)

# EDBO Smoothed preset
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
    SmoothedEDBOMeanFactory,
)

# Hvarfner preset
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerKernelFactory,
    HvarfnerLikelihoodFactory,
    HvarfnerMeanFactory,
)

__all__ = [
    # Core
    "FitCriterion",
    "GaussianProcessPreset",
    # BayBE preset
    "BayBEFitCriterionFactory",
    "BayBEKernelFactory",
    "BayBELikelihoodFactory",
    "BayBEMeanFactory",
    # BoTorch preset
    "BotorchKernelFactory",
    "BotorchLikelihoodFactory",
    "BotorchMeanFactory",
    # Chen preset
    "ChenKernelFactory",
    # EDBO preset
    "EDBOKernelFactory",
    "EDBOLikelihoodFactory",
    "EDBOMeanFactory",
    # EDBO Smoothed preset
    "SmoothedEDBOKernelFactory",
    "SmoothedEDBOLikelihoodFactory",
    "SmoothedEDBOMeanFactory",
    # Hvarfner preset
    "HvarfnerKernelFactory",
    "HvarfnerLikelihoodFactory",
    "HvarfnerMeanFactory",
]
