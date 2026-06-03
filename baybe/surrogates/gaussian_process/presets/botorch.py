"""BoTorch preset for Gaussian process surrogates.

Currently mimics the Hvarfner preset :cite:p:`Hvarfner2024`.
"""

from baybe.surrogates.gaussian_process.presets.hvarfner import (
    FIT_CRITERION_FACTORY,
    KERNEL_FACTORY,
    LIKELIHOOD_FACTORY,
    MEAN_FACTORY,
)
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerKernelFactory as BotorchKernelFactory,
)
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerLikelihoodFactory as BotorchLikelihoodFactory,
)
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerMeanFactory as BotorchMeanFactory,
)

__all__ = [
    "BotorchKernelFactory",
    "BotorchLikelihoodFactory",
    "BotorchMeanFactory",
    "FIT_CRITERION_FACTORY",
    "KERNEL_FACTORY",
    "LIKELIHOOD_FACTORY",
    "MEAN_FACTORY",
]
