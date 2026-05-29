"""Gaussian process surrogate components."""

from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterion,
    FitCriterionFactoryProtocol,
    PlainFitCriterionFactory,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
    KernelFactoryProtocol,
    PlainKernelFactory,
)
from baybe.surrogates.gaussian_process.components.likelihood import (
    LikelihoodFactoryProtocol,
    PlainLikelihoodFactory,
)
from baybe.surrogates.gaussian_process.components.mean import (
    LazyConstantMeanFactory,
    MeanFactoryProtocol,
    PlainMeanFactory,
)

__all__ = [
    # Fit Criterion
    "FitCriterion",
    "FitCriterionFactoryProtocol",
    "PlainFitCriterionFactory",
    # Kernel
    "ICMKernelFactory",
    "KernelFactoryProtocol",
    "PlainKernelFactory",
    # Likelihood
    "LikelihoodFactoryProtocol",
    "PlainLikelihoodFactory",
    # Mean
    "LazyConstantMeanFactory",
    "MeanFactoryProtocol",
    "PlainMeanFactory",
]
