"""Gaussian process surrogate components."""

from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterion,
    FitCriterionFactoryProtocol,
    PlainFitCriterionFactory,
)
from baybe.surrogates.gaussian_process.components.kernel import (
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
    PriorMeanFactory,
)

__all__ = [
    # Fit Criterion
    "FitCriterion",
    "FitCriterionFactoryProtocol",
    "PlainFitCriterionFactory",
    # Kernel
    "KernelFactoryProtocol",
    "PlainKernelFactory",
    # Likelihood
    "LikelihoodFactoryProtocol",
    "PlainLikelihoodFactory",
    # Mean
    "LazyConstantMeanFactory",
    "MeanFactoryProtocol",
    "PlainMeanFactory",
    "PriorMeanFactory",
]
