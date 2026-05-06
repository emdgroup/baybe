"""Gaussian process surrogate components."""

from baybe.surrogates.gaussian_process.components.criterion import (
    Criterion,
    CriterionFactoryProtocol,
    PlainCriterionFactory,
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
)

__all__ = [
    # Criterion
    "Criterion",
    "CriterionFactoryProtocol",
    "PlainCriterionFactory",
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
]
