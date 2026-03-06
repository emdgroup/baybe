"""Likelihood factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactory,
    PlainGPComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood

    LikelihoodFactory = GPComponentFactory[GPyTorchLikelihood]
    PlainLikelihoodFactory = PlainGPComponentFactory[GPyTorchLikelihood]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    LikelihoodFactory = GPComponentFactory[Any]
    PlainLikelihoodFactory = PlainGPComponentFactory[Any]
