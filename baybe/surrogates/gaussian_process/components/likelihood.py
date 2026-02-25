"""Likelihood factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baybe.surrogates.gaussian_process.components.generic import (
    ComponentFactory,
    PlainComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood

    LikelihoodFactory = ComponentFactory[GPyTorchLikelihood]
    PlainLikelihoodFactory = PlainComponentFactory[GPyTorchLikelihood]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    LikelihoodFactory = ComponentFactory[Any]
    PlainLikelihoodFactory = PlainComponentFactory[Any]
