"""Likelihood factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baybe.surrogates.gaussian_process.components import (
    ComponentFactory,
    PlainComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood

    LikelihoodFactory = ComponentFactory[GPyTorchLikelihood]
    PlainLikelihoodFactory = PlainComponentFactory[GPyTorchLikelihood]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    LikelihoodFactory = ComponentFactory[Any]
    PlainLikelihoodFactory = PlainComponentFactory[Any]
