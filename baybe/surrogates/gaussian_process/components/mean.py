"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactory,
    PlainGPComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.means import Mean as GPyTorchMean

    MeanFactory = GPComponentFactory[GPyTorchMean]
    PlainMeanFactory = PlainGPComponentFactory[GPyTorchMean]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    MeanFactory = GPComponentFactory[Any]
    PlainMeanFactory = PlainGPComponentFactory[Any]
