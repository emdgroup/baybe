"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baybe.surrogates.gaussian_process.components.generic import (
    ComponentFactoryProtocol,
    PlainComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.means import Mean as GPyTorchMean

    MeanFactoryProtocol = ComponentFactoryProtocol[GPyTorchMean]
    PlainMeanFactory = PlainComponentFactory[GPyTorchMean]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    MeanFactoryProtocol = ComponentFactoryProtocol[Any]
    PlainMeanFactory = PlainComponentFactory[Any]
