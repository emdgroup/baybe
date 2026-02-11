"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from baybe.surrogates.gaussian_process.components.generic import (
    ComponentFactory,
    PlainComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.means import Mean as GPyTorchMean

    MeanFactory = ComponentFactory[GPyTorchMean]
    PlainMeanFactory = PlainComponentFactory[GPyTorchMean]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    MeanFactory = ComponentFactory[Any]
    PlainMeanFactory = PlainComponentFactory[Any]
