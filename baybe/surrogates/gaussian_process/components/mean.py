"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from attrs import define, field
from typing_extensions import override

from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    PlainGPComponentFactory,
)

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor

    MeanFactoryProtocol = GPComponentFactoryProtocol[GPyTorchMean]
    PlainMeanFactory = PlainGPComponentFactory[GPyTorchMean]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    MeanFactoryProtocol = GPComponentFactoryProtocol[Any]
    PlainMeanFactory = PlainGPComponentFactory[Any]


@define
class LazyConstantMeanFactory(MeanFactoryProtocol):
    """A factory providing constant mean functions using lazy loading."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        return ConstantMean()


@define
class PriorMeanFactory(MeanFactoryProtocol):
    """A factory that creates a prior mean from a trained botorch GP model.

    The mean function uses the trained GP's posterior mean predictions.
    The provided model is deep-copied and its parameters are frozen.
    """

    _prior_model: SingleTaskGP = field()

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        import gpytorch
        import torch

        class PriorMean(gpytorch.means.Mean):
            """GPyTorch mean using a trained GP's posterior as the mean function."""

            def __init__(self, gp: SingleTaskGP) -> None:
                super().__init__()
                self.gp: SingleTaskGP = deepcopy(gp)
                for param in self.gp.parameters():
                    param.requires_grad = False

            def forward(self, x: Tensor) -> Tensor:
                """Compute the mean using the wrapped GP's posterior."""
                self.gp.eval()
                self.gp.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    return self.gp(x).mean.detach()

        return PriorMean(self._prior_model)


gc.collect()  # Collect leftover original slotted classes created by attrs
