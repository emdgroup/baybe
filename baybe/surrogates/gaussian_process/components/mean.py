"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

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


class PriorMean:
    """GPyTorch mean module using a trained GP's posterior as the mean function.

    Wraps a trained Gaussian Process and uses its posterior mean predictions
    as the mean function for another GP.
    The provided GP is deep-copied and its parameters are frozen.

    Args:
        gp: Trained botorch SingleTaskGP to use as the mean function.
        batch_shape: Batch shape for the mean module.
    """

    def __init__(
        self,
        gp: SingleTaskGP,
        batch_shape: Any = None,
    ) -> None:
        import gpytorch
        import torch

        gpytorch.means.Mean.__init__(self)

        self.gp: SingleTaskGP = deepcopy(gp)
        self.batch_shape: torch.Size = (
            batch_shape if batch_shape is not None else torch.Size()
        )

        for param in self.gp.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """Compute the mean function using the wrapped GP's posterior."""
        import gpytorch
        import torch

        self.gp.eval()
        self.gp.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean = self.gp(x).mean.detach()

        target_shape = torch.broadcast_shapes(self.batch_shape, x.shape[:-1])
        return mean.reshape(target_shape)


@define
class PriorMeanFactory(MeanFactoryProtocol):
    """A factory that creates a :class:`PriorMean` from a trained botorch GP model.

    The provided model is used as-is (no additional copy is made).
    """

    _prior_model: SingleTaskGP = field()

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        return PriorMean(self._prior_model)
