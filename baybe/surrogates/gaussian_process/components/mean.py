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
        from botorch.models.transforms.input import Normalize

        from baybe.surrogates.gaussian_process.core import _ModelContext

        context = _ModelContext(searchspace)

        # Build the same normalization used by _fit for the new GP.
        # In eval mode, untransform reverses the normalization.
        input_transform = Normalize(
            train_x.shape[-1],
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )
        input_transform.eval()

        class PriorMean(gpytorch.means.Mean):
            """GPyTorch mean using a trained GP's posterior as the mean function."""

            def __init__(self, gp: SingleTaskGP, input_transform: Normalize) -> None:
                super().__init__()
                self.gp: SingleTaskGP = deepcopy(gp)
                for param in self.gp.parameters():
                    param.requires_grad = False
                self.gp.eval()
                self.gp.likelihood.eval()
                self.input_transform = input_transform

            def forward(self, x: Tensor) -> Tensor:
                """Compute the mean using the wrapped GP's posterior."""
                with gpytorch.settings.fast_pred_var():
                    # Unnormalize to raw physical space so that posterior()
                    # can apply the prior GP's own input normalization.
                    x_raw = self.input_transform.untransform(x)
                    return self.gp.posterior(x_raw).mean.squeeze(-1)

        return PriorMean(self._prior_model, input_transform)


gc.collect()  # Collect leftover original slotted classes created by attrs
