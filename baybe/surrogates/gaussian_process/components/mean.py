"""Mean factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization.core import (
    block_deserialization_hook,
    block_serialization_hook,
    converter,
)
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
    """A factory protocol for Gaussian process mean functions."""

    PlainMeanFactory = PlainGPComponentFactory[Any]
    """A trivial factory returning a fixed, pre-defined mean function."""


@define
class LazyConstantMeanFactory(MeanFactoryProtocol):
    """A factory providing constant mean functions using lazy loading."""

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        return ConstantMean()


@define
class _PosteriorMeanFactory(MeanFactoryProtocol):
    """A mean factory producing a posterior mean from a trained BoTorch GP.

    The mean function uses the trained GP's posterior mean predictions.
    The provided model is deep-copied and its parameters are frozen.

    Surrogates using this factory are not serializable because the underlying
    BoTorch model is not covered by BayBE's serialization system.
    """

    _pretrained_gp = field(alias="pretrained_gp")
    """The pretrained BoTorch GP whose posterior mean is used as the mean function."""

    @override
    def __call__(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> GPyTorchMean:
        import gpytorch
        from botorch.models.transforms.input import Normalize

        from baybe.surrogates.gaussian_process.core import _ModelContext

        context = _ModelContext(searchspace, objective, measurements)

        # The new GP applies its input normalization before calling this mean module,
        # so x arrives in the new GP's scaled coordinate system. Undo that scaling
        # before calling the pretrained GP — it will apply its own normalization.
        input_transform = Normalize(
            len(searchspace.comp_rep_columns),
            bounds=context.parameter_bounds,
            indices=context.numerical_indices,
        )
        input_transform.eval()

        class _PosteriorMean(gpytorch.means.Mean):
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
                    x_raw = self.input_transform.untransform(x)
                    return self.gp.posterior(x_raw).mean.squeeze(-1)

        return _PosteriorMean(self._pretrained_gp, input_transform)


# Prevent (de-)serialization since it wraps a raw BoTorch model
converter.register_unstructure_hook(_PosteriorMeanFactory, block_serialization_hook)
converter.register_structure_hook(_PosteriorMeanFactory, block_deserialization_hook)

gc.collect()  # Collect leftover original slotted classes created by attrs
