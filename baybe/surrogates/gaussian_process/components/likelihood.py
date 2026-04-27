"""Likelihood factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from attrs import define
from typing_extensions import override

from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    PlainGPComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from torch import Tensor

    LikelihoodFactoryProtocol = GPComponentFactoryProtocol[GPyTorchLikelihood]
    PlainLikelihoodFactory = PlainGPComponentFactory[GPyTorchLikelihood]
else:
    # At runtime, we avoid loading GPyTorch eagerly for performance reasons
    LikelihoodFactoryProtocol = GPComponentFactoryProtocol[Any]
    PlainLikelihoodFactory = PlainGPComponentFactory[Any]


@define
class LazyGaussianLikelihoodFactory(LikelihoodFactoryProtocol):
    """A factory providing Gaussian likelihoods using lazy loading."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchLikelihood:
        from gpytorch.likelihoods import GaussianLikelihood

        return GaussianLikelihood()
