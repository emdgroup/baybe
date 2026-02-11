"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory
from baybe.surrogates.gaussian_process.likelihoods import LikelihoodFactory
from baybe.surrogates.gaussian_process.mean_factory import MeanFactory
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace


@define
class DefaultKernelFactory(KernelFactory):
    """A factory providing the default kernel for Gaussian process surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return SmoothedEDBOKernelFactory()(searchspace, train_x, train_y)


@define
class DefaultMeanFactory(MeanFactory):
    """A factory providing the default mean function for Gaussian process surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        return ConstantMean()


@define
class DefaultLikelihoodFactory(LikelihoodFactory):
    """A factory providing the default likelihood for Gaussian process surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchLikelihood:
        return SmoothedEDBOLikelihoodFactory()(searchspace, train_x, train_y)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
