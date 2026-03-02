"""BoTorch preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define
from gpytorch.kernels import Kernel as GPyTorchKernel
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components import LikelihoodFactoryProtocol
from baybe.surrogates.gaussian_process.components.kernel import (
    KernelFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.mean import MeanFactoryProtocol

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor


@define
class BotorchKernelFactory(KernelFactoryProtocol):
    """A factory providing BoTorch kernels."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel | GPyTorchKernel:
        from botorch.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )

        base = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(searchspace.comp_rep_columns)
        )

        if searchspace.n_tasks == 1:
            return base

        raise NotImplementedError()


class BotorchMeanFactory(MeanFactoryProtocol):
    """A factory providing BoTorch mean functions."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        if searchspace.n_tasks == 1:
            return ConstantMean()

        raise NotImplementedError()


class BotorchLikelihoodFactory(LikelihoodFactoryProtocol):
    """A factory providing BoTorch likelihoods."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchLikelihood:
        from botorch.models.utils.gpytorch_modules import (
            get_gaussian_likelihood_with_lognormal_prior,
        )

        if searchspace.n_tasks == 1:
            return get_gaussian_likelihood_with_lognormal_prior()

        raise NotImplementedError()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Aliases for generic preset imports
PresetKernelFactory = BotorchKernelFactory
PresetMeanFactory = BotorchMeanFactory
PresetLikelihoodFactory = BotorchLikelihoodFactory
