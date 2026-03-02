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
from baybe.surrogates.gaussian_process.components._gpytorch import (
    make_botorch_multitask_likelihood,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
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
        from botorch.models.kernels.positive_index import PositiveIndexKernel
        from botorch.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )

        if searchspace.n_tasks == 1:
            return get_covar_module_with_dim_scaled_prior(
                ard_num_dims=len(searchspace.comp_rep_columns), active_dims=None
            )

        assert searchspace.task_idx is not None
        base_idcs = [
            idx
            for idx in range(len(searchspace.comp_rep_columns))
            if idx != searchspace.task_idx
        ]
        base = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(base_idcs), active_dims=base_idcs
        )
        index_kernel = PositiveIndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            active_dims=[searchspace.task_idx],
        )
        return ICMKernelFactory(base, index_kernel)(searchspace, train_x, train_y)


class BotorchMeanFactory(MeanFactoryProtocol):
    """A factory providing BoTorch mean functions."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> GPyTorchMean:
        from gpytorch.means import ConstantMean

        from baybe.surrogates.gaussian_process.components._gpytorch import (
            HadamardConstantMean,
        )

        if searchspace.n_tasks == 1:
            return ConstantMean()

        assert searchspace.task_idx is not None
        return HadamardConstantMean(
            ConstantMean(), searchspace.n_tasks, searchspace.task_idx
        )


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

        assert searchspace.task_idx is not None
        return make_botorch_multitask_likelihood(
            num_tasks=searchspace.n_tasks, task_feature=searchspace.task_idx
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Aliases for generic preset imports
PresetKernelFactory = BotorchKernelFactory
PresetMeanFactory = BotorchMeanFactory
PresetLikelihoodFactory = BotorchLikelihoodFactory
