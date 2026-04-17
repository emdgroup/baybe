"""BoTorch preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from itertools import chain
from typing import TYPE_CHECKING, ClassVar

from attrs import define
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.parameters.enum import _ParameterKind
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components import LikelihoodFactoryProtocol
from baybe.surrogates.gaussian_process.components._gpytorch import (
    make_botorch_multitask_likelihood,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
    _PureKernelFactory,
)
from baybe.surrogates.gaussian_process.components.mean import MeanFactoryProtocol
from baybe.surrogates.gaussian_process.presets.baybe import BayBEFitCriterionFactory

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor


@define
class BotorchKernelFactory(_PureKernelFactory):
    """A factory providing BoTorch kernels."""

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    _supported_parameter_kinds: ClassVar[_ParameterKind] = (
        _ParameterKind.REGULAR | _ParameterKind.TASK
    )
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel | GPyTorchKernel:
        from botorch.models.kernels.positive_index import PositiveIndexKernel
        from botorch.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )

        parameter_names = self.get_parameter_names(searchspace)

        # Resolve parameter names to active dimension indices
        active_dims: list[int] | None
        if parameter_names is not None:
            active_dims = list(
                chain.from_iterable(
                    searchspace.get_comp_rep_parameter_indices(name)
                    for name in parameter_names
                )
            )
            ard_num_dims = len(active_dims)
        else:
            active_dims = None
            ard_num_dims = len(searchspace.comp_rep_columns)

        # Determine if the selected parameters include a task parameter
        task_idx = searchspace.task_idx
        is_multitask = task_idx is not None and (
            active_dims is None or task_idx in active_dims
        )

        if not is_multitask:
            return get_covar_module_with_dim_scaled_prior(
                ard_num_dims=ard_num_dims, active_dims=active_dims
            )

        assert task_idx is not None
        base_idcs = [
            idx
            for idx in (active_dims or range(len(searchspace.comp_rep_columns)))
            if idx != task_idx
        ]
        base = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=len(base_idcs), active_dims=base_idcs
        )
        index_kernel = PositiveIndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            active_dims=[task_idx],
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
KERNEL_FACTORY = BotorchKernelFactory()
MEAN_FACTORY = BotorchMeanFactory()
LIKELIHOOD_FACTORY = BotorchLikelihoodFactory()
# Botorch dictates no specific criterion, so we fill the preset with our default
FIT_CRITERION_FACTORY = BayBEFitCriterionFactory()
