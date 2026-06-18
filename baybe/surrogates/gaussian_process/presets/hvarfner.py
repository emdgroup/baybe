"""Hvarfner preset :cite:p:`Hvarfner2024`."""

from __future__ import annotations

import gc
from itertools import chain
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.objectives.base import Objective
from baybe.parameters.enum import _ParameterKind
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components import LikelihoodFactoryProtocol
from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterion,
    PlainFitCriterionFactory,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
    _PureKernelFactory,
)
from baybe.surrogates.gaussian_process.components.mean import MeanFactoryProtocol

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean


@define
class HvarfnerKernelFactory(_PureKernelFactory):
    """A factory providing kernels with dimension-scaled priors as proposed by :cite:p:`Hvarfner2024`."""  # noqa: E501

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    _supported_parameter_kinds: ClassVar[_ParameterKind] = (
        _ParameterKind.REGULAR | _ParameterKind.TASK
    )
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        from botorch.models.kernels.positive_index import PositiveIndexKernel
        from botorch.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )

        parameter_names = self.get_parameter_names(searchspace)

        # For regular parameters, resolve parameter names to active dimension indices
        active_dims = list(
            chain.from_iterable(
                searchspace.get_comp_rep_parameter_indices(name)
                for name in parameter_names
                if searchspace.get_parameters_by_name([name])[0]._kind
                is _ParameterKind.REGULAR
            )
        )
        ard_num_dims = len(active_dims)

        # Create the base kernel for the regular parameters
        base_kernel = get_covar_module_with_dim_scaled_prior(
            ard_num_dims=ard_num_dims, active_dims=active_dims
        )

        # Single-task case
        if (task_idx := searchspace.task_idx) is None:
            return base_kernel

        index_kernel = PositiveIndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            active_dims=[task_idx],
        )
        return ICMKernelFactory(base_kernel, index_kernel)(
            searchspace, objective, measurements
        )


class HvarfnerMeanFactory(MeanFactoryProtocol):
    """A factory providing mean functions as proposed by :cite:p:`Hvarfner2024`."""

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
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


class HvarfnerLikelihoodFactory(LikelihoodFactoryProtocol):
    """A factory providing likelihoods as proposed by :cite:p:`Hvarfner2024`."""

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchLikelihood:

        if searchspace.n_tasks == 1:
            from botorch.models.utils.gpytorch_modules import (
                get_gaussian_likelihood_with_lognormal_prior,
            )

            return get_gaussian_likelihood_with_lognormal_prior()

        from baybe.surrogates.gaussian_process.components._gpytorch import (
            make_botorch_multitask_likelihood,
        )

        assert searchspace.task_idx is not None
        return make_botorch_multitask_likelihood(
            num_tasks=searchspace.n_tasks, task_feature=searchspace.task_idx
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Aliases for generic preset imports
KERNEL_FACTORY = HvarfnerKernelFactory()
MEAN_FACTORY = HvarfnerMeanFactory()
LIKELIHOOD_FACTORY = HvarfnerLikelihoodFactory()
FIT_CRITERION_FACTORY = PlainFitCriterionFactory(FitCriterion.MARGINAL_LOG_LIKELIHOOD)
