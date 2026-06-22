"""BoTorch preset for Gaussian process surrogates."""

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
from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterion,
    PlainFitCriterionFactory,
)
from baybe.surrogates.gaussian_process.components.kernel import (
    ICMKernelFactory,
    _PureKernelFactory,
)
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerLikelihoodFactory as BotorchLikelihoodFactory,
)
from baybe.surrogates.gaussian_process.presets.hvarfner import (
    HvarfnerMeanFactory as BotorchMeanFactory,
)

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel


@define
class BotorchKernelFactory(_PureKernelFactory):
    """A factory providing kernels matching BoTorch's :class:`~botorch.models.MultiTaskGP` defaults."""  # noqa: E501

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

        # BoTorch's MultiTaskGP added BetaPrior(2.5, 1.5) as the default task
        # covariance prior starting from version 0.18.0. For older versions, the
        # prior is not available and we fall back to no prior (matching that version's
        # MultiTaskGP behavior).
        try:
            from botorch.models.utils.priors import BetaPrior

            task_prior = BetaPrior(concentration1=2.5, concentration0=1.5)
        except ImportError:
            task_prior = None

        index_kernel = PositiveIndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            task_prior=task_prior,
            active_dims=[task_idx],
        )
        return ICMKernelFactory(base_kernel, index_kernel)(
            searchspace, objective, measurements
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Aliases for generic preset imports
KERNEL_FACTORY = BotorchKernelFactory()
MEAN_FACTORY = BotorchMeanFactory()
LIKELIHOOD_FACTORY = BotorchLikelihoodFactory()
FIT_CRITERION_FACTORY = PlainFitCriterionFactory(FitCriterion.MARGINAL_LOG_LIKELIHOOD)

__all__ = [
    "BotorchKernelFactory",
    "BotorchLikelihoodFactory",
    "BotorchMeanFactory",
    "FIT_CRITERION_FACTORY",
    "KERNEL_FACTORY",
    "LIKELIHOOD_FACTORY",
    "MEAN_FACTORY",
]
