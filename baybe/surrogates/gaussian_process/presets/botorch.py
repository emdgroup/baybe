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

# The minimum BoTorch version required for the preset
_MIN_BOTORCH_VERSION = "0.18.0"


@define
class BotorchKernelFactory(_PureKernelFactory):
    """A factory providing kernels matching BoTorch's :class:`~botorch.models.MultiTaskGP` defaults."""  # noqa: E501

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    _supported_parameter_kinds: ClassVar[_ParameterKind] = (
        _ParameterKind.REGULAR | _ParameterKind.TASK
    )
    # See base class.

    def __attrs_pre_init__(self) -> None:
        self._validate_botorch_version()

    @override
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:

        from botorch.models.kernels.positive_index import PositiveIndexKernel
        from botorch.models.utils.gpytorch_modules import (
            get_covar_module_with_dim_scaled_prior,
        )
        from botorch.models.utils.priors import BetaPrior

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

        task_prior = BetaPrior(concentration1=2.5, concentration0=1.5)
        index_kernel = PositiveIndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            task_prior=task_prior,
            active_dims=[task_idx],
        )
        return ICMKernelFactory(base_kernel, index_kernel)(
            searchspace, objective, measurements
        )

    def _validate_botorch_version(self) -> None:
        """Verify that the installed BoTorch version meets the minimum requirement.

        Raises:
            IncompatibilityError: If the installed BoTorch version is too old.
        """
        from importlib.metadata import version

        from packaging.version import Version

        from baybe.exceptions import IncompatibilityError

        installed = version("botorch")
        if Version(installed) < Version(_MIN_BOTORCH_VERSION):
            raise IncompatibilityError(
                f"The '{self.__class__.__name__}' requires BoTorch >= "
                f"{_MIN_BOTORCH_VERSION}, but version {installed} is installed. "
                f"Please upgrade BoTorch: pip install 'botorch>="
                f"{_MIN_BOTORCH_VERSION}'."
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
