"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.basic import IndexKernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import _ParameterKind
from baybe.parameters.selectors import (
    ParameterSelectorProtocol,
    TypeSelector,
    to_parameter_selector,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.fit_criterion import (
    FitCriterion,
    FitCriterionFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.kernel import _PureKernelFactory
from baybe.surrogates.gaussian_process.components.mean import LazyConstantMeanFactory
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
    _SmoothedEDBONumericalKernelFactory,
)

if TYPE_CHECKING:
    from torch import Tensor


class _BayBENumericalKernelFactory(_SmoothedEDBONumericalKernelFactory):
    """The default numerical kernel factory for GP surrogates."""


class BayBEKernelFactory(SmoothedEDBOKernelFactory):  # type: ignore[valid-type, misc]
    """The default kernel factory for GP surrogates."""


@define
class _BayBETaskKernelFactory(_PureKernelFactory):
    """The default task kernel factory for GP surrogates."""

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    _supported_parameter_kinds: ClassVar[_ParameterKind] = _ParameterKind.TASK
    # See base class.

    parameter_selector: ParameterSelectorProtocol | None = field(
        factory=lambda: TypeSelector([TaskParameter]),
        converter=to_parameter_selector,
    )
    # TODO: Reuse base attribute (https://github.com/python-attrs/attrs/pull/1429)

    @override
    def _make(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return IndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            parameter_names=self.get_parameter_names(searchspace),
        )


class BayBEMeanFactory(LazyConstantMeanFactory):
    """The default mean factory for GP surrogates."""


class BayBELikelihoodFactory(SmoothedEDBOLikelihoodFactory):
    """The default likelihood factory for GP surrogates."""


@define
class BayBEFitCriterionFactory(FitCriterionFactoryProtocol):
    """The factory providing the default fitting criterion for Gaussian process surrogates."""  # noqa: E501

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> FitCriterion:
        return (
            FitCriterion.MARGINAL_LOG_LIKELIHOOD
            if searchspace.task_idx is None
            else FitCriterion.LEAVE_ONE_OUT_PSEUDOLIKELIHOOD
        )


# Preset defaults
KERNEL_FACTORY = BayBEKernelFactory()
MEAN_FACTORY = BayBEMeanFactory()
LIKELIHOOD_FACTORY = BayBELikelihoodFactory()
FIT_CRITERION_FACTORY = BayBEFitCriterionFactory()
