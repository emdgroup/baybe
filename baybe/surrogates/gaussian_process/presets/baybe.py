"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar, TypeVar

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.basic import PositiveIndexKernel
from baybe.objectives.base import Objective
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
from baybe.surrogates.gaussian_process.components.kernel import (
    _enable_transfer_learning,
    _PureKernelFactory,
)
from baybe.surrogates.gaussian_process.components.likelihood import (
    LikelihoodFactoryProtocol,
)
from baybe.surrogates.gaussian_process.components.mean import MeanFactoryProtocol
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOLikelihoodFactory,
    SmoothedEDBOMeanFactory,
    _SmoothedEDBONumericalKernelFactory,
)

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean

_T = TypeVar("_T")


def _dispatch(
    factory_with_substance: Callable[[SearchSpace, Objective, pd.DataFrame], _T],
    factory_without_substance: Callable[[SearchSpace, Objective, pd.DataFrame], _T],
    searchspace: SearchSpace,
    objective: Objective,
    measurements: pd.DataFrame,
) -> _T:
    """Select a GP component factory based on search space content.

    Delegates to ``factory_with_substance`` when a
    :class:`~baybe.parameters.substance.SubstanceParameter` is present in the
    search space, and to ``factory_without_substance`` otherwise.

    Args:
        factory_with_substance: The factory to use when a
            ``SubstanceParameter`` is present.
        factory_without_substance: The factory to use otherwise.
        searchspace: The search space.
        objective: The objective.
        measurements: The available measurements.

    Returns:
        The component produced by the selected factory.
    """
    # IMPROVE: Consider additional dispatch criteria such as dimensionality
    # or CustomDiscreteParameter presence in the future.
    from baybe.parameters.substance import SubstanceParameter

    if any(isinstance(p, SubstanceParameter) for p in searchspace.discrete.parameters):
        return factory_with_substance(searchspace, objective, measurements)
    return factory_without_substance(searchspace, objective, measurements)


@define
class _BayBENumericalKernelFactory(_PureKernelFactory):
    """The default numerical kernel factory for GP surrogates."""

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel:
        from baybe.surrogates.gaussian_process.presets.chen import (
            _ChenNumericalKernelFactory,
        )

        return _dispatch(
            _ChenNumericalKernelFactory(parameter_selector=self.parameter_selector),
            _SmoothedEDBONumericalKernelFactory(
                parameter_selector=self.parameter_selector
            ),
            searchspace,
            objective,
            measurements,
        )


BayBEKernelFactory = _enable_transfer_learning(
    _BayBENumericalKernelFactory, "BayBEKernelFactory"
)
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
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel:
        return PositiveIndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            parameter_names=self.get_parameter_names(searchspace),
        )


@define
class BayBEMeanFactory(MeanFactoryProtocol):
    """The default mean factory for GP surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchMean:
        from baybe.surrogates.gaussian_process.presets.chen import ChenMeanFactory

        return _dispatch(
            ChenMeanFactory(),
            SmoothedEDBOMeanFactory(),
            searchspace,
            objective,
            measurements,
        )


@define
class BayBELikelihoodFactory(LikelihoodFactoryProtocol):
    """The default likelihood factory for GP surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchLikelihood:
        from baybe.surrogates.gaussian_process.presets.chen import (
            ChenLikelihoodFactory,
        )

        return _dispatch(
            ChenLikelihoodFactory(),
            SmoothedEDBOLikelihoodFactory(),
            searchspace,
            objective,
            measurements,
        )


@define
class BayBEFitCriterionFactory(FitCriterionFactoryProtocol):
    """The factory providing the default fitting criterion for Gaussian process surrogates."""  # noqa: E501

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> FitCriterion:
        return (
            FitCriterion.MARGINAL_LOG_LIKELIHOOD
            if searchspace.task_idx is None
            else FitCriterion.LEAVE_ONE_OUT_PSEUDOLIKELIHOOD
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Preset defaults
KERNEL_FACTORY = BayBEKernelFactory()
MEAN_FACTORY = BayBEMeanFactory()
LIKELIHOOD_FACTORY = BayBELikelihoodFactory()
FIT_CRITERION_FACTORY = BayBEFitCriterionFactory()
