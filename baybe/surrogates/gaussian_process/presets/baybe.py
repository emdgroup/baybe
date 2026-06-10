"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable
from itertools import chain
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
from baybe.surrogates.gaussian_process.components.mean import (
    LazyConstantMeanFactory,
    MeanFactoryProtocol,
)

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean

_T = TypeVar("_T")


##### Private custom-scaled factories #####


@define
class _CustomScaledNumericalKernelFactory(_PureKernelFactory):
    """A numerical kernel factory with dimension-scaled Gamma lengthscale prior.

    Inspired by the dimension-scaled priors in :cite:p:`Hvarfner2024` but with slight
    adjustments:

    * Uses Matern instead of RBF kernel.
    * Uses a Gamma prior instead of a LogNormal prior for faster convergence (less heavy
      tails). The parameters of the Gamma are set such that:
        - The concentration matches that of the conventional (i.e., Hvarfner
          predecessor) Gamma distribution used by BoTorch.
        - The mode matches that of the LogNormal and thus also scales with sqrt(d).
    """

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        from gpytorch.constraints import GreaterThan
        from gpytorch.kernels import MaternKernel
        from gpytorch.priors import GammaPrior

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

        concentration = 3.0
        rate = (
            (concentration - 1) / math.exp(math.sqrt(2) - 3) / math.sqrt(ard_num_dims)
        )
        lengthscale_prior = GammaPrior(concentration, rate)
        base_kernel = MaternKernel(
            ard_num_dims=ard_num_dims,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mode
            ),
            active_dims=active_dims,
        )

        return base_kernel


@define
class _CustomScaledLikelihoodFactory(LikelihoodFactoryProtocol):
    """A likelihood factory with dimension-scaled Gamma noise prior.

    Inspired by the likelihood proposed in :cite:p:`Hvarfner2024` but uses a Gamma prior
    instead of a LogNormal prior for faster convergence (less heavy tails). The
    parameters of the Gamma are set such that:

    * The mode matches that of the LogNormal and thus also scales with sqrt(d).
    * The curvature at the mode is slightly stronger than that of the LogNormal,
      resulting in similar but more attracted convergence behavior in the vicinity of
      the mode.
    """

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchLikelihood:
        from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
        from gpytorch.constraints import GreaterThan
        from gpytorch.likelihoods import GaussianLikelihood
        from gpytorch.priors import GammaPrior

        concentration = 3.0  # using 2.0 would match the Hvarfner LogNormal curvature
        rate = (concentration - 1) / math.exp(-4.0 - 1.0**2)
        noise_prior = GammaPrior(concentration, rate)
        return GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                MIN_INFERRED_NOISE_LEVEL,
                transform=None,
                initial_value=noise_prior.mode,
            ),
        )


class _CustomScaledMeanFactory(LazyConstantMeanFactory):
    """A mean factory for the custom-scaled preset."""


##### Dispatch logic #####


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


##### Public BayBE component factories #####


@define
class _BayBENumericalKernelFactory(_PureKernelFactory):
    """The default numerical kernel factory for GP surrogates."""

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        from baybe.surrogates.gaussian_process.presets.chen import (
            _ChenNumericalKernelFactory,
        )

        return _dispatch(
            _ChenNumericalKernelFactory(parameter_selector=self.parameter_selector),
            _CustomScaledNumericalKernelFactory(
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
            _CustomScaledMeanFactory(),
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
        from baybe.surrogates.gaussian_process.presets.chen import ChenLikelihoodFactory

        return _dispatch(
            ChenLikelihoodFactory(),
            _CustomScaledLikelihoodFactory(),
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
