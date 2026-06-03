"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define, field
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.basic import MaternKernel, PositiveIndexKernel
from baybe.kernels.composite import ScaleKernel
from baybe.objectives.base import Objective
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import _ParameterKind
from baybe.parameters.selectors import (
    ParameterSelectorProtocol,
    TypeSelector,
    to_parameter_selector,
)
from baybe.priors.basic import GammaPrior
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
from baybe.surrogates.gaussian_process.components.mean import LazyConstantMeanFactory

if TYPE_CHECKING:
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood

    from baybe.searchspace.core import SearchSpace

# Noise prior mode: arithmetic midpoint of BoTorch (exp(-5) ~ 0.007) and
# Smoothed EDBO at d=1 (0.1).
_NOISE_PRIOR_MODE = 0.053

# Effective dimensionality at which the interpolated priors are close to the
# CHEN preset.
_CROSSOVER_DIM = 100


@define
class _BayBENumericalKernelFactory(_PureKernelFactory):
    """The default numerical kernel factory for GP surrogates.

    Uses dimension-dependent priors that interpolate between conservative low-d
    behavior and the CHEN preset at high d. The lengthscale prior mode starts near
    the Smoothed EDBO value at d=1 and asymptotically approaches the CHEN formula.
    The outputscale prior mode starts above CHEN and converges to it from above.
    Both transitions follow an exponential decay parameterized by
    ``_CROSSOVER_DIM``.
    """

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel:
        d = self._get_effective_dimensionality(searchspace)
        alpha = 3.0 / (_CROSSOVER_DIM - 1)
        transition = math.exp(-alpha * (d - 1))

        # Lengthscale prior: mode starts at ~0.18 (Smoothed EDBO at d=1),
        # converges to CHEN (0.4*sqrt(d) + 3.5).
        # Rate interpolates from 0.9 (wider at low d) to 2.0 (CHEN).
        ls_mode = 0.4 * math.sqrt(d) + 3.5 - 3.72 * transition
        ls_rate = 2.0 - 1.1 * transition
        ls_conc = ls_rate * ls_mode + 1
        lengthscale_prior = GammaPrior(ls_conc, ls_rate)
        lengthscale_initial_value = ls_mode

        # Outputscale prior: mode starts at 5.0 (above CHEN at d=1),
        # converges to CHEN (0.4*sqrt(d) + 3.0).
        # Rate interpolates from 0.6 (wider at low d) to 1.0 (CHEN).
        os_mode = 0.4 * math.sqrt(d) + 3.0 + 1.6 * transition
        os_rate = 1.0 - 0.4 * transition
        os_conc = os_rate * os_mode + 1
        outputscale_prior = GammaPrior(os_conc, os_rate)
        outputscale_initial_value = os_mode

        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
                parameter_names=self.get_parameter_names(searchspace),
            ),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
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


class BayBEMeanFactory(LazyConstantMeanFactory):
    """The default mean factory for GP surrogates."""


@define
class BayBELikelihoodFactory(LikelihoodFactoryProtocol):
    """The default likelihood factory for GP surrogates.

    Uses a flat Gamma noise prior with mode at the arithmetic midpoint between
    the BoTorch and Smoothed EDBO noise prior modes at d=1.
    """

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> GPyTorchLikelihood:
        import torch
        from gpytorch.likelihoods import GaussianLikelihood

        noise_rate = 2.0
        noise_conc = noise_rate * _NOISE_PRIOR_MODE + 1
        prior = GammaPrior(noise_conc, noise_rate)

        likelihood = GaussianLikelihood(prior.to_gpytorch())
        likelihood.noise = torch.tensor([_NOISE_PRIOR_MODE])
        return likelihood


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
