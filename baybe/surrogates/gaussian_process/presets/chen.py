"""Preset for adaptive kernel hyperpriors proposed by :cite:p:`Chen2026`."""

from __future__ import annotations

import gc
import math
from typing import TYPE_CHECKING, ClassVar

from attrs import define
from typing_extensions import override

from baybe.kernels.basic import MaternKernel
from baybe.kernels.composite import ScaleKernel
from baybe.priors.basic import GammaPrior
from baybe.surrogates.gaussian_process.components.kernel import (
    _PureKernelFactory,
)
from baybe.surrogates.gaussian_process.presets.baybe import (
    BayBELikelihoodFactory,
    BayBEMeanFactory,
)

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace


@define
class CHENKernelFactory(_PureKernelFactory):
    """A factory providing adaptive hyperprior kernels as proposed by :cite:p:`Chen2026`."""  # noqa: E501

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def _make(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        lengthscale = 0.4 * math.sqrt(train_x.shape[-1]) + 4.0
        lengthscale_prior = GammaPrior(2.0 * lengthscale, 2.0)
        lengthscale_initial_value = lengthscale
        outputscale_prior = GammaPrior(1.0 * lengthscale, 1.0)
        outputscale_initial_value = lengthscale

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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

# Aliases for generic preset imports
PresetKernelFactory = CHENKernelFactory
PresetMeanFactory = BayBEMeanFactory
PresetLikelihoodFactory = BayBELikelihoodFactory
