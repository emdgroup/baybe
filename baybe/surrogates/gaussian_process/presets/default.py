"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
from attrs import define

from baybe.kernels.basic import MaternKernel
from baybe.kernels.composite import ScaleKernel
from baybe.parameters import TaskParameter
from baybe.priors.basic import GammaPrior
from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace

# Boundaries for low and high dimension limits
_DIM_LIMITS = (8, 75)


@define
class DefaultKernelFactory(KernelFactory):
    """A factory providing the default kernel for Gaussian process surrogates.

    This is taking the low and high dimensional limits of
    :class:`baybe.surrogates.gaussian_process.presets.edbo.EDBOKernelFactory`
    and interpolates the prior moments linearly between them.
    """

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        # See base class.
        effective_dims = train_x.shape[-1] - len(
            [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
        )

        # Interpolate prior moments linearly between low D and high D regime
        # The high D regime itself is the average of the EDBO OHE and Mordred regime
        # Values outside the dimension limits will get the border value assigned
        lengthscale_prior = GammaPrior(
            np.interp(effective_dims, _DIM_LIMITS, [1.2, 2.5]),
            np.interp(effective_dims, _DIM_LIMITS, [1.1, 0.55]),
        )
        lengthscale_initial_value = np.interp(effective_dims, _DIM_LIMITS, [0.2, 6.0])
        outputscale_prior = GammaPrior(
            np.interp(effective_dims, _DIM_LIMITS, [5.0, 3.5]),
            np.interp(effective_dims, _DIM_LIMITS, [0.5, 0.15]),
        )
        outputscale_initial_value = np.interp(effective_dims, _DIM_LIMITS, [8.0, 15.0])

        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
            ),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )


def _default_noise_factory(
    searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
) -> tuple[GammaPrior, float]:
    """Create the default noise settings for the Gaussian process surrogate.

    This is taking the low and high dimensional limits of
    :func:`baybe.surrogates.gaussian_process.presets.edbo._edbo_noise_factory`
    and interpolates the prior moments linearly between them.
    """
    # TODO: Replace this function with a proper likelihood factory

    # Interpolate prior moments linearly between low D and high D regime
    # The high D regime itself is the average of the EDBO OHE and Mordred regime
    # Values outside the dimension limits will get the border value assigned
    effective_dims = train_x.shape[-1] - len(
        [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
    )
    return (
        GammaPrior(
            np.interp(effective_dims, _DIM_LIMITS, [1.05, 1.5]),
            np.interp(effective_dims, _DIM_LIMITS, [0.5, 0.1]),
        ),
        np.interp(effective_dims, _DIM_LIMITS, [0.1, 5.0]).item(),
    )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
