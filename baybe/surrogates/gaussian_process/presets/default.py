"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define

from baybe.kernels.basic import MaternKernel, ScaleKernel
from baybe.priors.basic import GammaPrior
from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace


@define
class DefaultKernelFactory(KernelFactory):
    """A factory providing the default kernel for Gaussian process surrogates.

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        # See base class.

        # TODO: temporary prior choices adapted from edbo, replace later on
        mordred = (searchspace.contains_mordred or searchspace.contains_rdkit) and (
            train_x.shape[-1] >= 50
        )

        # TODO Until now, only the kernels use our custom priors, hence the explicit
        # to_gpytorch() calls for all others
        # low D priors
        if train_x.shape[-1] < 10:
            lengthscale_prior = GammaPrior(1.2, 1.1)
            lengthscale_initial_value = 0.2
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0

        # DFT optimized priors
        elif mordred and train_x.shape[-1] < 100:
            lengthscale_prior = GammaPrior(2.0, 0.2)
            lengthscale_initial_value = 5.0
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0

        # Mordred optimized priors
        elif mordred:
            lengthscale_prior = GammaPrior(2.0, 0.1)
            lengthscale_initial_value = 10.0
            outputscale_prior = GammaPrior(2.0, 0.1)
            outputscale_initial_value = 10.0

        # OHE optimized priors
        else:
            lengthscale_prior = GammaPrior(3.0, 1.0)
            lengthscale_initial_value = 2.0
            outputscale_prior = GammaPrior(5.0, 0.2)
            outputscale_initial_value = 20.0

        # ---------- End: GP prior selection ---------- #

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
    """Create the default noise settings for the Gaussian process surrogate."""
    # TODO: Replace this function with a proper likelihood factory

    # TODO: temporary prior choices adapted from edbo, replace later on
    mordred = (searchspace.contains_mordred or searchspace.contains_rdkit) and (
        train_x.shape[-1] >= 50
    )

    # TODO Until now, only the kernels use our custom priors, hence the explicit
    # to_gpytorch() calls for all others
    # low D priors
    if train_x.shape[-1] < 10:
        return [GammaPrior(1.05, 0.5), 0.1]

    # DFT optimized priors
    elif mordred and train_x.shape[-1] < 100:
        return [GammaPrior(1.5, 0.1), 5.0]

    # Mordred optimized priors
    elif mordred:
        return [GammaPrior(1.5, 0.1), 5.0]

    # OHE optimized priors
    else:
        return [GammaPrior(1.5, 0.1), 5.0]
