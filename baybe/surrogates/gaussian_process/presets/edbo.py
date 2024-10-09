"""EDBO preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

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


@define
class EDBOKernelFactory(KernelFactory):
    """A factory providing the kernel for Gaussian process surrogates adapted from EDBO.

    References:
        * https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
        * https://doi.org/10.1038/s41586-021-03213-y
    """

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        # See base class.
        effective_dims = train_x.shape[-1] - len(
            [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
        )

        mordred = (searchspace.contains_mordred or searchspace.contains_rdkit) and (
            effective_dims >= 50
        )

        # low D priors
        if effective_dims < 5:
            lengthscale_prior = GammaPrior(1.2, 1.1)
            lengthscale_initial_value = 0.2
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0

        # DFT optimized priors
        elif mordred and effective_dims < 100:
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

        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
            ),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )


def _edbo_noise_factory(
    searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
) -> tuple[GammaPrior, float]:
    """Create the default noise settings for the Gaussian process surrogate.

    The logic is adapted from EDBO (Experimental Design via Bayesian Optimization).

    References:
        * https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
        * https://doi.org/10.1038/s41586-021-03213-y
    """
    # TODO: Replace this function with a proper likelihood factory
    effective_dims = train_x.shape[-1] - len(
        [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
    )

    uses_descriptors = (
        searchspace.contains_mordred or searchspace.contains_rdkit
    ) and (effective_dims >= 50)

    # low D priors
    if effective_dims < 5:
        return (GammaPrior(1.05, 0.5), 0.1)

    # DFT optimized priors
    elif uses_descriptors and effective_dims < 100:
        return (GammaPrior(1.5, 0.1), 5.0)

    # Mordred optimized priors
    elif uses_descriptors:
        return (GammaPrior(1.5, 0.1), 5.0)

    # OHE optimized priors
    else:
        return (GammaPrior(1.5, 0.1), 5.0)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
