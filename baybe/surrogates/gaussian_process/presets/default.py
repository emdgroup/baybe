"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

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
class DefaultKernelFactory(KernelFactory):
    """A factory providing the default kernel for Gaussian process surrogates.

    This is taking the low and high dimensional limits of
    :class:`baybe.surrogates.gaussian_process.presets.default.EDBOKernelFactory`
    and interpolates the prior moments linearly between them.
    """

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        # See base class.

        # Calculate effective dimensions, ignoring task parameters
        n_task_parameters = len(
            [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
        )
        effective_dims = train_x.shape[-1] - n_task_parameters

        # Fix dimensionality limits
        LOW_D_LIM, HIGH_D_LIM = 8, 100

        # Apply prior logic
        if effective_dims < LOW_D_LIM:
            # Low D priors
            lengthscale_prior = GammaPrior(1.2, 1.1)
            lengthscale_initial_value = 0.2
            outputscale_prior = GammaPrior(5.0, 0.5)
            outputscale_initial_value = 8.0
        elif effective_dims > HIGH_D_LIM:
            # High D priors
            lengthscale_prior = GammaPrior(2.5, 0.55)
            lengthscale_initial_value = 6.0
            outputscale_prior = GammaPrior(3.5, 0.15)
            outputscale_initial_value = 15.0
        else:
            # Interpolate prior moments linearly between low D and high D regime
            interp_factor = (effective_dims - LOW_D_LIM) / (HIGH_D_LIM - LOW_D_LIM)

            def _interp(a, b):
                return a + (b - a) * interp_factor

            lengthscale_prior = GammaPrior(_interp(1.2, 2.5), _interp(1.1, 0.55))
            lengthscale_initial_value = _interp(0.2, 6.0)
            outputscale_prior = GammaPrior(_interp(5.0, 3.5), _interp(0.5, 0.15))
            outputscale_initial_value = _interp(8.0, 15.0)

        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
            ),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )


@define
class EDBOKernelFactory(KernelFactory):
    """A factory providing the kernel for Gaussian process surrogates.

    The logic is adapted from EDBO (Experimental Design via Bayesian Optimization).

    References:
        * https://github.com/b-shields/edbo/blob/master/edbo/bro.py#L664
        * https://doi.org/10.1038/s41586-021-03213-y
    """

    def __call__(  # noqa: D102
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        # See base class.

        # Calculate effective dimensions, ignoring task parameters
        n_task_parameters = len(
            [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
        )
        effective_dims = train_x.shape[-1] - n_task_parameters

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


def _default_noise_factory(
    searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
) -> tuple[GammaPrior, float]:
    """Create the default noise settings for the Gaussian process surrogate.

    This is taking the low and high dimensional limits of
    :func:`baybe.surrogates.gaussian_process.presets.default._edbo_noise_factory`
    and interpolates the prior moments linearly between them.
    """
    # TODO: Replace this function with a proper likelihood factory

    # Calculate effective dimensions, ignoring task parameters
    n_task_parameters = len(
        [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
    )
    effective_dims = train_x.shape[-1] - n_task_parameters

    # Fix dimensionality limits
    LOW_D_LIM, HIGH_D_LIM = 8, 100

    # Apply prior logic
    if effective_dims < LOW_D_LIM:
        # Low D prior
        return [GammaPrior(1.05, 0.5), 0.1]
    elif effective_dims > HIGH_D_LIM:
        # High D prior
        return [GammaPrior(1.5, 0.1), 5.0]
    else:
        # Interpolate prior moments linearly between low D and high D regime
        interp_factor = (effective_dims - LOW_D_LIM) / (HIGH_D_LIM - LOW_D_LIM)

        def _interp(a, b):
            return a + (b - a) * interp_factor

        return [GammaPrior(_interp(1.05, 1.5), _interp(0.5, 0.1)), _interp(0.1, 5.0)]


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

    # Calculate effective dimensions, ignoring task parameters
    n_task_parameters = len(
        [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
    )
    effective_dims = train_x.shape[-1] - n_task_parameters

    uses_descriptors = (
        searchspace.contains_mordred or searchspace.contains_rdkit
    ) and (effective_dims >= 50)

    # low D priors
    if effective_dims < 5:
        return [GammaPrior(1.05, 0.5), 0.1]

    # DFT optimized priors
    elif uses_descriptors and effective_dims < 100:
        return [GammaPrior(1.5, 0.1), 5.0]

    # Mordred optimized priors
    elif uses_descriptors:
        return [GammaPrior(1.5, 0.1), 5.0]

    # OHE optimized priors
    else:
        return [GammaPrior(1.5, 0.1), 5.0]
