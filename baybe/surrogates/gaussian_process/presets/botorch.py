"""Presets adapted from BoTorch."""

from __future__ import annotations

from math import log, sqrt
from typing import TYPE_CHECKING

from attrs import define
from gpytorch.constraints import GreaterThan
from typing_extensions import override

from baybe.kernels.basic import RBFKernel
from baybe.parameters import TaskParameter
from baybe.priors.basic import LogNormalPrior
from baybe.searchspace import SearchSpace
from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.kernels.base import Kernel


@define
class BotorchKernelFactory(KernelFactory):
    """A kernel factory for Gaussian process surrogates adapted from BoTorch.

    References:
        * https://github.com/pytorch/botorch/blob/a018a5ffbcbface6229d6c39f7ac6ef9baf5765e/botorch/models/multitask.py#L220
        * https://github.com/pytorch/botorch/blob/a018a5ffbcbface6229d6c39f7ac6ef9baf5765e/botorch/models/utils/gpytorch_modules.py#L100

    """

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        ard_num_dims = train_x.shape[-1] - len(
            [
                param
                for param in searchspace.discrete.parameters
                if isinstance(param, TaskParameter)
            ]
        )
        lengthscale_prior = LogNormalPrior(
            loc=sqrt(2) + log(ard_num_dims) * 0.5, scale=sqrt(3)
        )

        return RBFKernel(
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=GreaterThan(
                2.5e-2,
                transform=None,
                initial_value=lengthscale_prior.to_gpytorch().mode,
            ),
        )
