"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.basic import IndexKernel
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.kernel import KernelFactory
from baybe.surrogates.gaussian_process.components.mean import LazyConstantMeanFactory
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
)

if TYPE_CHECKING:
    from torch import Tensor


@define
class BayBEKernelFactory(KernelFactory):
    """The default kernel factory for Gaussian process surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory

        is_multitask = searchspace.n_tasks > 0
        factory = ICMKernelFactory if is_multitask else BayBENumericalKernelFactory
        return factory()(searchspace, train_x, train_y)


BayBENumericalKernelFactory = SmoothedEDBOKernelFactory
"""The factory providing the default numerical kernel for Gaussian process surrogates."""  # noqa: E501


@define
class BayBETaskKernelFactory(KernelFactory):
    """The factory providing the default task kernel for Gaussian process surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return IndexKernel(num_tasks=searchspace.n_tasks, rank=searchspace.n_tasks)


BayBEMeanFactory = LazyConstantMeanFactory
"""The factory providing the default mean function for Gaussian process surrogates."""

BayBELikelihoodFactory = SmoothedEDBOLikelihoodFactory
"""The factory providing the default likelihood for Gaussian process surrogates."""

# Aliases for generic preset imports
PresetKernelFactory = BayBEKernelFactory
PresetMeanFactory = BayBEMeanFactory
PresetLikelihoodFactory = BayBELikelihoodFactory
