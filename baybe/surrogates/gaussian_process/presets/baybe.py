"""Default preset for Gaussian process surrogates."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.basic import IndexKernel
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.kernel import (
    KernelFactory,
    KernelFactoryProtocol,
)
from baybe.surrogates.gaussian_process.presets.edbo_smoothed import (
    SmoothedEDBOKernelFactory,
    SmoothedEDBOLikelihoodFactory,
)
from baybe.surrogates.gaussian_process.presets.factories import LazyConstantMeanFactory

if TYPE_CHECKING:
    from torch import Tensor


@define
class DefaultKernelFactory(KernelFactoryProtocol):
    """The default kernel factory for Gaussian process surrogates."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        from baybe.surrogates.gaussian_process.components.kernel import ICMKernelFactory

        is_multitask = searchspace.n_tasks > 0
        factory = ICMKernelFactory if is_multitask else DefaultNumericalKernelFactory
        return factory()(searchspace, train_x, train_y)


DefaultNumericalKernelFactory = SmoothedEDBOKernelFactory
"""The factory providing the default numerical kernel for Gaussian process surrogates."""  # noqa: E501


@define
class DefaultTaskKernelFactory(KernelFactory):
    """The factory providing the default task kernel for Gaussian process surrogates."""

    _uses_parameter_names: ClassVar[bool] = True
    # See base class.

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return IndexKernel(
            num_tasks=searchspace.n_tasks,
            rank=searchspace.n_tasks,
            parameter_names=self.get_parameter_names(searchspace),
        )


DefaultMeanFactory = LazyConstantMeanFactory
"""The factory providing the default mean function for Gaussian process surrogates."""


DefaultLikelihoodFactory = SmoothedEDBOLikelihoodFactory
"""The factory providing the default likelihood for Gaussian process surrogates."""
