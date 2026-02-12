"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, field
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.composite import ProductKernel
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.generic import (
    ComponentFactoryProtocol,
    PlainComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from torch import Tensor

    KernelFactoryProtocol = ComponentFactoryProtocol[Kernel | GPyTorchKernel]
    PlainKernelFactory = PlainComponentFactory[Kernel | GPyTorchKernel]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    KernelFactoryProtocol = ComponentFactoryProtocol[Kernel]
    PlainKernelFactory = PlainComponentFactory[Kernel]


@define
class ICMKernelFactory(KernelFactoryProtocol):
    """A kernel factory that constructs an ICM kernel for transfer learning.

    ICM: Intrinsic model of coregionalization
    """

    base_kernel_factory: KernelFactoryProtocol = field(alias="base_kernel_or_factory")
    """The factory for the base kernel operating on numerical input features."""

    task_kernel_factory: KernelFactoryProtocol = field(alias="task_kernel_or_factory")
    """The factory for the task kernel operating on the task indices."""

    @base_kernel_factory.default
    def _default_base_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            DefaultNumericalKernelFactory,
        )

        return DefaultNumericalKernelFactory()

    @task_kernel_factory.default
    def _default_task_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            DefaultTaskKernelFactory,
        )

        return DefaultTaskKernelFactory()

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        base_kernel = self.base_kernel_factory(searchspace, train_x, train_y)
        task_kernel = self.task_kernel_factory(searchspace, train_x, train_y)
        return ProductKernel([base_kernel, task_kernel])
