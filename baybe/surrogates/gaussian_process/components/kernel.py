"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.validators import is_callable
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.composite import ProductKernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.selectors import (
    TypeSelector,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    GPComponentType,
    PlainGPComponentFactory,
    to_component_factory,
)

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from torch import Tensor

    KernelFactoryProtocol = GPComponentFactoryProtocol[Kernel | GPyTorchKernel]
    PlainKernelFactory = PlainGPComponentFactory[Kernel | GPyTorchKernel]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    KernelFactoryProtocol = GPComponentFactoryProtocol[Kernel]
    PlainKernelFactory = PlainGPComponentFactory[Kernel]


@define
class ICMKernelFactory(KernelFactoryProtocol):
    """A kernel factory that constructs an ICM kernel for transfer learning.

    ICM: Intrinsic Coregionalization Model :cite:p:`NIPS2007_66368270`
    """

    base_kernel_factory: KernelFactoryProtocol = field(
        alias="base_kernel_or_factory",
        converter=partial(to_component_factory, component_type=GPComponentType.KERNEL),
        validator=is_callable(),
    )
    """The factory for the base kernel operating on numerical input features."""

    task_kernel_factory: KernelFactoryProtocol = field(
        alias="task_kernel_or_factory",
        converter=partial(to_component_factory, component_type=GPComponentType.KERNEL),
        validator=is_callable(),
    )
    """The factory for the task kernel operating on the task indices."""

    @base_kernel_factory.default
    def _default_base_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            BayBENumericalKernelFactory,
        )

        return BayBENumericalKernelFactory(TypeSelector((TaskParameter,), exclude=True))

    @task_kernel_factory.default
    def _default_task_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            BayBETaskKernelFactory,
        )

        return BayBETaskKernelFactory(TypeSelector((TaskParameter,)))

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        base_kernel = self.base_kernel_factory(searchspace, train_x, train_y)
        task_kernel = self.task_kernel_factory(searchspace, train_x, train_y)
        return ProductKernel([base_kernel, task_kernel])
