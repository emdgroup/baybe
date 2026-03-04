"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.composite import ProductKernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.selector import (
    ParameterSelectorProtocol,
    TypeSelector,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    PlainGPComponentFactory,
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
class KernelFactory(KernelFactoryProtocol):
    """Base class for kernel factories."""

    # For internal use only: sanity check mechanism to remind developers of new
    # factories to actually use the parameter selector when it is provided
    # TODO: Perhaps we can find a more elegant way to enforce this by design
    _uses_parameter_names: ClassVar[bool] = False

    parameter_selector: ParameterSelectorProtocol | None = field(default=None)
    """An optional selector to specify which parameters are considered by the kernel."""

    def get_parameter_names(self, searchspace: SearchSpace) -> tuple[str, ...] | None:
        """Get the names of the parameters to be considered by the kernel."""
        if self.parameter_selector is None:
            return None

        return tuple(
            p.name for p in searchspace.parameters if self.parameter_selector(p)
        )

    def __attrs_post_init__(self):
        # This helps to ensure that new factories actually use the parameter selector
        # by requiring the developer to explicitly set the flag to `True`
        if self.parameter_selector is not None:
            assert self._uses_parameter_names


@define
class ICMKernelFactory(KernelFactoryProtocol):
    """A kernel factory that constructs an ICM kernel for transfer learning.

    ICM: Intrinsic Coregionalization Model :cite:p:`NIPS2007_66368270`
    """

    base_kernel_factory: KernelFactoryProtocol = field(alias="base_kernel_or_factory")
    """The factory for the base kernel operating on numerical input features."""

    task_kernel_factory: KernelFactoryProtocol = field(alias="task_kernel_or_factory")
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
