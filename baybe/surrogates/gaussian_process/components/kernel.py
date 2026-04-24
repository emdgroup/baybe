"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from attrs.converters import optional
from attrs.validators import is_callable
from typing_extensions import override

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.kernels.base import Kernel
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import ParameterKind
from baybe.parameters.selectors import (
    ParameterSelectorProtocol,
    TypeSelector,
    to_parameter_selector,
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

    from baybe.parameters.base import Parameter

    KernelFactoryProtocol = GPComponentFactoryProtocol[Kernel | GPyTorchKernel]
    PlainKernelFactory = PlainGPComponentFactory[Kernel | GPyTorchKernel]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    KernelFactoryProtocol = GPComponentFactoryProtocol[Kernel]
    PlainKernelFactory = PlainGPComponentFactory[Kernel]


@define
class _PureKernelFactory(KernelFactoryProtocol, ABC):
    """Base class for pure kernel factories."""

    # For internal use only: sanity check mechanism to remind developers of new
    # factories to actually use the parameter selector when it is provided
    # TODO: Perhaps we can find a more elegant way to enforce this by design
    _uses_parameter_names: ClassVar[bool] = False

    supported_parameter_kinds: ClassVar[ParameterKind] = ParameterKind.REGULAR
    """The parameter kinds supported by the kernel factory."""

    parameter_selector: ParameterSelectorProtocol | None = field(
        default=None, converter=optional(to_parameter_selector)
    )
    """An optional selector to specify which parameters are considered by the kernel."""

    def __attrs_post_init__(self):
        if self.parameter_selector is not None and not self._uses_parameter_names:
            raise AssertionError(
                f"A `parameter_selector` was provided to "
                f"`{type(self).__name__}`, but the class does not set "
                f"`_uses_parameter_names = True`. Subclasses that accept a "
                f"parameter selector must explicitly set this flag to confirm "
                f"they actually use the selected parameter names."
            )

    def get_parameter_names(self, searchspace: SearchSpace) -> tuple[str, ...] | None:
        """Get the names of the parameters to be considered by the kernel."""
        if self.parameter_selector is None:
            return None

        return tuple(
            p.name for p in searchspace.parameters if self.parameter_selector(p)
        )

    def _validate_parameter_kinds(self, parameters: Iterable[Parameter]) -> None:
        """Validate that the given parameters are supported by the factory.

        Args:
            parameters: The parameters to validate.

        Raises:
            IncompatibleSearchSpaceError: If unsupported parameter kinds are found.
        """
        if unsupported := [
            p.name for p in parameters if not (p.kind & self.supported_parameter_kinds)
        ]:
            raise IncompatibleSearchSpaceError(
                f"'{type(self).__name__}' does not support parameter kind(s) for "
                f"parameter(s) {unsupported}. Supported kinds: "
                f"{self.supported_parameter_kinds}."
            )

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        """Construct the kernel, validating parameter kinds before construction."""
        if self.parameter_selector is not None:
            params = [p for p in searchspace.parameters if self.parameter_selector(p)]
        else:
            params = list(searchspace.parameters)
        self._validate_parameter_kinds(params)

        return self._make(searchspace, train_x, train_y)

    @abstractmethod
    def _make(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        """Construct the kernel."""


@define
class _MetaKernelFactory(KernelFactoryProtocol, ABC):
    """Base class for meta kernel factories that orchestrate other kernel factories."""

    @override
    @abstractmethod
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel: ...


@define
class ICMKernelFactory(_MetaKernelFactory):
    """A kernel factory that constructs an ICM kernel for transfer learning.

    ICM: Intrinsic Coregionalization Model :cite:p:`NIPS2007_66368270`
    """

    base_kernel_factory: KernelFactoryProtocol = field(
        alias="base_kernel_or_factory",
        converter=partial(to_component_factory, component_type=GPComponentType.KERNEL),  # type: ignore[misc]
        validator=is_callable(),
    )
    """The factory for the base kernel operating on numerical input features."""

    task_kernel_factory: KernelFactoryProtocol = field(
        alias="task_kernel_or_factory",
        converter=partial(to_component_factory, component_type=GPComponentType.KERNEL),  # type: ignore[misc]
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
        if isinstance(base_kernel, Kernel):
            base_kernel = base_kernel.to_gpytorch(searchspace)
        if isinstance(task_kernel, Kernel):
            task_kernel = task_kernel.to_gpytorch(searchspace)
        return base_kernel * task_kernel
