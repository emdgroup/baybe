"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import functools
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
from baybe.parameters.enum import _ParameterKind
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

    _supported_parameter_kinds: ClassVar[_ParameterKind] = _ParameterKind.REGULAR
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

    def get_parameter_names(self, searchspace: SearchSpace) -> tuple[str, ...]:
        """Get the names of the parameters to be considered by the kernel."""
        selector = self.parameter_selector or (lambda _: True)
        return tuple(p.name for p in searchspace.parameters if selector(p))

    def _get_effective_dimensionality(self, searchspace: SearchSpace) -> int:
        """Get the number of computational columns for the selected parameters."""
        names = self.get_parameter_names(searchspace)
        if names is None:
            return len(searchspace.comp_rep_columns)
        return sum(
            len(searchspace.get_comp_rep_parameter_indices(name)) for name in names
        )

    def _validate_parameter_kinds(self, parameters: Iterable[Parameter]) -> None:
        """Validate that the given parameters are supported by the factory.

        Args:
            parameters: The parameters to validate.

        Raises:
            IncompatibleSearchSpaceError: If unsupported parameter kinds are found.
        """
        if unsupported := [
            p.name
            for p in parameters
            if not (p._kind & self._supported_parameter_kinds)
        ]:
            raise IncompatibleSearchSpaceError(
                f"'{type(self).__name__}' does not support parameter kind(s) for "
                f"parameter(s) {unsupported}. Supported kinds: "
                f"{self._supported_parameter_kinds}."
            )

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel | GPyTorchKernel:
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
    ) -> Kernel | GPyTorchKernel:
        """Construct the kernel."""


def _enable_transfer_learning(
    cls: type[_PureKernelFactory], name: str | None = None, /
) -> type[_PureKernelFactory]:
    """Class decorator enabling BayBE's default transfer learning mechanism.

    When the search space contains a task parameter, the decorated factory
    automatically composes its kernel with BayBE's default task kernel.
    Otherwise, the factory behaves unchanged.

    When used as a decorator (without ``name``), the class is modified in-place.
    When called with a ``name`` argument, a new subclass is created so that the
    original class remains unmodified. The latter form is intended for cases where
    the original class is reused independently elsewhere.

    Args:
        cls: The kernel factory class to decorate.
        name: Optional name for the created class. If provided, a new subclass is
            created instead of modifying ``cls`` in-place.

    Raises:
        TypeError: If the factory already supports task parameters.

    Returns:
        The decorated kernel factory class with transfer learning enabled.
    """
    if cls._supported_parameter_kinds & _ParameterKind.TASK:
        raise TypeError(f"'{cls.__name__}' already supports task parameters.")

    # This distinction is important for serialization so that the classes can be
    # correctly identified by their names in the subclass registry
    if name is not None:
        # Create a subclass so the original class remains unmodified
        target_cls = type(name, (cls,), {"__doc__": cls.__doc__})
    else:
        # Modify the class in-place (avoids name collision in subclass registry)
        target_cls = cls

    original_call = cls.__call__
    original_supported_kinds = cls._supported_parameter_kinds
    _task_exclude_selector = TypeSelector((TaskParameter,), exclude=True)

    @functools.wraps(original_call)
    def __call__(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor):
        # Temporarily narrow the supported parameter kinds to those of the original
        # class. If the decorator logic is correct, the original factory should never
        # see the extended scope, but this acts as a sanity check to prevent regressions
        broadened_kinds = target_cls._supported_parameter_kinds
        target_cls._supported_parameter_kinds = original_supported_kinds

        # Split off the task parameters
        original_selector = self.parameter_selector
        if original_selector is None:
            self.parameter_selector = _task_exclude_selector
        else:
            self.parameter_selector = lambda p: (
                _task_exclude_selector(p) and original_selector(p)
            )
        try:
            base_kernel = original_call(self, searchspace, train_x, train_y)
        finally:
            target_cls._supported_parameter_kinds = broadened_kinds
            self.parameter_selector = original_selector

        if searchspace.task_idx is not None:
            icm = ICMKernelFactory(base_kernel_or_factory=base_kernel)
            return icm(searchspace, train_x, train_y)
        return base_kernel

    target_cls.__call__ = __call__  # type: ignore[method-assign]
    target_cls._supported_parameter_kinds = (
        cls._supported_parameter_kinds | _ParameterKind.TASK
    )
    return target_cls


@define
class _MetaKernelFactory(KernelFactoryProtocol, ABC):
    """Base class for meta kernel factories that orchestrate other kernel factories."""

    @override
    @abstractmethod
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel | GPyTorchKernel: ...


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
            _BayBENumericalKernelFactory,
        )

        assert (
            _BayBENumericalKernelFactory._supported_parameter_kinds
            is _ParameterKind.REGULAR
        )
        return _BayBENumericalKernelFactory(
            TypeSelector((TaskParameter,), exclude=True)
        )

    @task_kernel_factory.default
    def _default_task_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            _BayBETaskKernelFactory,
        )

        assert _BayBETaskKernelFactory._supported_parameter_kinds is _ParameterKind.TASK
        return _BayBETaskKernelFactory()

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel | GPyTorchKernel:
        if searchspace.task_idx is None:
            raise IncompatibleSearchSpaceError(
                f"'{type(self).__name__}' can only be used with a searchspace that "
                f"contains a '{TaskParameter.__name__}'."
            )

        base_kernel = self.base_kernel_factory(searchspace, train_x, train_y)
        task_kernel = self.task_kernel_factory(searchspace, train_x, train_y)
        if isinstance(base_kernel, Kernel):
            base_kernel = base_kernel.to_gpytorch(searchspace)
        if isinstance(task_kernel, Kernel):
            task_kernel = task_kernel.to_gpytorch(searchspace)

        # Ensure correct partitioning between base and task kernels active dimensions
        all_idcs = set(range(len(searchspace.comp_rep_columns)))
        allowed_task_idcs = {searchspace.task_idx}
        allowed_base_idcs = all_idcs - allowed_task_idcs
        base_idcs = (
            set(d.tolist()) if (d := base_kernel.active_dims) is not None else all_idcs
        )
        task_idcs = (
            set(d.tolist()) if (d := task_kernel.active_dims) is not None else all_idcs
        )

        if not base_idcs <= allowed_base_idcs:
            raise ValueError(
                f"The base kernel's 'active_dims' {base_idcs} must be a subset of "
                f"the non-task indices {allowed_base_idcs}."
            )
        if task_idcs != allowed_task_idcs:
            raise ValueError(
                f"The task kernel's 'active_dims' {task_idcs} does not match "
                f"the task index {allowed_task_idcs}."
            )

        return base_kernel * task_kernel
