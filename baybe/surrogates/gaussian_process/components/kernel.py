"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define, field, fields
from attrs.converters import optional
from attrs.validators import is_callable
from typing_extensions import override

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.kernels.base import Kernel
from baybe.objectives.base import Objective
from baybe.parameters.categorical import TaskParameter
from baybe.parameters.enum import _ParameterKind
from baybe.parameters.fidelity import (
    CategoricalFidelityParameter,
    NumericalDiscreteFidelityParameter,
)
from baybe.parameters.selectors import (
    ParameterSelectorProtocol,
    TypeSelector,
    to_parameter_selector,
)
from baybe.searchspace.core import SearchSpace, SearchSpaceFidelityType
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.gaussian_process.components.generic import (
    GPComponentFactoryProtocol,
    GPComponentType,
    PlainGPComponentFactory,
    to_component_factory,
)

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.parameters.base import Parameter

    KernelFactoryProtocol = GPComponentFactoryProtocol[Kernel | GPyTorchKernel]
    PlainKernelFactory = PlainGPComponentFactory[Kernel | GPyTorchKernel]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    KernelFactoryProtocol = GPComponentFactoryProtocol[Kernel]
    """A factory protocol for Gaussian process kernels."""

    PlainKernelFactory = PlainGPComponentFactory[Kernel]
    """A trivial factory returning a fixed, pre-defined kernel."""


@define
class _PureKernelFactory(KernelFactoryProtocol, SerialMixin, ABC):
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
        return searchspace._get_n_comp_rep_columns(
            self.parameter_selector or (lambda _: True)
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
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        """Construct the kernel, validating parameter kinds before construction."""
        if self.parameter_selector is not None:
            params = [p for p in searchspace.parameters if self.parameter_selector(p)]
        else:
            params = list(searchspace.parameters)
        self._validate_parameter_kinds(params)

        return self._make(searchspace, objective, measurements)

    @abstractmethod
    def _make(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        """Construct the kernel."""


def _enable_kernel_composition(
    cls: type[_PureKernelFactory], name: str | None = None, /
) -> type[_PureKernelFactory]:
    """Class decorator enabling automatic kernel composition for non-design parameters.

    The decorated factory produces the base kernel for design parameters. Depending
    on the search space, the decorator composes it with the appropriate meta-kernel:

    * **Task / categorical fidelity**: Composed with an ``ICMKernelFactory``
      (``IndexKernel`` × base kernel). Used for transfer learning
      (``TaskParameter``) and categorical multi-fidelity
      (``CategoricalFidelityParameter``).
    * **Numerical discrete fidelity**: The base kernel is not used. Instead, a
      ``DownsamplingKernelFactory`` builds the full kernel (replicating BoTorch's
      ``SingleTaskMultiFidelityGP`` structure).
    * **No task or fidelity parameters**: The base kernel is returned unchanged.

    When used as a decorator (without ``name``), the class is modified in-place.
    When called with a ``name`` argument, a new subclass is created so that the
    original class remains unmodified. The latter form is intended for cases where
    the original class is reused independently elsewhere.

    Args:
        cls: The kernel factory class to decorate.
        name: Optional name for the created class. If provided, a new subclass is
            created instead of modifying ``cls`` in-place.

    Raises:
        TypeError: If the factory already supports task or fidelity parameters.

    Returns:
        The decorated kernel factory class with kernel composition enabled.
    """
    _extended_kinds = _ParameterKind.TASK | _ParameterKind.FIDELITY
    if cls._supported_parameter_kinds & _extended_kinds:
        raise TypeError(
            f"'{cls.__name__}' already supports task or fidelity parameters."
        )

    # This distinction is important for serialization so that the classes can be
    # correctly identified by their names in the subclass registry
    if name is None:
        # Modify the class in-place (avoids name collision in subclass registry)
        # -> For the use with `@` syntax, where the original class gets overridden by
        #    the decorated version, i.e., no references to the original class remain.
        target_cls = cls
    else:
        # Create a sibling class so the original class remains unmodified.
        # We use cls.__bases__ (not (cls,)) because the new class is conceptually
        # an equivalent variant, not a specialization. Concrete (non-dunder)
        # attributes are copied so the sibling has the same behavior.
        # __module__ must be set explicitly because the Protocol metaclass
        # would otherwise default it to "abc".
        # -> For the assignment-based use, i.e.,
        #   `DecoratedX = _enable_kernel_composition(X, name="DecoratedX")`,
        #    where both the original and decorated versions remain accessible and
        #    are intended to be used independently.
        ns = {
            k: v
            for k, v in cls.__dict__.items()
            if not (k.startswith("__") and k.endswith("__"))
        }
        ns["__doc__"] = cls.__doc__
        ns["__module__"] = cls.__module__
        target_cls = type(name, cls.__bases__, ns)

    original_call = cls.__call__
    original_supported_kinds = cls._supported_parameter_kinds
    _index_exclude_selector = TypeSelector(
        (
            TaskParameter,
            CategoricalFidelityParameter,
            NumericalDiscreteFidelityParameter,
        ),
        exclude=True,
    )

    @functools.wraps(original_call)
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ):
        # Numerical fidelity: the full kernel is built by DownsamplingKernelFactory
        # (replicates BoTorch's SingleTaskMultiFidelityGP kernel structure).
        # The base kernel from this factory is not used in this path.
        if (
            searchspace.fidelity_type
            == SearchSpaceFidelityType.NUMERICALDISCRETEMULTIFIDELITY
        ):
            return DownsamplingKernelFactory()(searchspace, objective, measurements)

        # Compute the base kernel, excluding task/fidelity parameters from scope
        broadened_kinds = target_cls._supported_parameter_kinds
        target_cls._supported_parameter_kinds = original_supported_kinds

        original_selector = self.parameter_selector
        if original_selector is None:
            self.parameter_selector = _index_exclude_selector
        else:
            self.parameter_selector = lambda p: (
                _index_exclude_selector(p) and original_selector(p)
            )

        try:
            base_kernel = original_call(self, searchspace, objective, measurements)
        finally:
            target_cls._supported_parameter_kinds = broadened_kinds
            self.parameter_selector = original_selector

        # Task / categorical fidelity: compose base kernel with IndexKernel
        if searchspace.task_idx is not None or (
            searchspace.fidelity_type
            == SearchSpaceFidelityType.CATEGORICALMULTIFIDELITY
        ):
            icm = ICMKernelFactory(base_kernel_or_factory=base_kernel)
            return icm(searchspace, objective, measurements)

        return base_kernel

    target_cls.__call__ = __call__  # type: ignore[method-assign]
    target_cls._supported_parameter_kinds = (
        cls._supported_parameter_kinds | _extended_kinds
    )
    return target_cls


@define
class _MetaKernelFactory(KernelFactoryProtocol, ABC):
    """Base class for meta kernel factories that orchestrate other kernel factories."""

    @override
    @abstractmethod
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
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

        return _BayBENumericalKernelFactory(
            TypeSelector(
                (TaskParameter, CategoricalFidelityParameter),
                exclude=True,
            )
        )

    @task_kernel_factory.default
    def _default_task_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            _BayBEIndexKernelFactory,
        )

        return _BayBEIndexKernelFactory(
            TypeSelector((TaskParameter, CategoricalFidelityParameter))
        )

    @base_kernel_factory.validator
    def _validate_base_kernel_factory(self, _, factory: KernelFactoryProtocol):
        _index_kinds = _ParameterKind.TASK | _ParameterKind.FIDELITY
        if (
            isinstance(factory, _PureKernelFactory)
            and factory._supported_parameter_kinds & _index_kinds
        ):
            raise TypeError(
                f"The specified '{fields(ICMKernelFactory).base_kernel_factory.alias}' "
                f"must not support task or fidelity parameters."
            )

    @task_kernel_factory.validator
    def _validate_task_kernel_factory(self, _, factory: KernelFactoryProtocol):
        if (
            isinstance(factory, _PureKernelFactory)
            and factory._supported_parameter_kinds & _ParameterKind.REGULAR
        ):
            raise TypeError(
                f"The specified '{fields(ICMKernelFactory).task_kernel_factory.alias}' "
                f"must support only task or fidelity parameters."
            )

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        if searchspace.task_idx is None and (
            searchspace.fidelity_type
            != SearchSpaceFidelityType.CATEGORICALMULTIFIDELITY
        ):
            raise IncompatibleSearchSpaceError(
                f"'{type(self).__name__}' can only be used with a searchspace that "
                f"contains a '{TaskParameter.__name__}' or a "
                f"'{CategoricalFidelityParameter.__name__}'."
            )

        base_kernel = self.base_kernel_factory(searchspace, objective, measurements)
        index_kernel = self.task_kernel_factory(searchspace, objective, measurements)
        if isinstance(base_kernel, Kernel):
            base_kernel = base_kernel.to_gpytorch(searchspace)
        if isinstance(index_kernel, Kernel):
            index_kernel = index_kernel.to_gpytorch(searchspace)

        # Ensure correct partitioning between base and index kernels active dimensions.
        # The index dimension is either the task or the categorical fidelity column —
        # exactly one is non-None after the guard at the start of this method.
        index_idx = (
            searchspace.task_idx
            if searchspace.task_idx is not None
            else searchspace.fidelity_idx
        )
        all_idcs = set(range(len(searchspace.comp_rep_columns)))
        allowed_index_idcs = {index_idx}
        allowed_base_idcs = all_idcs - allowed_index_idcs
        base_idcs = (
            set(d.tolist()) if (d := base_kernel.active_dims) is not None else all_idcs
        )
        index_idcs = (
            set(d.tolist()) if (d := index_kernel.active_dims) is not None else all_idcs
        )

        if not base_idcs <= allowed_base_idcs:
            raise ValueError(
                f"The base kernel's 'active_dims' {base_idcs} must be a subset of "
                f"the non-index indices {allowed_base_idcs}."
            )
        if index_idcs != allowed_index_idcs:
            raise ValueError(
                f"The index kernel's 'active_dims' {index_idcs} does not match "
                f"the index column {allowed_index_idcs}."
            )

        return base_kernel * index_kernel


@define
class DownsamplingKernelFactory(_MetaKernelFactory):
    """A kernel factory for numerical fidelity via DownsamplingKernel composition.

    Replicates BoTorch's ``SingleTaskMultiFidelityGP`` kernel structure:
    ``ProductKernel(base_kernel, DownsamplingKernel)``.
    """

    base_kernel_factory: KernelFactoryProtocol = field(
        alias="base_kernel_or_factory",
        converter=partial(to_component_factory, component_type=GPComponentType.KERNEL),  # type: ignore[misc]
        validator=is_callable(),
    )
    """The factory for the base kernel operating on non-fidelity input features."""

    @base_kernel_factory.default
    def _default_base_kernel_factory(self) -> KernelFactoryProtocol:
        from baybe.surrogates.gaussian_process.presets.baybe import (
            _BayBEDownsamplingBaseKernelFactory,
        )

        return _BayBEDownsamplingBaseKernelFactory(
            TypeSelector((NumericalDiscreteFidelityParameter,), exclude=True)
        )

    @override
    def __call__(
        self, searchspace: SearchSpace, objective: Objective, measurements: pd.DataFrame
    ) -> Kernel | GPyTorchKernel:
        """Construct the fidelity kernel for numerical fidelity spaces."""
        from botorch.models.kernels.downsampling import DownsamplingKernel
        from gpytorch.kernels import ProductKernel
        from gpytorch.priors import GammaPrior

        if (
            searchspace.fidelity_type
            != SearchSpaceFidelityType.NUMERICALDISCRETEMULTIFIDELITY
        ):
            raise IncompatibleSearchSpaceError(
                f"'{type(self).__name__}' can only be used with a search space that "
                f"contains a '{NumericalDiscreteFidelityParameter.__name__}'."
            )

        base_kernel = self.base_kernel_factory(searchspace, objective, measurements)
        if isinstance(base_kernel, Kernel):
            base_kernel = base_kernel.to_gpytorch(searchspace)

        fidelity_idx = searchspace.fidelity_idx
        downsampling = DownsamplingKernel(
            offset_prior=GammaPrior(3.0, 6.0),
            power_prior=GammaPrior(3.0, 6.0),
            active_dims=[fidelity_idx],
        )

        return ProductKernel(base_kernel, downsampling)
