"""Base classes for all parameters."""

from __future__ import annotations

import gc
import sys
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import attrs
import pandas as pd
from attrs import Converter, define, field
from attrs.converters import optional as optional_c
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.parameters.enum import ParameterEncoding
from baybe.serialization import (
    SerialMixin,
)
from baybe.utils.basic import to_tuple
from baybe.utils.metadata import MeasurableMetadata, to_metadata

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.parameters.enum import _ParameterKind
    from baybe.searchspace.continuous import SubspaceContinuous
    from baybe.searchspace.core import SearchSpace
    from baybe.searchspace.discrete import SubspaceDiscrete

    KernelOverride: TypeAlias = Kernel | GPyTorchKernel
else:
    KernelOverride: TypeAlias = Kernel

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164


def _iter_basic_kernels(kernel: Kernel):
    """Iterate over the basic kernel leaves of a BayBE kernel."""
    from baybe.kernels.base import BasicKernel
    from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel

    if isinstance(kernel, BasicKernel):
        yield kernel
    elif isinstance(kernel, ScaleKernel):
        yield from _iter_basic_kernels(kernel.base_kernel)
    elif isinstance(kernel, (AdditiveKernel, ProductKernel)):
        for sub in kernel.base_kernels:
            yield from _iter_basic_kernels(sub)


def _to_kernel_override(
    value: KernelOverride | None, instance: Parameter
) -> KernelOverride | None:
    """Validate a kernel override and scope BayBE kernels to their parameter.

    Args:
        value: The provided kernel override.
        instance: The parameter the override belongs to.

    Raises:
        ValueError: If a BayBE kernel targets a different parameter or a GPyTorch
            kernel specifies explicit active dimensions.
        TypeError: If the object is neither a BayBE nor a GPyTorch kernel.

    Returns:
        The validated override, with BayBE kernels scoped to the parameter.
    """
    if value is None:
        return None

    # BayBE kernels: every basic leaf must be unscoped or scoped to the owner. The
    # kernel is then rebound to the owning parameter (dropping unspecified names).
    if isinstance(value, Kernel):
        if any(
            leaf.parameter_names not in (None, (instance.name,))
            for leaf in _iter_basic_kernels(value)
        ):
            raise ValueError(
                f"The kernel provided for the kernel override of "
                f"'{instance.__class__.__name__}' may only act on the parameter "
                f"itself. Its basic kernels must specify 'parameter_names' as "
                f"``None`` or ({instance.name!r},)."
            )
        return value._with_parameter(instance.name)

    # GPyTorch kernels: no explicit active dimensions allowed anywhere in the tree.
    if sys.modules.get("gpytorch") is not None:
        from gpytorch.kernels import Kernel as GPyTorchKernel

        if isinstance(value, GPyTorchKernel):
            if any(
                k.active_dims is not None
                for k in value.modules()
                if isinstance(k, GPyTorchKernel)
            ):
                raise ValueError(
                    "The GPyTorch kernel provided for the kernel override must not "
                    "specify 'active_dims'."
                )
            return value

    raise TypeError(
        f"The object provided for the kernel override of "
        f"'{instance.__class__.__name__}' must be a BayBE or GPyTorch kernel. "
        f"Got: {type(value)}"
    )


@define(frozen=True, slots=False)
class Parameter(ABC, SerialMixin):
    """Abstract base class for all parameters.

    Stores information about the type, range, constraints, etc. and handles in-range
    checks, transformations etc.
    """

    # class variables
    is_numerical: ClassVar[bool]
    """Class variable encoding whether this parameter is numeric."""

    # object variables
    name: str = field(validator=(instance_of(str), min_len(1)))
    """The name of the parameter"""

    kernel_override: KernelOverride | None = field(
        default=None,
        converter=Converter(_to_kernel_override, takes_self=True),  # type: ignore[misc, call-overload]
        kw_only=True,
    )
    """An optional kernel replacing the overall kernel for this parameter."""

    metadata: MeasurableMetadata = field(
        factory=MeasurableMetadata,
        converter=lambda x: to_metadata(x, MeasurableMetadata),
        kw_only=True,
    )
    """Optional metadata containing description, unit, and other information."""

    @abstractmethod
    def is_in_range(self, item: Any) -> bool:
        """Return whether an item is within the parameter range.

        Args:
            item: The item to be checked.

        Returns:
            ``True`` if the item is within the parameter range, ``False`` otherwise.
        """

    @override
    def __str__(self) -> str:
        return str(self.summary())

    @property
    def is_continuous(self) -> bool:
        """Boolean indicating if this is a continuous parameter."""
        return isinstance(self, ContinuousParameter)

    @property
    def is_discrete(self) -> bool:
        """Boolean indicating if this is a discrete parameter."""
        return isinstance(self, DiscreteParameter)

    @property
    def _kind(self) -> _ParameterKind:
        """The kind of the parameter."""
        from baybe.parameters.enum import _ParameterKind

        return _ParameterKind.from_parameter(self)

    @property
    @abstractmethod
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""

    def to_searchspace(self) -> SearchSpace:
        """Create a one-dimensional search space from the parameter."""
        from baybe.searchspace.core import SearchSpace

        return SearchSpace.from_parameter(self)

    def is_equivalent(self, other: Parameter) -> bool:
        """Check if this parameter is equivalent to another, ignoring the name.

        Two parameters are considered equivalent if they have the same type and
        all attributes are equal except for the name.

        Args:
            other: The parameter to compare against.

        Returns:
            ``True`` if the parameters are equivalent, ``False`` otherwise.
        """
        if type(self) is not type(other):
            return False
        # The override is owner-scoped, so rebind it to the other parameter's name.
        kernel_override = self.kernel_override
        if isinstance(kernel_override, Kernel):
            kernel_override = kernel_override._with_parameter(other.name)
        return (
            attrs.evolve(self, name=other.name, kernel_override=kernel_override)
            == other
        )

    @abstractmethod
    def summary(self) -> dict:
        """Return a custom summarization of the parameter."""

    @property
    def description(self) -> str | None:
        """The description of the parameter."""
        return self.metadata.description

    @property
    def unit(self) -> str | None:
        """The unit of measurement for the parameter."""
        return self.metadata.unit


@define(frozen=True, slots=False)
class DiscreteParameter(Parameter, ABC):
    """Abstract class for discrete parameters."""

    # class variables
    encoding: ParameterEncoding | None = field(init=False, default=None)
    """An optional encoding for the parameter."""

    @property
    @abstractmethod
    def values(self) -> tuple:
        """The values the parameter can take."""

    @property
    def active_values(self) -> tuple:
        """The values that are considered for recommendation."""
        return self.values

    @cached_property
    @abstractmethod
    def comp_df(self) -> pd.DataFrame:
        # TODO: Should be renamed to `comp_rep`
        """Return the computational representation of the parameter."""

    @override
    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        return tuple(self.comp_df.columns)

    def to_subspace(self) -> SubspaceDiscrete:
        """Create a one-dimensional search space from the parameter."""
        from baybe.searchspace.discrete import SubspaceDiscrete

        return SubspaceDiscrete.from_parameter(self)

    @override
    def is_in_range(self, item: Any) -> bool:
        return item in self.values

    def transform(self, series: pd.Series, /) -> pd.DataFrame:
        """Transform parameter values to computational representation.

        Args:
            series: The parameter values in experimental representation to be
                transformed.

        Returns:
            A series containing the transformed values. The series name matches
            that of the input.
        """
        if self.encoding:
            # replace each label with the corresponding encoding
            transformed = pd.merge(
                left=series.rename("Labels").to_frame(),
                left_on="Labels",
                right=self.comp_df,
                right_index=True,
                how="left",
            ).drop(columns="Labels")
        else:
            transformed = series.to_frame()

        return transformed

    @override
    def summary(self) -> dict:
        param_dict = dict(
            Name=self.name,
            Type=self.__class__.__name__,
            nValues=len(self.values),
            Encoding=self.encoding,
        )
        return param_dict


@define(frozen=True, slots=False)
class _DiscreteLabelLikeParameter(DiscreteParameter, ABC):
    """Abstract class for discrete label-like parameters.

    In general, these are parameters with non-numerical experimental representations.
    """

    # class variables
    is_numerical: ClassVar[bool] = False
    # See base class.

    # object variables
    _active_values: tuple[str | bool, ...] | None = field(
        default=None,
        converter=optional_c(to_tuple),
        kw_only=True,
        alias="active_values",
    )
    """Optional labels identifying the ones which should be actively recommended."""

    @override
    @property
    def active_values(self) -> tuple[str | bool, ...]:
        if self._active_values is None:
            return self.values

        return self._active_values

    @_active_values.validator
    def _validate_active_values(  # noqa: DOC101, DOC103
        self, _: Any, content: tuple[str | bool, ...]
    ) -> None:
        """Validate the active parameter values.

        If no such list is provided, no validation is being performed. In particular,
        the errors listed below are only relevant if the ``values`` list is provided.

        Raises:
            ValueError: If an empty active parameters list is provided.
            ValueError: If the active parameter values are not unique.
            ValueError: If not all active values are valid parameter choices.
        """
        if content is None:
            return

        if len(content) == 0:
            raise ValueError(
                "If an active parameters list is provided, it must not be empty."
            )
        if len(set(content)) != len(content):
            raise ValueError("The active parameter values must be unique.")
        if not all(v in self.values for v in content):
            raise ValueError(
                f"All active values must be valid parameter choices from: "
                f"{self.values}, provided: {content}"
            )

    @override
    def summary(self) -> dict:
        return {**super().summary(), "nActiveValues": len(self.active_values)}


@define(frozen=True, slots=False)
class ContinuousParameter(Parameter):
    """Abstract class for continuous parameters."""

    def to_subspace(self) -> SubspaceContinuous:
        """Create a one-dimensional search space from the parameter."""
        from baybe.searchspace.continuous import SubspaceContinuous

        return SubspaceContinuous.from_parameter(self)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
