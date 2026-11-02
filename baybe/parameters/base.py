"""Base classes for all parameters."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cached_property
from itertools import chain
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias, cast, overload

import attrs
import narwhals.stable.v2 as nw
import pandas as pd
from attrs import Converter, define, field
from attrs.converters import optional as optional_c
from attrs.validators import instance_of, min_len
from narwhals.stable.v2.dependencies import is_into_series
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.serialization import (
    SerialMixin,
)
from baybe.settings import active_settings
from baybe.utils.conversion import nonstring_to_tuple
from baybe.utils.metadata import MeasurableMetadata, to_metadata

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    import polars as pl
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from narwhals.stable.v2.typing import IntoDataFrame

    from baybe.kernels.base import BasicKernel
    from baybe.parameters.enum import _ParameterKind
    from baybe.searchspace.continuous import SubspaceContinuous
    from baybe.searchspace.core import SearchSpace
    from baybe.searchspace.discrete import SubspaceDiscrete

    KernelOverride: TypeAlias = Kernel | GPyTorchKernel
else:
    KernelOverride: TypeAlias = Kernel

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164

# Sentinel column name used internally during joins to avoid name conflicts
_JOIN_KEY = "__join_key__"


def _iter_basic_kernels(kernel: Kernel) -> Iterator[BasicKernel]:
    """Iterate over the basic kernel leaves of a BayBE kernel.

    Args:
        kernel: The kernel to traverse.

    Yields:
        The basic kernel leaves.

    Raises:
        TypeError: If the kernel structure is unsupported.
    """
    from baybe.kernels.base import BasicKernel
    from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel

    if isinstance(kernel, BasicKernel):
        yield kernel
    elif isinstance(kernel, ScaleKernel):
        yield from _iter_basic_kernels(kernel.base_kernel)
    elif isinstance(kernel, (AdditiveKernel, ProductKernel)):
        for sub in kernel.base_kernels:
            yield from _iter_basic_kernels(sub)

    else:
        raise TypeError(f"Cannot traverse kernel '{type(kernel).__name__}'.")


def _is_gpytorch_kernel_equivalent(
    kernel: GPyTorchKernel, other: GPyTorchKernel, /
) -> bool:
    """Check if two GPyTorch kernels are equivalent.

    Two kernels are considered equivalent if they have the same module structure,
    the same public module attributes, and agree on the values of all tensors that
    are not changed by fitting. Tensors changed by fitting are only compared by
    shape, since their values merely serve as initialization.

    Args:
        kernel: The first kernel.
        other: The kernel to compare against.

    Returns:
        ``True`` if the kernels are equivalent, ``False`` otherwise.
    """
    modules = list(kernel.named_modules())
    other_modules = list(other.named_modules())
    if [(n, type(m)) for n, m in modules] != [(n, type(m)) for n, m in other_modules]:
        return False
    for (_, module), (_, other_module) in zip(modules, other_modules):
        public = {k: v for k, v in vars(module).items() if not k.startswith("_")}
        other_public = {
            k: v for k, v in vars(other_module).items() if not k.startswith("_")
        }
        if public != other_public:
            return False

    # Tensors requiring gradients are changed by fitting, so only shapes must agree
    tensors = dict(chain(kernel.named_parameters(), kernel.named_buffers()))
    other_tensors = dict(chain(other.named_parameters(), other.named_buffers()))
    if tensors.keys() != other_tensors.keys():
        return False
    for name, tensor in tensors.items():
        other_tensor = other_tensors[name]
        if (tensor.shape, tensor.requires_grad) != (
            other_tensor.shape,
            other_tensor.requires_grad,
        ):
            return False
        if not tensor.requires_grad and not tensor.equal(other_tensor):
            return False
    return True


def _to_kernel_override(value: KernelOverride, instance: Parameter) -> KernelOverride:
    """Validate a kernel override and store BayBE kernels unscoped.

    Args:
        value: The provided kernel override.
        instance: The parameter the override belongs to.

    Raises:
        ValueError: If a BayBE kernel targets a different parameter or a GPyTorch
            kernel specifies explicit active dimensions.
        TypeError: If ``value`` is neither a BayBE nor a GPyTorch kernel.

    Returns:
        The validated override, with BayBE kernels unscoped.
    """
    # BayBE kernels: every basic leaf must be unscoped or scoped to the owner.
    if isinstance(value, Kernel):
        from baybe.kernels.base import BasicKernel

        names_alias = attrs.fields(BasicKernel).parameter_names.alias
        if any(
            leaf.parameter_names not in (None, (instance.name,))
            for leaf in _iter_basic_kernels(value)
        ):
            raise ValueError(
                f"The kernel provided for the kernel override of "
                f"'{instance.__class__.__name__}' may only act on the parameter "
                f"itself. Its basic kernels must specify '{names_alias}' as "
                f"``None`` or ({instance.name!r},)."
            )
        # NOTE: Validated BayBE kernels are stored unscoped and are only scoped to
        #   the owning parameter when accessed via `override_kernel`. This keeps the
        #   stored value independent of the parameter name, so that renaming (e.g.
        #   via `attrs.evolve`) and `is_equivalent` work without rescoping. User
        #   input is still restricted to unscoped kernels or kernels scoped to the
        #   owning parameter.
        return value._scope_to_parameter(None)

    # GPyTorch kernels: no explicit active dimensions allowed anywhere in the tree.
    from baybe.surrogates.gaussian_process.components.generic import (
        GPComponentType,
        _is_gpytorch_component,
    )

    if _is_gpytorch_component(value, (GPComponentType.KERNEL,)):
        if any(k.active_dims is not None for k in (value, *value.sub_kernels())):
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

    _override_kernel: KernelOverride | None = field(
        default=None,
        alias="override_kernel",
        converter=optional_c(Converter(_to_kernel_override, takes_self=True)),  # type: ignore[misc, call-overload]
        kw_only=True,
    )
    """The optional kernel override, exposed via :attr:`override_kernel`."""

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

    @cached_property
    def override_kernel(self) -> KernelOverride | None:
        """An optional kernel replacing the overall kernel for this parameter."""
        if isinstance(kernel := self._override_kernel, Kernel):
            return kernel._scope_to_parameter(self.name)
        return kernel

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

        GPyTorch kernel overrides are compared structurally, since GPyTorch kernels
        define no equality. Values of tensors changed by fitting (i.e., initial
        hyperparameter values) do not affect equivalence.

        Args:
            other: The parameter to compare against.

        Returns:
            ``True`` if the parameters are equivalent, ``False`` otherwise.
        """
        if type(self) is not type(other):
            return False
        # Stored overrides are unscoped (see `_to_kernel_override`), so only the name
        # needs to be aligned for the comparison
        changes: dict[str, Any] = {}
        kernel, other_kernel = self._override_kernel, other._override_kernel
        if (
            kernel is not None
            and not isinstance(kernel, Kernel)
            and other_kernel is not None
            and not isinstance(other_kernel, Kernel)
            and _is_gpytorch_kernel_equivalent(kernel, other_kernel)
        ):
            # GPyTorch kernels define no equality, so substitute equivalent overrides
            # TODO[typing]: attrs resolves all aliases during class creation, see
            #   https://github.com/python-attrs/attrs/issues/1540
            alias = cast(str, attrs.fields(Parameter)._override_kernel.alias)
            changes[alias] = other_kernel
        return attrs.evolve(self, name=other.name, **changes) == other

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

    @property
    @abstractmethod
    def values(self) -> tuple:
        """The values the parameter can take."""

    @property
    def is_finite(self) -> bool:
        """Indicates whether the parameter has a finite number of values."""
        len(self.values)  # <-- raises an error if the parameter is infinite
        return True

    @property
    def active_values(self) -> tuple:
        """The values that are considered for recommendation."""
        return self.values

    @property
    def comp_df(self) -> pd.DataFrame:
        """Deprecated! Use :meth:`transform` instead."""
        warnings.warn(
            f"'{self.__class__.__name__}.comp_df' is deprecated and will be removed "
            f"in a future version. Use '.{self.transform.__name__}()' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            nw.from_native(self.transform(), eager_only=True).to_pandas()
        )

    def to_subspace(self) -> SubspaceDiscrete:
        """Create a one-dimensional search space from the parameter."""
        from baybe.searchspace.discrete import SubspaceDiscrete

        return SubspaceDiscrete.from_parameter(self)

    @override
    def is_in_range(self, item: Any) -> bool:
        return item in self.values

    @overload
    def transform(self, series: None = None, /) -> IntoDataFrame: ...
    @overload
    def transform(self, series: pd.Series[Any], /) -> pd.DataFrame: ...
    @overload
    def transform(self, series: pl.Series, /) -> pl.DataFrame: ...
    @overload
    def transform(self, series: Iterable[Any], /) -> IntoDataFrame: ...

    def transform(
        self, series: nw.IntoSeries | Iterable[Any] | None = None, /
    ) -> IntoDataFrame:
        """Transform parameter values to computational representation.

        Args:
            series: The parameter values in experimental representation to be
                transformed. If ``None``, the full computational representation
                for all parameter values is returned.

        Returns:
            A native eager frame containing the transformed values, in the same
            backend as the input series.

        Raises:
            ValueError: If the series name does not match the parameter name.
        """
        all_values = series is None
        if is_into_series(series):
            # TODO[typing]: https://github.com/narwhals-dev/narwhals/issues/4064
            series = cast(
                "nw.Series[Any]",
                nw.from_native(series, series_only=True),  # pyrefly: ignore[no-matching-overload]
            )
            if series.name != self.name:
                raise ValueError(
                    f"The provided series name '{series.name}' does not match "
                    f"parameter name '{self.name}'."
                )
        else:
            series = nw.new_series(
                name=self.name,
                values=self.values if all_values else series,
                backend=active_settings.default_dataframe_backend,
            )

        from baybe.utils.dataframe import _copy_index

        table = self._encoding_table(series.unique())
        out = (
            series.rename(_JOIN_KEY)
            .to_frame()
            .join(table, on=_JOIN_KEY, how="left")
            .drop(_JOIN_KEY)
        )
        return out.to_native() if all_values else _copy_index(out, series).to_native()

    @abstractmethod
    def _encoding_table(self, values: nw.Series, /) -> nw.DataFrame:
        """Create the encoding table for the given unique parameter values.

        The returned dataframe must use :data:`_JOIN_KEY` as the key column (holding the
        experimental values) plus one column per entry in :attr:`comp_rep_columns`.

        Args:
            values: The unique experimental values to encode.

        Returns:
            A dataframe mapping parameter values to their encoded representation.
        """

    @override
    def summary(self) -> dict:
        return dict(
            Name=self.name,
            Type=self.__class__.__name__,
            nValues=len(self.values),
        )


@define(frozen=True, slots=False)
class _EncodedDiscreteParameter(DiscreteParameter, ABC):
    """Abstract class for encoded discrete parameters.

    In general, these are parameters with non-numerical experimental representations.
    """

    # class variables
    is_numerical: ClassVar[bool] = False
    # See base class.

    # object variables
    _active_values: tuple[str | bool, ...] | None = field(
        default=None,
        converter=optional_c(
            Converter(  # type: ignore[misc, call-overload]
                nonstring_to_tuple, takes_self=True, takes_field=True
            )
        ),
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
