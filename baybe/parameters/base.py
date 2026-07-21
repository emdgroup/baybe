"""Base classes for all parameters."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, overload

import narwhals.stable.v2 as nw
import pandas as pd
from attr.converters import optional as optional_c
from attrs import Converter, define, field
from attrs.validators import instance_of, min_len
from narwhals.stable.v2.dependencies import is_into_series
from typing_extensions import override

from baybe.serialization import (
    SerialMixin,
)
from baybe.settings import active_settings
from baybe.utils.conversion import nonstring_to_tuple
from baybe.utils.metadata import MeasurableMetadata, to_metadata

if TYPE_CHECKING:
    from typing import Any

    import polars as pl
    from narwhals.stable.v2.typing import IntoDataFrame

    from baybe.parameters.enum import _ParameterKind
    from baybe.searchspace.continuous import SubspaceContinuous
    from baybe.searchspace.core import SearchSpace
    from baybe.searchspace.discrete import SubspaceDiscrete

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164

# Sentinel column name used internally during joins to avoid name conflicts
_JOIN_KEY = "__join_key__"


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
    def transform(self, series: Iterable[Any], /) -> IntoDataFrame: ...  # type: ignore[overload-cannot-match]

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
            series = nw.from_native(series, series_only=True)
            if series.name != self.name:
                raise ValueError(
                    f"The provided series name '{series.name}' does not match "
                    f"parameter name '{self.name}'."
                )
        else:
            # TODO[typing]: https://github.com/narwhals-dev/narwhals/issues/3808
            series = nw.new_series(
                name=self.name,
                values=self.values if all_values else series,
                backend=active_settings.default_dataframe_backend,  # type: ignore[arg-type]
            )

        table = self._encoding_table(series.unique())
        result = (
            series.rename(_JOIN_KEY)
            .to_frame()
            .join(table, on=_JOIN_KEY, how="left")
            .drop(_JOIN_KEY)
        )

        # TODO[narwhalify]: drop once pandas index handling is removed globally
        if nw.get_native_namespace(series) is pd:
            native = result.to_native()
            native.index = series.to_list() if all_values else series.to_pandas().index
            return native

        return result.to_native()

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
