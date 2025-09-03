"""Base classes for all parameters."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import instance_of, min_len
from typing_extensions import override

from baybe.parameters.enum import ParameterEncoding
from baybe.serialization import (
    SerialMixin,
)
from baybe.utils.basic import to_tuple
from baybe.utils.metadata import MeasurableMetadata, to_metadata

if TYPE_CHECKING:
    from baybe.searchspace.continuous import SubspaceContinuous
    from baybe.searchspace.core import SearchSpace
    from baybe.searchspace.discrete import SubspaceDiscrete

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164


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
