"""Base classes for all parameters."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len
from cattrs.gen import override

from baybe.parameters.enum import ParameterEncoding
from baybe.serialization import (
    SerialMixin,
    converter,
    get_base_structure_hook,
    unstructure_base,
)

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

    @abstractmethod
    def is_in_range(self, item: Any) -> bool:
        """Return whether an item is within the parameter range.

        Args:
            item: The item to be checked.

        Returns:
            ``True`` if the item is within the parameter range, ``False`` otherwise.
        """

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


@define(frozen=True, slots=False)
class DiscreteParameter(Parameter, ABC):
    """Abstract class for discrete parameters."""

    # TODO [15280]: needs to be refactored

    # class variables
    encoding: ParameterEncoding | None = field(init=False, default=None)
    """An optional encoding for the parameter."""

    @property
    @abstractmethod
    def values(self) -> tuple:
        """The values the parameter can take."""

    @cached_property
    @abstractmethod
    def comp_df(self) -> pd.DataFrame:
        # TODO: Should be renamed to `comp_rep`
        """Return the computational representation of the parameter."""

    @property
    def comp_rep_columns(self) -> tuple[str, ...]:  # noqa: D102
        # See base class.
        return tuple(self.comp_df.columns)

    def to_subspace(self) -> SubspaceDiscrete:
        """Create a one-dimensional search space from the parameter."""
        from baybe.searchspace.discrete import SubspaceDiscrete

        return SubspaceDiscrete.from_parameter(self)

    def is_in_range(self, item: Any) -> bool:  # noqa: D102
        # See base class.
        return item in self.values

    def transform(self, series: pd.Series, /) -> pd.DataFrame:
        """Transform parameter values from experimental to computational representation.

        Args:
            series: The parameter values to be transformed.

        Returns:
            The transformed parameter values.
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

    def summary(self) -> dict:  # noqa: D102
        # See base class.
        param_dict = dict(
            Name=self.name,
            Type=self.__class__.__name__,
            Num_Values=len(self.values),
            Encoding=self.encoding,
        )
        return param_dict


@define(frozen=True, slots=False)
class ContinuousParameter(Parameter):
    """Abstract class for continuous parameters."""

    def to_subspace(self) -> SubspaceContinuous:
        """Create a one-dimensional search space from the parameter."""
        from baybe.searchspace.continuous import SubspaceContinuous

        return SubspaceContinuous.from_parameter(self)


# Register (un-)structure hooks
_overrides = {"_values": override(rename="values")}
# FIXME[typing]: https://github.com/python/mypy/issues/4717
converter.register_structure_hook(
    Parameter,
    get_base_structure_hook(Parameter, overrides=_overrides),  # type: ignore
)
converter.register_unstructure_hook(
    Parameter, partial(unstructure_base, overrides=_overrides)
)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
