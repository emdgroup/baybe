"""Base classes for all parameters."""

from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import Any, ClassVar, Optional

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

# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164


@define(frozen=True, slots=False)
class Parameter(ABC, SerialMixin):
    """Abstract base class for all parameters.

    Stores information about the type, range, constraints, etc. and handles in-range
    checks, transformations etc.
    """

    # class variables
    is_numeric: ClassVar[bool]
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

    @abstractmethod
    def summary(self) -> dict:
        """Return a custom summarization of the parameter."""


@define(frozen=True, slots=False)
class DiscreteParameter(Parameter, ABC):
    """Abstract class for discrete parameters."""

    # TODO [15280]: needs to be refactored

    # class variables
    encoding: Optional[ParameterEncoding] = field(init=False, default=None)
    """An optional encoding for the parameter."""

    @property
    @abstractmethod
    def values(self) -> tuple:
        """The values the parameter can take."""

    @cached_property
    @abstractmethod
    def comp_df(self) -> pd.DataFrame:
        """Return the computational representation of the parameter."""

    def is_in_range(self, item: Any) -> bool:  # noqa: D102
        # See base class.
        return item in self.values

    def transform_rep_exp2comp(self, data: pd.Series) -> pd.DataFrame:
        """Transform data from experimental to computational representation.

        Args:
            data: Data to be transformed.

        Returns:
            The transformed version of the data.
        """
        if self.encoding:
            # replace each label with the corresponding encoding
            transformed = pd.merge(
                left=data.rename("Labels").to_frame(),
                left_on="Labels",
                right=self.comp_df,
                right_index=True,
                how="left",
            ).drop(columns="Labels")
        else:
            transformed = data.to_frame()

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


# Register (un-)structure hooks
overrides = {
    "_values": override(rename="values"),
    "decorrelate": override(struct_hook=lambda x, _: x),
}
# FIXME[typing]: https://github.com/python/mypy/issues/4717
converter.register_structure_hook(
    Parameter,
    get_base_structure_hook(Parameter, overrides=overrides),  # type: ignore
)
converter.register_unstructure_hook(
    Parameter, partial(unstructure_base, overrides=overrides)
)
