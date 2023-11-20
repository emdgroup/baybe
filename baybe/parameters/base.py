"""Base classes for all parameters."""

from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import Any, ClassVar, Optional

import pandas as pd
from attr import define, field
from cattrs.gen import override

from baybe.parameters.enum import ParameterEncoding
from baybe.utils import SerialMixin, get_base_structure_hook, unstructure_base
from baybe.utils.serialization import converter

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

    is_discrete: ClassVar[bool]
    """Class variable encoding whether this parameter is discrete."""

    # object variables
    name: str = field()
    """The name of the parameter"""

    @abstractmethod
    def is_in_range(self, item: Any) -> bool:
        """Return whether an item is within the parameter range.

        Args:
            item: The item to be checked.

        Returns:
            ``True`` if the item is within the parameter range, ``False`` otherwise.
        """


@define(frozen=True, slots=False)
class DiscreteParameter(Parameter, ABC):
    """Abstract class for discrete parameters."""

    # TODO [15280]: needs to be refactored

    # class variables
    is_discrete: ClassVar[bool] = True
    # See base class.

    encoding: Optional[ParameterEncoding] = field(init=False, default=None)
    """An optional encoding strategy for the parameter."""

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
