"""Parameter selectors."""

import re
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import ClassVar, Protocol

from attrs import Converter, define, field
from attrs.converters import optional
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.searchspace.core import SearchSpace
from baybe.utils.basic import to_tuple
from baybe.utils.conversion import nonstring_to_tuple


class ParameterSelectorProtocol(Protocol):
    """Type protocol specifying the interface parameter selectors need to implement."""

    def __call__(self, parameter: Parameter) -> bool:
        """Determine if a parameter should be included in the selection."""


@define
class ParameterSelector(ParameterSelectorProtocol, ABC):
    """Base class for parameter selectors."""

    exclude: bool = field(default=False, validator=instance_of(bool), kw_only=True)
    """Boolean flag indicating whether invert the selection criterion."""

    @abstractmethod
    def _is_match(self, parameter: Parameter) -> bool:
        """Determine if a parameter meets the selection criterion."""

    @override
    def __call__(self, parameter: Parameter) -> bool:
        """Determine if a parameter should be included in the selection."""
        result = self._is_match(parameter)
        return not result if self.exclude else result


@define
class TypeSelector(ParameterSelector):
    """Select parameters by type."""

    parameter_types: tuple[type[Parameter], ...] = field(converter=to_tuple)
    """The parameter types to be selected."""

    @override
    def _is_match(self, parameter: Parameter) -> bool:
        return isinstance(parameter, self.parameter_types)


@define
class NameSelector(ParameterSelector):
    """Select parameters by name."""

    parameter_names: tuple[str, ...] = field(
        converter=Converter(nonstring_to_tuple, takes_self=True, takes_field=True),
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(str)),
        ],
    )
    """The parameter names to be matched against."""

    regex: bool = field(default=False, validator=instance_of(bool), kw_only=True)
    """If ``True``, the provided names are interpreted as regular expressions."""

    @override
    def _is_match(self, parameter: Parameter) -> bool:
        if self.regex:
            return any(re.fullmatch(p, parameter.name) for p in self.parameter_names)
        return parameter.name in self.parameter_names


def to_parameter_selector(
    x: (
        str
        | type[Parameter]
        | Collection[str]
        | Collection[type[Parameter]]
        | ParameterSelectorProtocol
    ),
    /,
) -> ParameterSelectorProtocol:
    """Convert shorthand notations to parameter selectors.

    Convenience converter that allows users to specify parameter selectors using
    simpler types:

    * A callable (i.e., an existing selector or any object satisfying
      :class:`ParameterSelectorProtocol`) is passed through unchanged.
    * A single string is interpreted as a parameter name and wrapped into a
      :class:`NameSelector`.
    * A single :class:`~baybe.parameters.base.Parameter` subclass is wrapped into a
      :class:`TypeSelector`.
    * A collection of strings is converted to a :class:`NameSelector`.
    * A collection of :class:`~baybe.parameters.base.Parameter` subclasses is converted
      to a :class:`TypeSelector`.

    Args:
        x: The object to convert.

    Returns:
        The corresponding parameter selector.

    Raises:
        TypeError: If the input cannot be converted to a parameter selector.
    """
    if isinstance(x, str):
        return NameSelector([x])

    if isinstance(x, type) and issubclass(x, Parameter):
        return TypeSelector([x])

    if callable(x):
        return x

    # At this point, x should be a collection of strings or parameter types
    items = tuple(x)

    if all(isinstance(item, str) for item in items):
        return NameSelector(items)

    if all(isinstance(item, type) and issubclass(item, Parameter) for item in items):
        return TypeSelector(items)

    raise TypeError(f"Cannot convert {x!r} to a parameter selector.")


@define
class _ParameterSelectorMixin:
    """A mixin class to enable parameter selection."""

    # For internal use only: sanity check mechanism to remind developers of new
    # subclasses to actually use the parameter selector when it is provided
    # TODO: Perhaps we can find a more elegant way to enforce this by design
    _uses_parameter_names: ClassVar[bool] = False

    parameter_selector: ParameterSelectorProtocol | None = field(
        default=None, converter=optional(to_parameter_selector), kw_only=True
    )
    """An optional selector to specify which parameters are to be considered."""

    def get_parameter_names(self, searchspace: SearchSpace) -> tuple[str, ...] | None:
        """Get the names of the parameters to be considered."""
        if self.parameter_selector is None:
            return None

        return tuple(
            p.name for p in searchspace.parameters if self.parameter_selector(p)
        )

    def __attrs_post_init__(self):
        # This helps to ensure that new subclasses actually use the parameter selector
        # by requiring the developer to explicitly set the flag to `True`
        if self.parameter_selector is not None:
            assert self._uses_parameter_names
