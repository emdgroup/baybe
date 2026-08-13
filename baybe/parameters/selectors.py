"""Parameter selectors."""

import re
from abc import ABC, abstractmethod
from collections.abc import Collection
from typing import Protocol, runtime_checkable

from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.utils.basic import to_tuple
from baybe.utils.conversion import nonstring_to_tuple


@runtime_checkable
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

    types: tuple[type[Parameter], ...] = field(converter=to_tuple)
    """The parameter types to be selected."""

    @override
    def _is_match(self, parameter: Parameter) -> bool:
        return isinstance(parameter, self.types)


@define
class NameSelector(ParameterSelector):
    """Select parameters by name patterns."""

    patterns: tuple[str, ...] = field(
        converter=Converter(  # type: ignore
            nonstring_to_tuple, takes_self=True, takes_field=True
        ),
        validator=[
            min_len(1),
            deep_iterable(member_validator=instance_of(str)),
        ],
    )
    """The patterns to be matched against."""

    regex: bool = field(default=True, validator=instance_of(bool), kw_only=True)
    """If ``False``, the provided patterns are interpreted as literal strings."""

    @override
    def _is_match(self, parameter: Parameter) -> bool:
        if self.regex:
            return any(re.fullmatch(p, parameter.name) for p in self.patterns)
        return parameter.name in self.patterns


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
    * A single string is interpreted as a parameter name pattern and wrapped into a
      :class:`NameSelector` with regex mode active.
    * A single :class:`~baybe.parameters.base.Parameter` subclass is wrapped into a
      :class:`TypeSelector`.
    * A collection of strings is converted to a :class:`NameSelector` with regex mode
      active.
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
        return NameSelector([x], regex=True)

    if isinstance(x, type) and issubclass(x, Parameter):
        return TypeSelector([x])

    if callable(x):
        return x

    # At this point, x should be a collection of strings or parameter types
    items = tuple(x)

    if all(isinstance(item, str) for item in items):
        return NameSelector(items, regex=True)

    if all(isinstance(item, type) and issubclass(item, Parameter) for item in items):
        return TypeSelector(items)

    raise TypeError(f"Cannot convert {x!r} to a parameter selector.")
