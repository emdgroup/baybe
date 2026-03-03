"""Parameter selectors."""

from abc import abstractmethod
from typing import Protocol

from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.parameters.base import Parameter


class ParameterSelectorProtocol(Protocol):
    """Type protocol specifying the interface parameter selectors need to implement."""

    def __call__(self, parameter: Parameter) -> bool:
        """Determine if a parameter should be included in the selection."""


@define
class ParameterSelector(ParameterSelectorProtocol):
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

    parameter_types: tuple[type[Parameter], ...] = field(converter=tuple)
    """The parameter types to be selected."""

    @override
    def _is_match(self, parameter: Parameter) -> bool:
        return isinstance(parameter, self.parameter_types)
