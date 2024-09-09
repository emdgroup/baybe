"""Serialization mixin class."""

import json
from typing import TypeVar

from baybe.serialization.core import converter

_T = TypeVar("_T")


class SerialMixin:
    """A mixin class providing serialization functionality."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def to_dict(self) -> dict:
        """Create an object's dictionary representation."""
        return converter.unstructure(self)

    @classmethod
    def from_dict(cls: type[_T], dictionary: dict) -> _T:
        """Create an object from its dictionary representation.

        Args:
            dictionary: The dictionary representation.

        Returns:
            The reconstructed object.
        """
        return converter.structure(dictionary, cls)

    def to_json(self) -> str:
        """Create an object's JSON representation.

        Returns:
            The JSON representation as a string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls: type[_T], string: str) -> _T:
        """Create an object from its JSON representation.

        Args:
            string: The JSON representation of the object.

        Returns:
            The reconstructed object.
        """
        return cls.from_dict(json.loads(string))
