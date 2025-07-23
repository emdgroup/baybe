"""Serialization mixin class."""

import json
from typing import TypeVar

from baybe.serialization.core import _add_type_to_dict, converter

_T = TypeVar("_T", bound="SerialMixin")


class SerialMixin:
    """A mixin class providing serialization functionality."""

    # Use slots so that derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def to_dict(self, *, add_type: bool = False) -> dict:
        """Create an object's dictionary representation.

        Args:
            add_type: Boolean flag allowing to include the object's type.

        Returns:
            The dictionary representation of the object.
        """
        dct = converter.unstructure(self)
        if add_type:
            dct = _add_type_to_dict(dct, self.__class__.__name__)
        return dct

    @classmethod
    def from_dict(cls: type[_T], dictionary: dict) -> _T:
        """Create an object from its dictionary representation.

        Args:
            dictionary: The dictionary representation.

        Returns:
            The reconstructed object.
        """
        return converter.structure(dictionary, cls)

    def to_json(self, *, add_type: bool = False) -> str:
        """Create an object's JSON representation.

        Args:
            add_type: Boolean flag allowing to include the object's type.

        Returns:
            The JSON representation as a string.
        """
        return json.dumps(self.to_dict(add_type=add_type))

    @classmethod
    def from_json(cls: type[_T], string: str) -> _T:
        """Create an object from its JSON representation.

        Args:
            string: The JSON representation of the object.

        Returns:
            The reconstructed object.
        """
        return cls.from_dict(json.loads(string))
