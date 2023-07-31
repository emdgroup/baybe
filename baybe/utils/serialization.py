"""Serialization utilities."""

import json
from typing import Type, TypeVar

import cattrs

T = TypeVar("T")


class SerialMixin:
    """A mixin class providing serialization functionality."""

    # Use slots so that the derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def to_dict(self) -> dict:
        """Create an object's dictionary representation."""
        return cattrs.unstructure(self)

    @classmethod
    def from_dict(cls: Type[T], dictionary: dict) -> T:
        """Create an object from its dictionary representation."""
        return cattrs.structure(dictionary, cls)

    def to_json(self) -> str:
        """Create an object's JSON representation."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls: Type[T], string: str) -> T:
        """Create an object from its JSON representation."""
        return cls.from_dict(json.loads(string))
