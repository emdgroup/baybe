# pylint: disable=missing-function-docstring


"""Serialization utilities."""

import json
from typing import Type, TypeVar

import cattrs

from baybe.utils import get_subclasses

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


def unstructure_base(base):
    converter = cattrs.global_converter
    return {
        "type": base.__class__.__name__,
        **converter.unstructure_attrs_asdict(base),
    }


def get_base_unstructure_hook(base):
    def structure_base(val, _):
        _type = val["type"]
        cls = next((cl for cl in get_subclasses(base) if cl.__name__ == _type), None)
        if cls is None:
            raise ValueError(f"Unknown subclass {_type}.")
        return cattrs.structure_attrs_fromdict(val, cls)

    return structure_base
