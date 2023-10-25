"""Serialization utilities."""

import json
from typing import Any, Callable, Type, TypeVar

import cattrs

from baybe.utils import get_subclasses

_T = TypeVar("_T")


class SerialMixin:
    """A mixin class providing serialization functionality."""

    # Use slots so that the derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def to_dict(self) -> dict:
        """Create an object's dictionary representation."""
        return cattrs.unstructure(self)

    @classmethod
    def from_dict(cls: Type[_T], dictionary: dict) -> _T:
        """Create an object from its dictionary representation.

        Args:
            dictionary: The dictionary representation.

        Returns:
            The reconstructed object.
        """
        return cattrs.structure(dictionary, cls)

    def to_json(self) -> str:
        """Create an object's JSON representation.

        Returns:
            The JSON representation as a string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls: Type[_T], string: str) -> _T:
        """Create an object from its JSON representation.

        Args:
            string: The JSON representation of the object.

        Returns:
            The reconstructed object.
        """
        return cls.from_dict(json.loads(string))


def unstructure_base(base: Any) -> dict:
    """Unstructures an object into a dictionary and adds an entry for the class name.

    Args:
        base: The object that should be unstructured.

    Returns:
        The unstructured dict with the additional entry.
    """
    converter = cattrs.global_converter
    return {
        "type": base.__class__.__name__,
        **converter.unstructure_attrs_asdict(base),
    }


def get_base_structure_hook(base: Type[_T]) -> Callable[[dict], _T]:
    """Return a hook for structuring a dictionary into an appropriate subclass.

    Provides the inverse operation to ```unstructure_base```.

    Args:
        base: The corresponding class

    Returns:
        The hook.
    """

    def structure_base(val: dict, _) -> _T:
        _type = val["type"]
        cls = next((cl for cl in get_subclasses(base) if cl.__name__ == _type), None)
        if cls is None:
            raise ValueError(f"Unknown subclass {_type}.")
        return cattrs.structure_attrs_fromdict(val, cls)

    return structure_base
