"""Serialization utilities."""

import json
from typing import Any, Callable, Optional, Type, TypeVar

import cattrs
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn

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


def unstructure_base(base: Any, overrides: Optional[dict] = None) -> dict:
    """Unstructures an object into a dictionary and adds an entry for the class name.

    Args:
        base: The object that should be unstructured.
        overrides: An optional dictionary of cattrs-overrides for certain attributes.

    Returns:
        The unstructured dict with the additional entry.
    """
    converter = cattrs.global_converter
    fun = make_dict_unstructure_fn(base.__class__, converter, **(overrides or {}))
    attrs_dict = fun(base)
    return {
        "type": base.__class__.__name__,
        **attrs_dict,
    }


def get_base_structure_hook(
    base: Type[_T],
    overrides: Optional[dict] = None,
) -> Callable[[dict], _T]:
    """Return a hook for structuring a dictionary into an appropriate subclass.

    Provides the inverse operation to ```unstructure_base```.

    Args:
        base: The corresponding class
        overrides: An optional dictionary of cattrs-overrides for certain attributes.

    Returns:
        The hook.
    """

    def structure_base(val: dict, _) -> _T:
        _type = val["type"]
        cls = next((cl for cl in get_subclasses(base) if cl.__name__ == _type), None)
        if cls is None:
            raise ValueError(f"Unknown subclass {_type}.")
        fun = make_dict_structure_fn(cls, cattrs.global_converter, **(overrides or {}))
        return fun(val, cls)

    return structure_base
