"""Serialization utilities."""
import base64
import json
from io import BytesIO
from typing import Any, Callable, Optional, Type, TypeVar

import cattrs
import pandas as pd
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn

from baybe.utils import get_subclasses

_T = TypeVar("_T")

converter = cattrs.Converter()
"""The default converter for (de-)serializing BayBE-related objects."""


class SerialMixin:
    """A mixin class providing serialization functionality."""

    # Use slots so that the derived classes also remain slotted
    # See also: https://www.attrs.org/en/stable/glossary.html#term-slotted-classes
    __slots__ = ()

    def to_dict(self) -> dict:
        """Create an object's dictionary representation."""
        return converter.unstructure(self)

    @classmethod
    def from_dict(cls: Type[_T], dictionary: dict) -> _T:
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
    def from_json(cls: Type[_T], string: str) -> _T:
        """Create an object from its JSON representation.

        Args:
            string: The JSON representation of the object.

        Returns:
            The reconstructed object.
        """
        return cls.from_dict(json.loads(string))


def unstructure_base(base: Any, overrides: Optional[dict] = None) -> dict:
    """Unstructure an object into a dictionary and adds an entry for the class name.

    Args:
        base: The object that should be unstructured.
        overrides: An optional dictionary of cattrs-overrides for certain attributes.

    Returns:
        The unstructured dict with the additional entry.
    """
    # TODO: use include_subclasses (https://github.com/python-attrs/cattrs/issues/434)

    fun = make_dict_unstructure_fn(base.__class__, converter, **(overrides or {}))
    attrs_dict = fun(base)
    return {
        "type": base.__class__.__name__,
        **attrs_dict,
    }


def get_base_structure_hook(
    base: Type[_T],
    overrides: Optional[dict] = None,
) -> Callable[[dict, Type[_T]], _T]:
    """Return a hook for structuring a dictionary into an appropriate subclass.

    Provides the inverse operation to ``unstructure_base``.

    Args:
        base: The corresponding class
        overrides: An optional dictionary of cattrs-overrides for certain attributes.

    Returns:
        The hook.
    """
    # TODO: use include_subclasses (https://github.com/python-attrs/cattrs/issues/434)

    def structure_base(val: dict, _: Type[_T]) -> _T:
        _type = val.pop("type")
        cls = next((cl for cl in get_subclasses(base) if cl.__name__ == _type), None)
        if cls is None:
            raise ValueError(f"Unknown subclass '{_type}'.")
        fun = make_dict_structure_fn(
            cls, converter, **(overrides or {}), _cattrs_forbid_extra_keys=True
        )
        return fun(val, cls)

    return structure_base


def _structure_dataframe_hook(string: str, _) -> pd.DataFrame:
    """De-serialize a DataFrame."""
    buffer = BytesIO()
    buffer.write(base64.b64decode(string.encode("utf-8")))
    return pd.read_parquet(buffer)


def _unstructure_dataframe_hook(df: pd.DataFrame) -> str:
    """Serialize a DataFrame."""
    return base64.b64encode(df.to_parquet()).decode("utf-8")


converter.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)


def block_serialization_hook(obj: Any) -> None:  # noqa: DOC101, DOC103
    """Prevent serialization of the passed object.

    Raises:
         NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"Serializing objects of type '{obj.__class__.__name__}' is not supported."
    )


def block_deserialization_hook(_: Any, cls: type) -> None:  # noqa: DOC101, DOC103
    """Prevent deserialization into a specific type.

    Raises:
         NotImplementedError: Always.
    """
    raise NotImplementedError(
        f"Deserialization into '{cls.__name__}' is not supported."
    )
