"""Converter and hooks."""
import base64
import pickle
from typing import Any, Callable, Optional, Type, TypeVar, get_type_hints

import cattrs
import pandas as pd
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn

_T = TypeVar("_T")

# TODO: This urgently needs the `forbid_extra_keys=True` flag, which requires us to
#   switch to the cattrs built-in subclass recommender.
converter = cattrs.Converter()
"""The default converter for (de-)serializing BayBE-related objects."""


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
    from baybe.utils.basic import get_subclasses

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
    """Deserialize a DataFrame."""
    pickled_df = base64.b64decode(string.encode("utf-8"))
    return pickle.loads(pickled_df)


def _unstructure_dataframe_hook(df: pd.DataFrame) -> str:
    """Serialize a DataFrame."""
    pickled_df = pickle.dumps(df)
    return base64.b64encode(pickled_df).decode("utf-8")


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


def select_constructor_hook(specs: dict, cls: Type[_T]) -> _T:
    """Use the constructor specified in the 'constructor' field for deserialization."""
    # If a constructor is specified, use it
    specs = specs.copy()
    if constructor_name := specs.pop("constructor", None):
        constructor = getattr(cls, constructor_name)

        # Extract the constructor parameter types and deserialize the arguments
        type_hints = get_type_hints(constructor)
        for key, value in specs.items():
            annotation = type_hints[key]
            specs[key] = converter.structure(specs[key], annotation)

        # Call the constructor with the deserialized arguments
        return constructor(**specs)

    # Otherwise, use the regular __init__ method
    return converter.structure_attrs_fromdict(specs, cls)


# Register un-/structure hooks
converter.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)
