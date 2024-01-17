"""Utilities for handling optional dependencies."""

import importlib
import importlib.util
import warnings
from typing import Any, Literal, Optional


def import_optional_module(
    name: str,
    attribute: Optional[str] = None,
    error: Literal["raise", "warn", "ignore"] = "raise",
) -> Optional[Any]:
    """Import an optional module.

    Args:
        name: The name of the module.
        attribute: The name of an attribute to import from the module.
        error: How to handle errors.
            One of:
                - "raise": Raise an error if the module cannot be imported.
                - "warn": Raise a warning if the module cannot be imported.
                - "ignore": Ignore the missing module and return `None`.

    Returns:
        Optional[Any]: The imported module or attribute from the module, or `None`
             if the module could not be imported.

    Raises:
        ValueError: If the given error type value is not in the provided list of
             accepted values: 'raise', 'warn', 'ignore'.
        ModuleNotFoundError: If the requested module is not found and the `error` is set
            to "raise".
    """
    if error not in ("raise", "warn", "ignore"):
        raise ValueError(
            "Expected `error` to be one of 'raise', 'warn, or 'ignore', "
            f"but got {error}.",
        )

    try:
        module = importlib.import_module(name)
        if attribute is not None and module is not None:
            module = getattr(module, attribute)
        return module
    except ModuleNotFoundError as exc:
        msg = (
            f"Missing optional dependency '{name}'. "
            f"Use pip or conda to install {name}."
        )
        if error == "raise":
            raise type(exc)(msg) from None
        if error == "warn":
            warnings.warn(msg, category=ImportWarning, stacklevel=2)

    return None
