"""A wrapper to load modules lazily."""

import importlib
import importlib.util
from typing import Any, Optional


def optional_import(name: str, attribute: Optional[str] = None) -> Any:
    """Import an optional module or its given attribute.

    Args:
        name: The name of the module.
        attribute: The name of an attribute to import from the module.

    Returns:
        The imported module or attribute from the module, or ``None``
            if the module could not be imported.

    Raises:
        ModuleNotFoundError: If the requested module is not found and the
            ``error`` is set to "raise".
    """
    try:
        module = importlib.import_module(name)
        if attribute is not None and module is not None:
            module = getattr(module, attribute)
        return module
    except ModuleNotFoundError as exc:
        msg = f"Missing dependency '{name}'. " f"Use pip or conda to install {name}."
        raise type(exc)(msg) from None
