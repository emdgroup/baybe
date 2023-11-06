# pylint: disable=import-outside-toplevel

"""Temporary serialization functionality for backward compatibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from cattrs.gen import make_dict_structure_fn

from baybe.utils.serialization import converter, get_subclasses

if TYPE_CHECKING:
    from baybe.strategies.base import Strategy


def structure_strategy(val: dict, _) -> Strategy:
    """A ```Strategy` structure hook that uses ```SplitStrategy``` as fallback type."""
    from baybe.strategies import SplitStrategy
    from baybe.strategies.base import Strategy

    try:
        _type = val["type"]
        cls = next(
            (cl for cl in get_subclasses(Strategy) if cl.__name__ == _type), None
        )
    except KeyError:
        cls = SplitStrategy
        warnings.warn(
            f"A strategy has been specified without a corresponding type. "
            f"As a fallback, '{SplitStrategy.__name__}' is used. "
            f"However, this behavior is deprecated and will be disabled in "
            f"a future version.",
            DeprecationWarning,
        )
    fun = make_dict_structure_fn(cls, converter)

    return fun(val, cls)
