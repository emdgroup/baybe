"""Temporary functionality for backward compatibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from cattrs.gen import make_dict_structure_fn

from baybe.utils.serialization import converter, get_subclasses

if TYPE_CHECKING:
    from baybe.strategies.base import Strategy as BaseStrategy
    from baybe.strategies.composite import TwoPhaseStrategy


def structure_strategy(val: dict, _) -> BaseStrategy:
    """A ``Strategy`` structure hook using ``TwoPhaseStrategy`` as fallback type."""  # noqa: D401 (imperative mood)
    from baybe.strategies.base import Strategy as BaseStrategy
    from baybe.strategies.composite import TwoPhaseStrategy

    try:
        _type = val["type"]
        cls = next(
            (cl for cl in get_subclasses(BaseStrategy) if cl.__name__ == _type), None
        )
        if cls is None:
            raise ValueError(f"Unknown subclass '{_type}'.")
    except KeyError:
        cls = TwoPhaseStrategy
        warnings.warn(
            f"A strategy has been specified without a corresponding type. "
            f"As a fallback, '{TwoPhaseStrategy.__name__}' is used. "
            f"However, this behavior is deprecated and will be disabled in "
            f"a future version.",
            DeprecationWarning,
        )
    fun = make_dict_structure_fn(cls, converter)

    return fun(val, cls)


def Strategy(*args, **kwargs) -> TwoPhaseStrategy:
    """A ``Strategy`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    from baybe.strategies.composite import TwoPhaseStrategy

    warnings.warn(
        f"Using 'Strategy' directly is deprecated and will be removed in a future "
        f"version. Please use '{TwoPhaseStrategy.__name__}' class instead.",
        DeprecationWarning,
    )

    return TwoPhaseStrategy(*args, **kwargs)
