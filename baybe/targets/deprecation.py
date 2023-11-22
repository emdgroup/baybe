"""Temporary name aliases for backward compatibility."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baybe.objective import Objective as NewObjective


def Objective(*args, **kwargs) -> NewObjective:
    """An ``Objective`` alias for backward compatibility."""  # noqa: D401 (imperative mood)
    from baybe.objective import Objective as NewObjective

    warnings.warn(
        "The use of `baybe.targets.Objective` is deprecated and will be disabled in "
        "a future version. Use `baybe.objective.Objective` instead.",
        DeprecationWarning,
    )

    return NewObjective(*args, **kwargs)
