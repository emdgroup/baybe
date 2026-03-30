"""Deprecation for kernels."""

from __future__ import annotations

import warnings
from typing import Any


def AdditiveKernel(*args: Any, **kwargs: Any):
    """Return an instance of the renamed SumKernel."""
    warnings.warn(
        "The use of `AdditiveKernel` is deprecated and will be "
        "removed in a future version. Use `SumKernel` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from baybe.kernels.composite import SumKernel

    return SumKernel(*args, **kwargs)
