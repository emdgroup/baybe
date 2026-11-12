"""Shared helpers for condition tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

if TYPE_CHECKING:
    from baybe.constraints.conditions import Condition


def apply_condition(
    condition: Condition,
    values: Sequence[Any],
    backend: nw.Implementation,
) -> list[Any]:
    """Apply a condition to the given values.

    Args:
        condition: The condition to apply.
        values: The values to apply the condition to.
        backend: The backend used to construct the series.

    Returns:
        The values that satisfy the condition.
    """
    series = nw.new_series("value", values, backend=backend)
    filtered = condition(series)
    return nw.from_native(filtered, series_only=True).to_list()
