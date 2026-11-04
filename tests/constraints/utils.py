"""Shared helpers for constraint tests."""

from __future__ import annotations

import os
from collections.abc import Collection
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrameT, IntoLazyFrameT

    from baybe.constraints.base import DiscreteFilteringConstraint

#: Module-level mark restricting a test module to the full test environment.
requires_fulltest = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Requires the optional Polars dependency.",
)


def apply_filtering_constraints(
    df: IntoDataFrameT | IntoLazyFrameT,
    constraints: Collection[DiscreteFilteringConstraint],
) -> IntoDataFrameT | IntoLazyFrameT:
    """Filter a native frame by sequentially applying the given constraints.

    Args:
        df: The native frame to filter.
        constraints: The filtering constraints to apply, in application order.

    Returns:
        The filtered frame.
    """
    for constraint in constraints:
        df = constraint(df)
    return df
