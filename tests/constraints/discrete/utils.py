"""Shared helpers for discrete constraint tests."""

import os
from collections.abc import Mapping, Sequence

import narwhals.stable.v2 as nw
import pytest
from narwhals.testing import assert_frame_equal as assert_nw_frame_equal

from baybe.constraints.base import DiscreteFilteringConstraint

#: Module-level mark restricting a test module to the full test environment.
requires_fulltest = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Requires the optional Polars dependency.",
)


def assert_filter(
    constraint: DiscreteFilteringConstraint,
    data: Sequence[Mapping[str, object]],
    expected_rows: Sequence[Mapping[str, object]],
    backend: nw.Implementation,
) -> None:
    """Assert that a constraint keeps exactly the expected rows.

    Args:
        constraint: The constraint to evaluate.
        data: The input rows.
        expected_rows: The rows the constraint is expected to keep.
        backend: The dataframe backend used to construct the frames.
    """
    frame = nw.from_dicts(data, backend=backend)
    result = constraint(frame.to_native())
    expected = (
        nw.from_dicts(expected_rows, backend=backend) if expected_rows else frame[[]]
    )
    assert_nw_frame_equal(nw.from_native(result, eager_only=True), expected)


def assert_partial_rejected(
    constraint: DiscreteFilteringConstraint,
    data: Sequence[Mapping[str, object]],
    backend: nw.Implementation,
) -> None:
    """Assert that a constraint rejects a non-evaluable partial frame.

    Args:
        constraint: The constraint to evaluate.
        data: The partial input rows.
        backend: The dataframe backend used to construct the frame.

    Raises:
        ValueError: If the constraint does not reject the frame.
    """
    assert constraint._can_evaluate(set(data[0])) is False
    frame = nw.from_dicts(data, backend=backend)
    with pytest.raises(ValueError, match="missing from the provided dataframe"):
        constraint(frame.to_native())
