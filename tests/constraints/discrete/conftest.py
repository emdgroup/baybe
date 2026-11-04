"""Fixtures for discrete constraint tests."""

import narwhals.stable.v2 as nw
import pytest


@pytest.fixture(
    params=[nw.Implementation.PANDAS, nw.Implementation.POLARS],
    ids=["pandas", "polars"],
    name="backend",
)
def fixture_backend(request: pytest.FixtureRequest) -> nw.Implementation:
    """Return the dataframe backend used to construct test frames."""
    return request.param
