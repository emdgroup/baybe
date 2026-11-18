"""Shared configuration for LLM tests."""

import pytest

from baybe.settings import Settings


@pytest.fixture(autouse=True)
def fixture_pandas_backend():
    """Run LLM tests with pandas as the dataframe backend."""
    with Settings(default_dataframe_backend="pandas"):
        yield
