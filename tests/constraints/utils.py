"""Shared helpers for constraint tests."""

from __future__ import annotations

import os

import pytest

#: Module-level mark restricting a test module to the full test environment.
requires_fulltest = pytest.mark.skipif(
    os.environ.get("BAYBE_TEST_ENV") != "FULLTEST",
    reason="Requires the optional Polars dependency.",
)
