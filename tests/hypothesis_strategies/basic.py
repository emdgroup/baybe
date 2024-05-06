"""Strategies for basic types."""

from functools import partial

import hypothesis.strategies as st

finite_floats = partial(st.floats, allow_infinity=False, allow_nan=False)
"""A strategy producing finite (i.e., non-nan and non-infinite) floats."""
