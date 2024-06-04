"""Strategies for basic types."""

from functools import partial

import hypothesis.strategies as st
import numpy as np

from baybe.utils.numerical import DTypeFloatNumpy

finite_floats = partial(
    st.floats,
    allow_infinity=False,
    allow_nan=False,
    width=32 if DTypeFloatNumpy == np.float32 else 64,
)
"""A strategy producing finite (i.e., non-nan and non-infinite) floats."""
