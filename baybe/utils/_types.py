"""Shared type aliases."""

from typing import Any, TypeAlias

import narwhals.stable.v2 as nw

_NarwhalsFrame: TypeAlias = nw.DataFrame[Any] | nw.LazyFrame[Any]
"""A Narwhals eager or lazy dataframe."""
