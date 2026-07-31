"""Tests for parameter encoders."""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from exceptiongroup import ExceptionGroup
from typing_extensions import Never

from baybe.parameters.encoding import _Encoder


def _pandas_encoder(series: pd.Series) -> pd.DataFrame:
    """Pandas passthrough encoder."""
    assert isinstance(series, pd.Series), f"Expected pd.Series, got {type(series)}"
    return series.to_frame()


def _polars_encoder(series: pl.Series) -> pl.DataFrame:
    """Polars passthrough encoder."""
    assert isinstance(series, pl.Series), f"Expected pl.Series, got {type(series)}"
    return series.to_frame()


def _nw_series(backend: str = "polars") -> nw.Series:
    """Create a narwhals Series."""
    return nw.new_series(name="x", values=["A", "B"], backend=backend)


@pytest.mark.parametrize("series_backend", ["pandas", "polars"])
@pytest.mark.parametrize("encoder_backend", ["pandas", "polars"])
def test_wrapped_encoder_accepts_any_input_backend(series_backend, encoder_backend):
    """A wrapped encoder handles any input/encoder backend combination."""
    user_encoder = globals()[f"_{encoder_backend}_encoder"]
    encoder = _Encoder(encoder=user_encoder)
    assert encoder._implementation is None

    series = _nw_series(backend=series_backend)
    encoded = encoder(series)

    # The wrapper correctly infers the user-native backend.
    assert encoder._implementation is nw.Implementation.from_string(encoder_backend)

    # The returned output backend matches the input series backend, regardless of the
    # encoder's native backend.
    assert nw.get_native_namespace(encoded) is nw.get_native_namespace(series)


def test_callable_not_retried_after_backend_cached():
    """Once the backend is cached, the encoder is queried once per call."""
    call_count = 0

    def counting_encoder(series: pd.Series) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        assert isinstance(series, pd.Series), f"Expected pd.Series, got {type(series)}"
        return pd.DataFrame()

    encoder = _Encoder(encoder=counting_encoder)

    # The first call may trigger several inner calls to determine the backend
    encoder(_nw_series())
    initial_call_count = call_count
    assert initial_call_count > 0

    # Once the backend is cached, only one call is required to encode
    encoder(_nw_series())
    assert call_count == initial_call_count + 1


def test_broken_encoder_raises_exception_group():
    """An encoder that fails for all backends raises an ExceptionGroup."""

    def broken_encoder(_: Any) -> Never:
        raise ValueError("intentional failure")

    enc = _Encoder(encoder=broken_encoder)
    with pytest.raises(ExceptionGroup, match="failed for all tried backends"):
        enc(_nw_series())
