"""Scaling utilities."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class ScalerProtocol(Protocol):
    """Type protocol specifying the interface scalers need to implement.

    The protocol is compatible with sklearn scalers such as
    :class:`sklearn.preprocessing.MinMaxScaler` or
    :class:`sklearn.preprocessing.MaxAbsScaler`.
    """

    def fit(self, df: pd.DataFrame, /) -> None:
        """Fit the scaler to a given data set."""

    def transform(self, df: pd.DataFrame, /) -> pd.DataFrame:
        """Transform a data using the fitted scaling logic."""
