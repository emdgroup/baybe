"""Scaling utilities."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from sklearn.base import BaseEstimator, TransformerMixin

    Scaler: TypeAlias = BaseEstimator | TransformerMixin


class ScalingMethod(Enum):
    """Available scaling methods."""

    IDENTITY = "IDENTITY"
    """Identity transformation (no scaling applied)."""

    MINMAX = "MINMAX"
    """Min-max scaling, mapping the observed value range to [0, 1]."""

    MAXABS = "MAXABS"
    """Max-abs scaling, normalizing by the largest observed absolute."""


def make_scaler(method: ScalingMethod, /) -> Scaler | Literal["passthrough"]:
    """Create a scaler object based on the specified method."""
    from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

    match method:
        case ScalingMethod.IDENTITY:
            return "passthrough"
        case ScalingMethod.MINMAX:
            return MinMaxScaler()
        case ScalingMethod.MAXABS:
            return MaxAbsScaler()
