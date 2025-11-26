"""A collection of serialization utilities."""

from typing import Any

import numpy as np
import pandas as pd


def serialize_ndarray(array: np.ndarray, /) -> Any:
    """Serialize a numpy ndarray."""
    from baybe.serialization import converter

    return converter.unstructure(array)


def deserialize_ndarray(serialized_array: Any, /) -> np.ndarray:
    """Deserialize a numpy ndarray."""
    from baybe.serialization import converter

    return converter.structure(serialized_array, np.ndarray)


def serialize_dataframe(df: pd.DataFrame, /) -> Any:
    """Serialize a pandas dataframe."""
    from baybe.serialization import converter

    return converter.unstructure(df)


def deserialize_dataframe(serialized_df: Any, /) -> pd.DataFrame:
    """Deserialize a pandas dataframe."""
    from baybe.serialization import converter

    return converter.structure(serialized_df, pd.DataFrame)
