"""A collection of serialization utilities."""

from typing import Any

import pandas as pd


def serialize_dataframe(df: pd.DataFrame, /) -> Any:
    """Serialize a pandas dataframe."""
    from baybe.searchspace.core import converter

    return converter.unstructure(df)


def deserialize_dataframe(serialized_df: Any, /) -> pd.DataFrame:
    """Deserialize a pandas dataframe."""
    from baybe.searchspace.core import converter

    return converter.structure(serialized_df, pd.DataFrame)
