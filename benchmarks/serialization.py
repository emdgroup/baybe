"""Serialization utilities for benchmarking objects."""

import base64
import json
import pickle
from datetime import datetime, timedelta
from typing import Any

import cattrs
import pandas as pd

from baybe.serialization.core import _CONSTRUCTOR_FIELD, select_constructor_hook

converter = cattrs.Converter(unstruct_collection_overrides={set: list})
"""The converter for benchmarking objects."""


def _structure_dataframe_hook(obj: str | dict, _) -> pd.DataFrame:
    """Deserialize a DataFrame."""
    if isinstance(obj, str):
        pickled_df = base64.b64decode(obj.encode("utf-8"))
        return pickle.loads(pickled_df)
    elif isinstance(obj, dict):
        if _CONSTRUCTOR_FIELD not in obj:
            raise ValueError(
                f"For deserializing a dataframe from a dictionary, "
                f"the '{_CONSTRUCTOR_FIELD}' key must be provided.",
            )
        return select_constructor_hook(obj, pd.DataFrame)
    else:
        raise ValueError(
            "Unknown object type for deserializing a dataframe. Supported types are "
            "strings and dictionaries.",
        )


def _unstructure_dataframe_hook(df: pd.DataFrame) -> str:
    """Serialize a DataFrame."""
    pickled_df = pickle.dumps(df)
    return base64.b64encode(pickled_df).decode("utf-8")


converter.register_unstructure_hook(pd.DataFrame, _unstructure_dataframe_hook)
converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)
converter.register_unstructure_hook(datetime, lambda x: x.isoformat())
converter.register_structure_hook(datetime, lambda x, _: datetime.fromisoformat(x))
converter.register_unstructure_hook(timedelta, lambda x: f"{x.total_seconds()}s")
converter.register_structure_hook(
    timedelta, lambda x, _: timedelta(seconds=float(x.removesuffix("s")))
)


class BenchmarkSerialization:
    """Mixin class providing serialization methods."""

    def to_dict(self) -> dict[str, Any]:
        """Create an object's dictionary representation."""
        return converter.unstructure(self)

    def to_json(self) -> str:
        """Create an object's JSON representation."""
        return json.dumps(self.to_dict())
