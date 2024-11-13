"""Serialization utilities for benchmarking objects."""

import json
from datetime import datetime, timedelta
from typing import Any

import cattrs
import pandas as pd

from baybe.serialization.core import (
    _structure_dataframe_hook,
    _unstructure_dataframe_hook,
)

benchmarking_converter = cattrs.GenConverter(unstruct_collection_overrides={set: list})
"""The converter for benchmarking objects."""

benchmarking_converter.register_unstructure_hook(
    pd.DataFrame, _unstructure_dataframe_hook
)
benchmarking_converter.register_structure_hook(pd.DataFrame, _structure_dataframe_hook)
benchmarking_converter.register_unstructure_hook(datetime, lambda x: x.isoformat())
benchmarking_converter.register_structure_hook(
    datetime, lambda x, _: datetime.fromisoformat(x)
)
benchmarking_converter.register_unstructure_hook(
    timedelta, lambda x: f"{x.total_seconds()}s"
)
benchmarking_converter.register_structure_hook(
    timedelta, lambda x, _: timedelta(seconds=float(x.removesuffix("s")))
)


class Serializable:
    """Decorator to add serialization methods to a class."""

    def to_dict(self) -> dict[str, Any]:
        """Create an object's dictionary representation."""
        return benchmarking_converter.unstructure(self)

    def to_json(self) -> str:
        """Create an object's JSON representation.

        Returns:
            The JSON representation as a string.
        """
        return json.dumps(self.to_dict())
