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

converter = cattrs.GenConverter(unstruct_collection_overrides={set: list})
"""The converter for benchmarking objects."""

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
