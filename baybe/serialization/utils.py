"""A collection of serialization utilities."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw
import pandas as pd

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame


def serialize_dataframe(df: IntoDataFrame, /) -> str:
    """Serialize a dataframe to a string."""
    return _unstructure_nw_dataframe(nw.from_native(df, eager_only=True))


def deserialize_dataframe(serialized_df: str | dict, /) -> IntoDataFrame:
    """Deserialize a dataframe.

    Supports two input formats:

    - A string as produced by :func:`serialize_dataframe`.
    - A dictionary with a ``constructor`` key specifying a constructor method name
      and the corresponding keyword arguments. The constructor is looked up on
      the native dataframe class of the active backend.

    The backend of the resulting frame is determined by
    :attr:`baybe.active_settings.default_dataframe_backend`.

    Args:
        serialized_df: The serialized dataframe.

    Returns:
        A native dataframe in the active backend.

    Raises:
        ValueError: If the input is neither a string nor a dictionary.
        ValueError: If the input is a dict without a ``constructor`` key.
    """
    from baybe.serialization.core import _CONSTRUCTOR_FIELD, select_constructor_hook

    if isinstance(serialized_df, str):
        return _structure_nw_dataframe(serialized_df, None).to_native()
    elif isinstance(serialized_df, dict):
        if _CONSTRUCTOR_FIELD not in serialized_df:
            raise ValueError(
                f"For deserializing a dataframe from a dictionary, "
                f"the '{_CONSTRUCTOR_FIELD}' key must be provided.",
            )
        return select_constructor_hook(serialized_df, pd.DataFrame)
    else:
        raise ValueError(
            "Unknown object type for deserializing a dataframe. Supported types are "
            "strings and dictionaries.",
        )


def _unstructure_nw_dataframe(df: nw.DataFrame, /) -> str:
    """Serialize a narwhals DataFrame to a base64-encoded Parquet string."""
    native = df.to_native()
    if isinstance(native, pd.DataFrame):
        idx = native.index
        is_default_range = (
            isinstance(idx, pd.RangeIndex)
            and idx.start == 0
            and idx.stop == len(native)
            and idx.step == 1
        )
        if not is_default_range:
            raise ValueError(
                f"Cannot serialize a pandas DataFrame with a non-trivial index "
                f"({idx!r}). Reset the index before serializing."
            )
    buf = io.BytesIO()
    df.write_parquet(buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _structure_nw_dataframe(obj: str | dict, _: Any, /) -> nw.DataFrame:
    """Deserialize a narwhals DataFrame from a base64-encoded Parquet string or dict."""
    from baybe.settings import active_settings

    if isinstance(obj, dict):
        from baybe.serialization.core import select_constructor_hook

        obj = obj.copy()
        obj.setdefault("backend", active_settings.default_dataframe_backend)
        return select_constructor_hook(obj, nw.DataFrame)

    # IMPROVE: Replace tempfile with a direct buffer once narwhals supports
    # nw.read_parquet from buffers (currently path-only):
    # https://github.com/narwhals-dev/narwhals/issues/3948
    import tempfile

    backend = active_settings.default_dataframe_backend
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        f.write(base64.b64decode(obj))
        f.flush()
        return nw.read_parquet(f.name, backend=backend)
