"""A collection of serialization utilities."""

from __future__ import annotations

import base64
import io
import warnings
from typing import TYPE_CHECKING

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
    - A dictionary with a ``constructor`` key specifying one of
      :class:`narwhals.stable.v2.DataFrame`'s constructor classmethods (e.g.
      ``from_dict``, ``from_dicts``) and the corresponding keyword arguments.

    The backend of the resulting frame is determined by
    :attr:`baybe.active_settings.default_dataframe_backend`.

    For backward compatibility, dictionaries specifying a pandas-only constructor
    (e.g. ``from_records``) are still supported but trigger a
    :class:`DeprecationWarning` and always return a pandas ``DataFrame``, regardless
    of the active backend.

    Args:
        serialized_df: The serialized dataframe.

    Returns:
        A native dataframe in the active backend.

    Raises:
        ValueError: If the input is neither a string nor a dictionary.
        ValueError: If the input is a dict without a ``constructor`` key.
    """
    from baybe.serialization.core import _CONSTRUCTOR_FIELD

    if isinstance(serialized_df, dict) and _CONSTRUCTOR_FIELD not in serialized_df:
        raise ValueError(
            f"For deserializing a dataframe from a dictionary, "
            f"the '{_CONSTRUCTOR_FIELD}' key must be provided.",
        )
    if not isinstance(serialized_df, str | dict):
        raise ValueError(
            "Unknown object type for deserializing a dataframe. Supported types are "
            "strings and dictionaries.",
        )

    # >>>>>>>>>> Deprecation
    if isinstance(serialized_df, dict):
        constructor_name = serialized_df[_CONSTRUCTOR_FIELD]
        if not hasattr(nw.DataFrame, constructor_name) and hasattr(
            pd.DataFrame, constructor_name
        ):
            from baybe.serialization.core import select_constructor_hook

            warnings.warn(
                f"Using the pandas-only constructor '{constructor_name}' for "
                f"deserializing a dataframe is deprecated and will be removed in a "
                f"future version. Use one of '{nw.DataFrame.__name__}'s "
                f"constructors instead (e.g. 'from_dict', 'from_dicts').",
                DeprecationWarning,
                stacklevel=2,
            )
            return select_constructor_hook(serialized_df.copy(), pd.DataFrame)
    # <<<<<<<<<< Deprecation

    return _structure_nw_dataframe(serialized_df).to_native()


def _unstructure_nw_dataframe(df: nw.DataFrame, /) -> str:
    """Serialize a narwhals DataFrame to a base64-encoded Parquet string."""
    # We forbid serializing pandas DataFrames with non-default indices because users
    # should explicitly reset the index rather than unintentionally relying on a
    # potentially lossy round trip. Pandas stores indices as metadata in Parquet and
    # restores them when reading the data back into pandas. Other backends, however, may
    # materialize the index as additional "private" columns. Since we cannot control
    # which backend is used for deserialization, we reject non-default indices instead
    # of pretending that the index can be preserved. This is consistent with BayBE's
    # general design philosophy, which does not rely on pandas indices.
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
                f"Cannot serialize a pandas DataFrame with a non-default index "
                f"({idx!r}). Reset the index before serializing."
            )
    buf = io.BytesIO()
    df.write_parquet(buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _structure_nw_dataframe(obj: str | dict, /) -> nw.DataFrame:
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
