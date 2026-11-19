"""Hypothesis strategies for dataframes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw
import pandas as pd
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes

from baybe.utils.dataframe import _df_with_backend
from tests.hypothesis_strategies.basic import finite_floats

if TYPE_CHECKING:
    from narwhals.stable.v2.typing import IntoDataFrame

# Text strategy excluding surrogate characters (Cs category, cannot be encoded as
# UTF-8) and null bytes (rejected by Parquet and Arrow column name schemas)
_safe_text = st.text(
    alphabet=st.characters(exclude_categories=("Cs",), exclude_characters=("\x00",))
)


@st.composite
def random_dataframes(
    draw: st.DrawFn,
    *,
    string_column_names_only: bool = False,
    backend: nw.Implementation = nw.Implementation.PANDAS,
) -> IntoDataFrame:
    """Generate dataframes of random shape and content.

    Args:
        draw: The hypothesis draw object for composite strategies.
        string_column_names_only: If ``True``, restrict column names to strings.
            Required when the output must be compatible with formats that mandate
            string column names (e.g. Parquet).
        backend: The target native backend for the generated frame.

    Returns:
        A dataframe in the requested backend.
    """
    # IMPROVE: Replace the manual backend conversion with a native hypothesis strategy
    #   for each backend once one becomes available (e.g. polars plugin or similar)
    index_elements = st.one_of(_safe_text, st.integers(), finite_floats())
    column_name_strategy = _safe_text if string_column_names_only else index_elements

    @st.composite
    def _random_column(draw: st.DrawFn):
        """Generate a random column with safe string elements."""
        name = draw(column_name_strategy)
        dtype = draw(st.sampled_from([int, float, str]))
        elements = _safe_text if dtype is str else None
        return column(name=name, dtype=dtype, elements=elements)

    columns = draw(st.lists(_random_column(), max_size=5, unique_by=lambda c: c.name))
    index_strategy = indexes(elements=index_elements, max_size=2)
    df: pd.DataFrame = draw(data_frames(columns=columns, index=index_strategy))

    # Always reset the index: BayBE frames are positional and index information
    # is neither meaningful nor preserved across serialization.
    df = df.reset_index(drop=True)

    return _df_with_backend(nw.from_native(df, eager_only=True), backend).to_native()
