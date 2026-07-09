"""Hypothesis strategies for dataframes."""

from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, indexes

from tests.hypothesis_strategies.basic import finite_floats

# Text strategy excluding surrogate characters (Cs category),
# which cannot be encoded as UTF-8
_safe_text = st.text(alphabet=st.characters(exclude_categories=("Cs",)))


@st.composite
def random_dataframes(draw: st.DrawFn):
    """Generate pandas dataframes of random shape and content."""
    index_elements = st.one_of(_safe_text, st.integers(), finite_floats())

    @st.composite
    def _random_column(draw: st.DrawFn):
        """Generate a random column with safe string elements."""
        name = draw(index_elements)
        dtype = draw(st.sampled_from([int, float, str]))
        elements = _safe_text if dtype is str else None
        return column(name=name, dtype=dtype, elements=elements)

    columns = draw(st.lists(_random_column(), max_size=5, unique_by=lambda c: c.name))
    index_strategy = indexes(elements=index_elements, max_size=2)
    return draw(data_frames(columns=columns, index=index_strategy))
