"""Hypothesis strategies for recommenders."""

import hypothesis.strategies as st

from baybe.recommenders.pure.llm.llm import (
    _CREDENTIAL_LITELLM_KEYS,
    _RESERVED_LITELLM_KEYS,
    LLMRecommender,
)

_FORBIDDEN_KEYS = _RESERVED_LITELLM_KEYS | _CREDENTIAL_LITELLM_KEYS

_litellm_arg_keys = st.text(min_size=1).filter(lambda k: k not in _FORBIDDEN_KEYS)

_litellm_arg_values = st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.text(),
)


@st.composite
def llm_recommenders(draw: st.DrawFn):
    """Generate ``LLMRecommender`` instances with varied fields."""
    return LLMRecommender(
        model=draw(st.text(min_size=1)),
        experiment_description=draw(st.text(min_size=1)),
        litellm_args=draw(
            st.dictionaries(keys=_litellm_arg_keys, values=_litellm_arg_values)
        ),
    )
