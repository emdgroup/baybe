"""Hypothesis strategies for recommenders."""

import hypothesis.strategies as st
from hypothesis import assume

from baybe.recommenders.pure.llm.llm import (
    _CREDENTIAL_LITELLM_KEYS,
    _RESERVED_LITELLM_KEYS,
    LLMRecommender,
)

_FORBIDDEN_LITELLM_KEYS = _RESERVED_LITELLM_KEYS | _CREDENTIAL_LITELLM_KEYS

_litellm_values = st.one_of(
    st.text(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)
"""A strategy generating JSON-serializable values suitable for litellm args."""


@st.composite
def _safe_litellm_args(draw: st.DrawFn) -> dict:
    """Generate a litellm args dict containing no reserved or credential keys."""
    args = draw(st.dictionaries(st.text(min_size=1), _litellm_values, max_size=3))
    assume(not set(args).intersection(_FORBIDDEN_LITELLM_KEYS))
    return args


def llm_recommenders() -> st.SearchStrategy[LLMRecommender]:
    """Strategy for :class:`~baybe.recommenders.pure.llm.llm.LLMRecommender`."""
    return st.builds(
        LLMRecommender,
        model=st.text(min_size=1),
        experiment_description=st.text(min_size=1),
        recovery_model=st.one_of(st.none(), st.text(min_size=1)),
        litellm_args=_safe_litellm_args(),
        recovery_litellm_args=st.one_of(st.none(), _safe_litellm_args()),
    )
