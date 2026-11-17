"""Hypothesis strategies for recommenders."""

from collections.abc import Sequence

import hypothesis.strategies as st

from baybe.recommenders.pure.llm.llm import _RESERVED_LITELLM_KEYS, LLMRecommender
from baybe.recommenders.pure.tfpr import TFPRRecommender
from tests.hypothesis_strategies.basic import finite_floats
from tests.hypothesis_strategies.targets import target_names as target_name_strategy

_litellm_arg_keys = st.text(min_size=1).filter(
    lambda k: k not in _RESERVED_LITELLM_KEYS
)

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


@st.composite
def tfpr_recommenders(draw: st.DrawFn, target_names: Sequence[str] | None = None):
    """Generate :class:`baybe.recommenders.pure.tfpr.TFPRRecommender`.

    Args:
        draw: Hypothesis draw object.
        target_names: Optional names of the targets for which the private ranking
            settings are drawn. If ``None``, the target names are drawn.

    Returns:
        The drawn recommender.
    """
    if target_names is None:
        target_names = draw(st.lists(target_name_strategy, min_size=2, unique=True))
    recommender = TFPRRecommender(
        optimism_lambda=draw(finite_floats(min_value=0.0, max_value=10.0)),
        top_fraction=draw(
            st.none() | finite_floats(min_value=0.0, max_value=1.0, exclude_min=True)
        ),
        dominance_threshold=draw(
            finite_floats(min_value=0.5, max_value=1.0, exclude_max=True)
        ),
    )
    weights = {n: finite_floats(min_value=0.0, max_value=10.0) for n in target_names}
    tolerances = {n: finite_floats(min_value=0.0, max_value=1.0) for n in target_names}
    recommender._weights = draw(
        st.none() | st.fixed_dictionaries(weights).filter(lambda dct: any(dct.values()))
    )
    recommender._tolerances = draw(st.none() | st.fixed_dictionaries(tolerances))
    return recommender
