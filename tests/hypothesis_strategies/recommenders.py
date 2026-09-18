"""Hypothesis strategies for recommenders."""

from hypothesis import strategies as st

from baybe.recommenders.pure.tfpr import TFPRRecommender

_TARGET_NAMES = ("t1", "t2", "t3")


@st.composite
def tfpr_recommenders(draw: st.DrawFn) -> TFPRRecommender:
    """Generate TFPR recommenders."""
    weights = draw(
        st.dictionaries(
            st.sampled_from(_TARGET_NAMES),
            st.integers(min_value=0, max_value=10),
            max_size=len(_TARGET_NAMES),
        )
    )
    tolerances = draw(
        st.dictionaries(
            st.sampled_from(_TARGET_NAMES),
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            max_size=len(_TARGET_NAMES),
        )
    )
    optimism_lambda = draw(
        st.floats(
            min_value=0.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    top_fraction = draw(
        st.none()
        | st.floats(
            min_value=1e-12,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return TFPRRecommender(
        weights=weights,
        tolerances=tolerances,
        optimism_lambda=optimism_lambda,
        top_fraction=top_fraction,
    )
