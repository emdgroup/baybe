"""Hypothesis strategies for acquisition functions."""

import hypothesis.strategies as st

from baybe.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
    PosteriorStandardDeviation,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qKnowledgeGradient,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qNegIntegratedPosteriorVariance,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

from ..hypothesis_strategies.basic import finite_floats


@st.composite
def _qNIPV_strategy(draw: st.DrawFn):
    sampling_fraction = draw(
        st.one_of(
            finite_floats(min_value=0.0, max_value=1.0, exclude_min=True),
            st.none(),
        )
    )

    sampling_n_points = None
    if sampling_fraction is None:
        sampling_n_points = draw(st.one_of(st.none(), st.integers(min_value=1)))

    return qNegIntegratedPosteriorVariance(
        sampling_fraction=sampling_fraction,
        sampling_n_points=sampling_n_points,
        sampling_method=draw(st.sampled_from(DiscreteSamplingMethod)),
    )


# These acqfs are ordered roughly according to increasing complexity
acquisition_functions = st.one_of(
    st.builds(ExpectedImprovement),
    st.builds(ProbabilityOfImprovement),
    st.builds(UpperConfidenceBound, beta=finite_floats(min_value=0.0)),
    st.builds(PosteriorMean),
    st.builds(PosteriorStandardDeviation, maximize=st.sampled_from([True, False])),
    st.builds(LogExpectedImprovement),
    st.builds(qExpectedImprovement),
    st.builds(qProbabilityOfImprovement),
    st.builds(qUpperConfidenceBound, beta=finite_floats(min_value=0.0)),
    st.builds(qSimpleRegret),
    st.builds(qLogExpectedImprovement),
    st.builds(
        qKnowledgeGradient, num_fantasies=st.integers(min_value=1, max_value=512)
    ),
    st.builds(qNoisyExpectedImprovement),
    st.builds(qLogNoisyExpectedImprovement),
    _qNIPV_strategy(),
)
