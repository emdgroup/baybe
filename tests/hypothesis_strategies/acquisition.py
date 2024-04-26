"""Hypothesis strategies for acquisition functions."""

import hypothesis.strategies as st

from baybe.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

# These acqfs are ordered roughly according to increasing complexity
acquisition_functions = st.one_of(
    st.builds(ExpectedImprovement),
    st.builds(ProbabilityOfImprovement),
    st.builds(
        UpperConfidenceBound, beta=st.floats(min_value=0.0, allow_infinity=False)
    ),
    st.builds(PosteriorMean),
    st.builds(LogExpectedImprovement),
    st.builds(qExpectedImprovement),
    st.builds(qProbabilityOfImprovement),
    st.builds(
        qUpperConfidenceBound, beta=st.floats(min_value=0.0, allow_infinity=False)
    ),
    st.builds(qSimpleRegret),
    st.builds(qLogExpectedImprovement),
    st.builds(qNoisyExpectedImprovement),
    st.builds(qLogNoisyExpectedImprovement),
)
