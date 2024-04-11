"""Hypothesis strategies for acquisition functions."""

import hypothesis.strategies as st

from baybe.acquisition import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)

acquisition_functions = st.one_of(
    st.builds(ExpectedImprovement),
    st.builds(ProbabilityOfImprovement),
    st.builds(qExpectedImprovement),
    st.builds(qProbabilityOfImprovement),
    st.builds(
        UpperConfidenceBound, beta=st.floats(min_value=0.0, allow_infinity=False)
    ),
    st.builds(
        qUpperConfidenceBound, beta=st.floats(min_value=0.0, allow_infinity=False)
    ),
)
