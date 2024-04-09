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

random_acqfs = st.one_of(
    st.sampled_from(
        [
            ExpectedImprovement(),
            ProbabilityOfImprovement(),
            qExpectedImprovement(),
            qProbabilityOfImprovement(),
        ]
    ),
    st.builds(
        UpperConfidenceBound, beta=st.floats(min_value=0.0, allow_infinity=False)
    ),
    st.builds(
        qUpperConfidenceBound, beta=st.floats(min_value=0.0, allow_infinity=False)
    ),
)
