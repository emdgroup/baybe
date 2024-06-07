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
    qNegIntegratedPosteriorVariance,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from baybe.utils.sampling_algorithms import SamplingMethod

from ..hypothesis_strategies.basic import finite_floats

# These acqfs are ordered roughly according to increasing complexity
acquisition_functions = st.one_of(
    st.builds(ExpectedImprovement),
    st.builds(ProbabilityOfImprovement),
    st.builds(UpperConfidenceBound, beta=finite_floats(min_value=0.0)),
    st.builds(PosteriorMean),
    st.builds(LogExpectedImprovement),
    st.builds(qExpectedImprovement),
    st.builds(qProbabilityOfImprovement),
    st.builds(qUpperConfidenceBound, beta=finite_floats(min_value=0.0)),
    st.builds(qSimpleRegret),
    st.builds(qLogExpectedImprovement),
    st.builds(qNoisyExpectedImprovement),
    st.builds(qLogNoisyExpectedImprovement),
    st.builds(
        qNegIntegratedPosteriorVariance,
        sampling_method=st.sampled_from(SamplingMethod),
        sampling_fraction=finite_floats(min_value=0.0, max_value=1.0, exclude_min=True),
        sampling_n_points=st.one_of(st.none(), st.integers(min_value=1)),
    ),
)
