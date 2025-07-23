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
    qLogNoisyExpectedHypervolumeImprovement,
    qLogNoisyExpectedImprovement,
    qNegIntegratedPosteriorVariance,
    qNoisyExpectedHypervolumeImprovement,
    qNoisyExpectedImprovement,
    qPosteriorStandardDeviation,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from baybe.utils.sampling_algorithms import DiscreteSamplingMethod
from tests.hypothesis_strategies.basic import finite_floats


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


@st.composite
def _reference_points(draw: st.DrawFn):
    """Draw reference points for hypervolume improvement acquisition functions."""
    if draw(st.booleans()):
        return draw(st.lists(finite_floats(), min_size=1))
    return draw(finite_floats())


# These acqfs are ordered roughly according to increasing complexity
acquisition_functions = st.one_of(
    st.builds(ExpectedImprovement),
    st.builds(ProbabilityOfImprovement),
    st.builds(UpperConfidenceBound, beta=finite_floats()),
    st.builds(PosteriorMean),
    st.builds(PosteriorStandardDeviation, maximize=st.sampled_from([True, False])),
    st.builds(qPosteriorStandardDeviation),
    st.builds(LogExpectedImprovement),
    st.builds(qExpectedImprovement),
    st.builds(qProbabilityOfImprovement),
    st.builds(qUpperConfidenceBound, beta=finite_floats()),
    st.builds(qSimpleRegret),
    st.builds(qLogExpectedImprovement),
    st.builds(
        qKnowledgeGradient, num_fantasies=st.integers(min_value=1, max_value=512)
    ),
    st.builds(qNoisyExpectedImprovement, prune_baseline=st.booleans()),
    st.builds(qLogNoisyExpectedImprovement, prune_baseline=st.booleans()),
    _qNIPV_strategy(),
    st.builds(
        qNoisyExpectedHypervolumeImprovement,
        prune_baseline=st.booleans(),
        reference_point=_reference_points(),
    ),
    st.builds(
        qLogNoisyExpectedHypervolumeImprovement,
        prune_baseline=st.booleans(),
        reference_point=_reference_points(),
    ),
)
