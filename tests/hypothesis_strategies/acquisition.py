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
from baybe.acquisition.base import AcquisitionFunction
from baybe.utils.basic import get_subclasses

acqf_long_names = [cl.__name__ for cl in get_subclasses(AcquisitionFunction)]
acqf_abbreviations = [
    cl._abbreviation
    for cl in get_subclasses(AcquisitionFunction)
    if hasattr(cl, "_abbreviation")
]

acqf_objects = st.one_of(
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

random_acqfs = st.one_of(
    st.sampled_from(acqf_long_names), st.sampled_from(acqf_long_names), acqf_objects
)
