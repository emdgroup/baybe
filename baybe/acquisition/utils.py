"""Utilities for acquisition functions."""

from functools import partial

from botorch.acquisition import (
    ExpectedImprovement,
    PosteriorMean,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)

acquisition_function_mapping = {
    "PM": PosteriorMean,
    "PI": ProbabilityOfImprovement,
    "EI": ExpectedImprovement,
    "UCB": partial(UpperConfidenceBound, beta=1.0),
    "qEI": qExpectedImprovement,
    "qPI": qProbabilityOfImprovement,
    "qUCB": partial(qUpperConfidenceBound, beta=1.0),
    "VarUCB": partial(UpperConfidenceBound, beta=100.0),
    "qVarUCB": partial(qUpperConfidenceBound, beta=100.0),
}
