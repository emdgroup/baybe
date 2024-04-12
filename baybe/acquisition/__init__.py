"""Acquisition function wrappers."""

from baybe.acquisition.acqfs import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from baybe.acquisition.adapter import AdapterModel
from baybe.acquisition.partial import PartialAcquisitionFunction

EI = ExpectedImprovement
PI = ProbabilityOfImprovement
UCB = UpperConfidenceBound
qEI = qExpectedImprovement
qPI = qProbabilityOfImprovement
qUCB = qUpperConfidenceBound
qSR = qSimpleRegret
qNEI = qNoisyExpectedImprovement

__all__ = [
    # ---------------------------
    # Acquisition functions
    "ExpectedImprovement",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
    "qExpectedImprovement",
    "qProbabilityOfImprovement",
    "qUpperConfidenceBound",
    "qSimpleRegret",
    "qNoisyExpectedImprovement",
    # ---------------------------
    # Abbreviations
    "EI",
    "PI",
    "UCB",
    "qEI",
    "qPI",
    "qUCB",
    "qSR",
    "qNEI",
    # ---------------------------
    # Helpers
    "AdapterModel",
    "PartialAcquisitionFunction",
]
