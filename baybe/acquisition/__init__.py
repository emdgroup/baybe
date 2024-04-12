"""Acquisition function wrappers."""

from baybe.acquisition.acqfs import (
    ExpectedImprovement,
    LogExpectedImprovement,
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
from baybe.acquisition.adapter import AdapterModel
from baybe.acquisition.partial import PartialAcquisitionFunction

EI = ExpectedImprovement
qEI = qExpectedImprovement
LogEI = LogExpectedImprovement
qLogEI = qLogExpectedImprovement
qNEI = qNoisyExpectedImprovement
qLogNEI = qLogNoisyExpectedImprovement
PI = ProbabilityOfImprovement
qPI = qProbabilityOfImprovement
UCB = UpperConfidenceBound
qUCB = qUpperConfidenceBound
qSR = qSimpleRegret

__all__ = [
    # -----------------------------
    # Acquisition functions
    "ExpectedImprovement",
    "qExpectedImprovement",
    "LogExpectedImprovement",
    "qLogExpectedImprovement",
    "qNoisyExpectedImprovement",
    "qLogNoisyExpectedImprovement",
    "UpperConfidenceBound",
    "qUpperConfidenceBound",
    "ProbabilityOfImprovement",
    "qProbabilityOfImprovement",
    "qSimpleRegret",
    # -----------------------------
    # Abbreviations
    "EI",
    "qEI",
    "LogEI",
    "qLogEI",
    "qNEI",
    "qLogNEI",
    "UCB",
    "qUCB",
    "PI",
    "qPI",
    "qSR",
    # -----------------------------
    # Helpers
    "AdapterModel",
    "PartialAcquisitionFunction",
]
