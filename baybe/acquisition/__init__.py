"""Acquisition function wrappers."""

from baybe.acquisition.acqfs import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from baybe.acquisition.adapter import AdapterModel, debotorchize
from baybe.acquisition.partial import PartialAcquisitionFunction

EI = ExpectedImprovement
PI = ProbabilityOfImprovement
UCB = UpperConfidenceBound
aEI = qExpectedImprovement
aPI = qProbabilityOfImprovement
qUCB = qUpperConfidenceBound

__all__ = [
    # ---------------------------
    # Acquisition functions
    "ExpectedImprovement",
    "ProbabilityOfImprovement",
    "UpperConfidenceBound",
    "qExpectedImprovement",
    "qProbabilityOfImprovement",
    "qUpperConfidenceBound",
    # ---------------------------
    # Abbreviations
    "EI",
    "PI",
    "UCB",
    "qEI",
    "qPI",
    "qUCB",
    # ---------------------------
    # Helpers
    "debotorchize",
    "AdapterModel",
    "PartialAcquisitionFunction",
]
