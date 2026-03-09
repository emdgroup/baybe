"""Custom acquisition functions."""

from baybe.acquisition.custom_acqfs.two_stage import (
    MultiFidelityUpperConfidenceBound,
)

MFUCB = MultiFidelityUpperConfidenceBound

__all__ = [
    ######################### Acquisition functions
    # Upper Confidence Bound
    "MultiFidelityUpperConfidenceBound",
    ######################### Abbreviations
    # Upper Confidence Bound
    "MFUCB",
]
