"""Custom acquisition functions."""

from baybe.acquisition.custom_acqfs.two_stage import (
    MultiFidelityUpperConfidenceBound,
)

__all__ = [
    ######################### Acquisition functions
    # Upper Confidence Bound
    "MultiFidelityUpperConfidenceBound",
]
