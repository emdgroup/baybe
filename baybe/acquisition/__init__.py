"""Acquisition function wrappers."""

from baybe.acquisition.adapter import AdapterModel, debotorchize
from baybe.acquisition.partial import PartialAcquisitionFunction

__all__ = [
    "debotorchize",
    "AdapterModel",
    "PartialAcquisitionFunction",
]
