"""Utilities for acquisition functions."""

from baybe.acquisition.base import AcquisitionFunction


def str_to_acqf(name: str, /) -> AcquisitionFunction:
    """Create an ACQF object from a given ACQF name."""
    return AcquisitionFunction.from_dict({"type": name})


def convert_acqf(acqf: AcquisitionFunction | str, /) -> AcquisitionFunction:
    """Convert an ACQF name into an ACQF object (with ACQF object passthrough)."""
    return acqf if isinstance(acqf, AcquisitionFunction) else str_to_acqf(acqf)
