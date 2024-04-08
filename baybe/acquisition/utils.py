"""Utilities for acquisition functions."""

from typing import Union

from baybe.acquisition.base import AcquisitionFunction
from baybe.utils.basic import get_subclasses


def str_to_acqf(name: str, /) -> AcquisitionFunction:
    """Create an ACQF object from a given ACQF name."""
    acqfs = get_subclasses(AcquisitionFunction)
    return next(acqf for acqf in acqfs if name in (acqf.__name__, acqf._abbreviation))()


def convert_acqf(acqf: Union[AcquisitionFunction, str], /) -> AcquisitionFunction:
    """Convert an ACQF name into an ACQF object (with ACQF object passthrough)."""
    return acqf if isinstance(acqf, AcquisitionFunction) else str_to_acqf(acqf)
