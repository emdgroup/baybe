"""Utilities for acquisition functions."""

from typing import Literal, Union

from baybe.acquisition.base import AcquisitionFunction
from baybe.utils.basic import get_subclasses

_ACQF_NAMES = Literal[
    "PM", "PI", "EI", "UCB", "qPI", "qEI", "qUCB", "VarUCB", "qVarUCB"
]


def str_to_acqf(name: _ACQF_NAMES, /) -> AcquisitionFunction:
    """Create an ACQF object from a given ACQF name."""
    acqfs = get_subclasses(AcquisitionFunction)
    return next(acqf for acqf in acqfs if acqf._abbreviation == name)()


def convert_acqf(
    acqf: Union[AcquisitionFunction, _ACQF_NAMES], /
) -> AcquisitionFunction:
    """Convert an ACQF name into an ACQF object (with ACQF object passthrough)."""
    return acqf if isinstance(acqf, AcquisitionFunction) else str_to_acqf(acqf)
