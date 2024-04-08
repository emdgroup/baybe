"""Utilities for acquisition functions."""
import warnings
from typing import Union

from baybe.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from baybe.acquisition.base import AcquisitionFunction
from baybe.utils.basic import get_subclasses


def str_to_acqf(name: str, /) -> AcquisitionFunction:
    """Create an ACQF object from a given ACQF name."""
    if name == "VarUCB":
        warnings.warn(
            "The use of `VarUCB` is deprecated and will be disabled in a future "
            "version. The get the same outcome, use the new UCB class instead with a "
            "beta of 100.0.",
            DeprecationWarning,
        )
        return UpperConfidenceBound(beta=100.0)
    elif name == "qVarUCB":
        warnings.warn(
            "The use of `qVarUCB` is deprecated and will be disabled in a future "
            "version. The get the same outcome, use the new qUCB class instead with a "
            "beta of 100.0.",
            DeprecationWarning,
        )
        return qUpperConfidenceBound(beta=100.0)

    acqfs = get_subclasses(AcquisitionFunction)
    acqf = next(
        (acqf for acqf in acqfs if name in (acqf.__name__, acqf._abbreviation)), None
    )
    if acqf is None:
        raise ValueError(
            f"{name} is not a recognized name or abbreviation for a BayBE acquisition "
            f"function."
        )
    return acqf()


def convert_acqf(acqf: Union[AcquisitionFunction, str], /) -> AcquisitionFunction:
    """Convert an ACQF name into an ACQF object (with ACQF object passthrough)."""
    return acqf if isinstance(acqf, AcquisitionFunction) else str_to_acqf(acqf)
