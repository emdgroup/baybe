"""Utilities for acquisition functions."""
import warnings
from typing import Union

from baybe.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from baybe.acquisition.base import AcquisitionFunction
from baybe.utils.basic import get_subclasses


def str_to_acqf(name: str, /) -> AcquisitionFunction:
    """Create an ACQF object from a given ACQF name."""
    UCB_DEPRECATIONS = {
        "VarUCB": UpperConfidenceBound,
        "qVarUCB": qUpperConfidenceBound,
    }
    if name in UCB_DEPRECATIONS:
        warnings.warn(
            f"The use of `{name}` is deprecated and will be disabled in a "
            f"future version. The get the same outcome, use the new "
            f"{UCB_DEPRECATIONS[name].__name__} class instead with a beta of 100.0.",
            DeprecationWarning,
        )
        return UCB_DEPRECATIONS[name](beta=100.0)

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
