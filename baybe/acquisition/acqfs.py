"""Available acquisition functions."""

from typing import ClassVar

from attrs import define, field
from attrs.validators import ge

from baybe.acquisition.base import AcquisitionFunction


@define(frozen=True)
class qExpectedImprovement(AcquisitionFunction):
    """Monte Carlo based expected improvement."""

    _abbreviation: ClassVar[str] = "qEI"


@define(frozen=True)
class qUpperConfidenceBound(AcquisitionFunction):
    """Monte Carlo based upper confidence bound."""

    _abbreviation: ClassVar[str] = "qUCB"

    beta: float = field(converter=float, validator=ge(0.0))
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """
