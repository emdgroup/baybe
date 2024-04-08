"""Available acquisition functions."""

from typing import ClassVar

from attrs import define, field
from attrs.validators import ge

from baybe.acquisition.base import AcquisitionFunction


@define(frozen=True)
class PosteriorMean(AcquisitionFunction):
    """Posterior Mean."""

    _abbreviation: ClassVar[str] = "PM"


@define(frozen=True)
class qExpectedImprovement(AcquisitionFunction):
    """Monte Carlo based expected improvement."""

    _abbreviation: ClassVar[str] = "qEI"


@define(frozen=True)
class ExpectedImprovement(AcquisitionFunction):
    """Analytical expected improvement."""

    _abbreviation: ClassVar[str] = "EI"


@define(frozen=True)
class qProbabilityOfImprovement(AcquisitionFunction):
    """Monte Carlo based probability of improvement."""

    _abbreviation: ClassVar[str] = "qPI"


@define(frozen=True)
class ProbabilityOfImprovement(AcquisitionFunction):
    """Analytical probability of improvement."""

    _abbreviation: ClassVar[str] = "PI"


@define(frozen=True)
class qUpperConfidenceBound(AcquisitionFunction):
    """Monte Carlo based upper confidence bound."""

    _abbreviation: ClassVar[str] = "qUCB"

    beta: float = field(converter=float, validator=ge(0.0), default=0.2)
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """


@define(frozen=True)
class UpperConfidenceBound(AcquisitionFunction):
    """Analytical upper confidence bound."""

    _abbreviation: ClassVar[str] = "UCB"

    beta: float = field(converter=float, validator=ge(0.0), default=0.2)
    """Trade-off parameter for mean and variance.

    A value of zero makes the acquisition mechanism consider the posterior predictive
    mean only, resulting in pure exploitation. Higher values shift the focus more and
    more toward exploration.
    """
