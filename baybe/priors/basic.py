"""A collection of common prior distributions."""

from typing import Any

from attrs import define, field
from attrs.validators import gt

from baybe.priors.base import Prior
from baybe.utils.validation import finite_float


@define(frozen=True)
class GammaPrior(Prior):
    """A Gamma prior parameterized by concentration and rate."""

    concentration: float = field(converter=float, validator=[finite_float, gt(0.0)])
    """The concentration."""

    rate: float = field(converter=float, validator=[finite_float, gt(0.0)])
    """The rate."""


@define(frozen=True)
class HalfCauchyPrior(Prior):
    """A Half-Cauchy prior parameterized by a scale."""

    scale: float = field(converter=float, validator=[finite_float, gt(0.0)])
    """The scale."""


@define(frozen=True)
class NormalPrior(Prior):
    """A Normal prior parameterized by location and scale."""

    loc: float = field(converter=float, validator=finite_float)
    """The location (mu)."""

    scale: float = field(converter=float, validator=[finite_float, gt(0.0)])
    """The scale (sigma)."""


@define(frozen=True)
class HalfNormalPrior(Prior):
    """A Half-Normal prior parameterized by a scale."""

    scale: float = field(converter=float, validator=[finite_float, gt(0.0)])
    """The scale (sigma)."""


@define(frozen=True)
class LogNormalPrior(Prior):
    """A Log-Normal prior parameterized by location and scale."""

    loc: float = field(converter=float, validator=finite_float)
    """The location (mu)."""

    scale: float = field(converter=float, validator=[finite_float, gt(0.0)])
    """The scale (sigma)."""


@define(frozen=True)
class SmoothedBoxPrior(Prior):
    """A Smoothed-Box prior parameterized by a, b and sigma."""

    a: float = field(converter=float, validator=finite_float)
    """The left/lower bound."""

    b: float = field(converter=float, validator=finite_float)
    """The right/upper bound."""

    sigma: float = field(
        converter=float, default=0.01, validator=[finite_float, gt(0.0)]
    )
    """The scale."""

    @b.validator
    def _validate_order(self, _: Any, b: float) -> None:  # noqa: DOC101, DOC103
        """Validate the order of both bounds.

        Raises:
            ValueError: If b is not larger than a.
        """
        if b <= self.a:
            raise ValueError(
                f"For {self.__class__.__name__}, the upper bound `b` (provided: {b}) "
                f"must be larger than the lower bound `a` (provided: {self.a})."
            )
