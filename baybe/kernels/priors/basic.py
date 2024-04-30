"""Priors that can be used for kernels."""
from typing import Any

from attrs import define, field
from attrs.validators import gt

from baybe.kernels.priors.base import Prior


@define(frozen=True)
class GammaPrior(Prior):
    """A Gamma prior parameterized by concentration and rate."""

    concentration: float = field(converter=float, validator=gt(0.0))
    """The concentration."""

    rate: float = field(converter=float, validator=gt(0.0))
    """The rate."""


@define(frozen=True)
class HalfCauchyPrior(Prior):
    """A HalfCauchyPrior parameterized by a scale."""

    scale: float = field(converter=float, validator=gt(0.0))
    """The scale."""


@define(frozen=True)
class NormalPrior(Prior):
    """A NormalPrior parameterized by location and scale."""

    loc: float = field(converter=float)
    """The location (mu)."""

    scale: float = field(converter=float, validator=gt(0.0))
    """The scale (sigma)."""


@define(frozen=True)
class HalfNormalPrior(Prior):
    """A HalfNormalPrior parameterized by a scale."""

    scale: float = field(converter=float, validator=gt(0.0))
    """The scale (sigma)."""


@define(frozen=True)
class LogNormalPrior(Prior):
    """A LogNormalPrior parameterized by location and scale."""

    loc: float = field(converter=float)
    """The location (mu)."""

    scale: float = field(converter=float, validator=gt(0.0))
    """The scale (sigma)."""


@define(frozen=True)
class SmoothedBoxPrior(Prior):
    """A SmoothedBoxPrior parameterized by a, b and sigma."""

    a: float = field(converter=float)
    """The left/lower bound."""

    b: float = field(converter=float)
    """The right/upper bound."""

    sigma: float = field(converter=float, default=0.01, validator=gt(0.0))
    """The scale."""

    @b.validator
    def _validate_order(self, _: Any, b: float) -> None:  # noqa: DOC101, DOC103
        """Validate the order of both bounds.

        Raises:
            ValueError: If b is not larger than a.
        """
        if b <= self.a:
            raise ValueError(
                f"For SmoothedBoxPrior, the upper bound `b` (provided: {b}) must be"
                f"larger than the lower bound `a` (provided: {self.a})."
            )
