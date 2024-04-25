"""Priors that can be used for kernels."""
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
