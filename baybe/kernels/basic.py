"""Collection of basic kernels."""

from typing import Optional

from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import in_, instance_of
from attrs.validators import optional as optional_v

from baybe.kernels.base import Kernel
from baybe.priors.base import Prior
from baybe.utils.conversion import fraction_to_float
from baybe.utils.validation import finite_float


@define(frozen=True)
class MaternKernel(Kernel):
    """A Matern kernel using a smoothness parameter."""

    nu: float = field(
        converter=fraction_to_float, validator=in_([0.5, 1.5, 2.5]), default=2.5
    )
    """A smoothness parameter.

    Only takes the values 0.5, 1.5 or 2.5. Larger values yield smoother interpolations.
    """

    lengthscale_prior: Optional[Prior] = field(
        default=None, validator=optional_v(instance_of(Prior))
    )
    """An optional prior on the kernel lengthscale."""

    lengthscale_initial_value: Optional[float] = field(
        default=None, converter=optional_c(float), validator=optional_v(finite_float)
    )
    """An optional initial value for the kernel lengthscale."""
