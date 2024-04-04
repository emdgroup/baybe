"""Collection of kernels."""

from attrs import define, field
from attrs.validators import in_

from baybe.kernels.base import Kernel


@define
class MaternKernel(Kernel):
    """A Matern kernel using a smoothness parameter."""

    nu: float = field(converter=float, validator=in_([0.5, 1.5, 2.5]), default=2.5)
    """A smoothness parameter.

    Only takes the values 0.5, 1.5 or 2.5. Larger values yield smoother interpolations.
    """
