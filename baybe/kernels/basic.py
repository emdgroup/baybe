"""Collection of kernels."""

from fractions import Fraction
from typing import Optional, Union

from attrs import define, field
from attrs.validators import in_

from baybe.kernels.base import Kernel
from baybe.kernels.priors.base import Prior


def _convert_fraction(value: Union[str, float, Fraction], /) -> float:
    """Convert the provided value into a float.

    Args:
        value: The parameter that should be converted.

    Returns:
        The float representation of the given input.

    Raises:
        ValueError: If the input was provided as string but could not be interpreted as
            fraction.
    """
    if isinstance(value, str):
        try:
            value = Fraction(value)
        except ValueError as err:
            raise ValueError(
                f"The provided input '{value}' could not be interpreted as a fraction."
            ) from err
    return float(value)


@define(frozen=True)
class MaternKernel(Kernel):
    """A Matern kernel using a smoothness parameter."""

    nu: float = field(
        converter=_convert_fraction, validator=in_([0.5, 1.5, 2.5]), default=2.5
    )
    """A smoothness parameter.

    Only takes the values 0.5, 1.5 or 2.5. Larger values yield smoother interpolations.
    """

    lengthscale_prior: Optional[Prior] = field(default=None)
    """An optional prior on the kernel lengthscale."""

    def to_gpytorch(self, *args, **kwargs):  # noqa: D102
        # See base class.
        # Provide the transformed lengthscale as keyword argument.
        return super().to_gpytorch(
            lengthscale_prior=self.lengthscale_prior.to_gpytorch()
        )


@define(frozen=True)
class ScaleKernel(Kernel):
    """A kernel for decorating existing kernels with an outputscale."""

    base_kernel: Kernel = field(factory=MaternKernel)
    """The base kernel that is being decorated."""

    outputscale_prior: Optional[Prior] = field(default=None, kw_only=True)
    """An optional prior on the output scale."""
