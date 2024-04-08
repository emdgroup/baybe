"""Collection of kernels."""

from fractions import Fraction
from typing import Union

from attrs import define, field
from attrs.validators import in_

from baybe.kernels.base import Kernel


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


@define
class MaternKernel(Kernel):
    """A Matern kernel using a smoothness parameter."""

    nu: float = field(
        converter=_convert_fraction, validator=in_([0.5, 1.5, 2.5]), default=2.5
    )
    """A smoothness parameter.

    Only takes the values 0.5, 1.5 or 2.5. Larger values yield smoother interpolations.
    """
