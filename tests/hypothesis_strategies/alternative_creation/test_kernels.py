"""Test alternative ways of creation not considered in the strategies."""

from fractions import Fraction

import pytest

from baybe.kernels import MaternKernel


@pytest.mark.parametrize("nu", [Fraction(1, 2), Fraction(3, 2), Fraction(5, 2)])
def test_different_fractions(nu: Fraction):
    """The nu parameter can be a Fraction."""
    MaternKernel(nu=nu)


@pytest.mark.parametrize("nu", ["1/2", "3/2", "5/2"])
def test_fraction_string_creation(nu: str):
    """The nu parameter can be a str representing a fraction."""
    MaternKernel(nu=nu)


@pytest.mark.parametrize("nu", ["0.5", "1.5", "2.5"])
def test_float_string_creation(nu: str):
    """The nu parameter can be a str representing a float."""
    MaternKernel(nu=nu)
