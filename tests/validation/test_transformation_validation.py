"""Validation tests for transformations."""

import numpy as np
import pytest
from pytest import param

from baybe.transformations import (
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    PowerTransformation,
    TriangularTransformation,
    TwoSidedAffineTransformation,
)


@pytest.mark.parametrize(
    ("min", "max", "match"),
    [
        param(np.nan, 0, "cannot be 'nan'", id="nan_min"),
        param(0, np.nan, "cannot be 'nan'", id="nan_max"),
        param("", 0, "could not convert", id="type_min"),
        param(0, "", "could not convert", id="type_max"),
        param(1, 0, "must be greater than", id="min_greater_than_max"),
    ],
)
def test_invalid_clamping_transformation(min, max, match):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(ValueError, match=match):
        ClampingTransformation(min, max)


@pytest.mark.parametrize(
    ("factor", "shift"),
    [
        param(np.inf, 0, id="inf_factor"),
        param(0, np.inf, id="inf_shift"),
        param(np.nan, 0, id="nan_factor"),
        param(0, np.nan, id="nan_shift"),
        param("", 0, id="type_factor"),
        param(0, "", id="type_shift"),
    ],
)
def test_invalid_affine_transformation(factor, shift):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(ValueError):
        AffineTransformation(factor, shift)


@pytest.mark.parametrize(
    ("slope_left", "slope_right", "center"),
    [
        param(np.inf, 1, 0, id="inf_slope_left"),
        param(0, np.inf, 0, id="inf_slope_right"),
        param(0, 0, np.inf, id="inf_center"),
        param(np.nan, 1, 0, id="nan_slope_left"),
        param(0, np.nan, 0, id="nan_slope_right"),
        param(0, 0, np.nan, id="nan_center"),
        param("", 1, 0, id="type_slope_left"),
        param(0, "", 0, id="type_slope_right"),
        param(0, 0, "", id="type_center"),
    ],
)
def test_invalid_two_sided_affine_transformation(slope_left, slope_right, center):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(ValueError):
        TwoSidedAffineTransformation(slope_left, slope_right, center)


@pytest.mark.parametrize(
    ("center", "sigma"),
    [
        param(np.inf, 1, id="inf_center"),
        param(0, np.inf, id="inf_sigma"),
        param(np.nan, 1, id="nan_center"),
        param(0, np.nan, id="nan_sigma"),
        param(0, 0, id="zero_sigma"),
        param(0, -1, id="negative_sigma"),
        param("", 1, id="type_center"),
        param(0, "", id="type_sigma"),
    ],
)
def test_invalid_bell_transformation(center, sigma):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(ValueError):
        BellTransformation(center, sigma)


@pytest.mark.parametrize(
    ("cutoffs", "peak", "match"),
    [
        param([2, 0], 1, "cannot be smaller", id="wrong_order_cutoffs"),
        param([0, np.inf], 1, "must be bounded", id="inf_cutoff"),
        param([0, 1], np.nan, "between the specified cutoff", id="outside_cutoffs"),
        param([0, 1], "", "could not convert", id="type_peak"),
    ],
)
def test_invalid_triangular_transformation(cutoffs, peak, match):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(ValueError, match=match):
        TriangularTransformation(cutoffs, peak)


@pytest.mark.parametrize(
    ("exponent", "error"),
    [
        param(np.inf, TypeError, id="inf"),
        param(np.nan, TypeError, id="nan"),
        param("", TypeError, id="type"),
        param(1.5, TypeError, id="non-int"),
        param(1, ValueError, id="trivial"),
    ],
)
def test_invalid_power_transformation(exponent, error):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(error):
        PowerTransformation(exponent)


@pytest.mark.parametrize(
    ("transformations", "error"),
    [
        param([], ValueError, id="empty"),
        param([None], TypeError, id="type"),
    ],
)
def test_invalid_chained_transformation(transformations, error):
    """Providing invalid arguments raises an exception."""
    with pytest.raises(error):
        ChainedTransformation(transformations)
