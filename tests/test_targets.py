"""Target tests."""

import numpy as np
import pytest
from pytest import param

from baybe.exceptions import IncompatibilityError
from baybe.targets.numerical import NumericalTarget as ModernTarget
from baybe.transformations import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    ExponentialTransformation,
    IdentityTransformation,
    LogarithmicTransformation,
    PowerTransformation,
    TriangularTransformation,
    TwoSidedLinearTransformation,
)
from baybe.utils.dataframe import to_tensor
from baybe.utils.interval import Interval

# Constants
c = 1

# Transformations
id = IdentityTransformation()
clamp = ClampingTransformation(min=2, max=5)
aff = AffineTransformation(factor=2, shift=1)
ts_v = TwoSidedLinearTransformation(slope_left=-4, slope_right=5, midpoint=2)
ts_l = TwoSidedLinearTransformation(slope_left=2, slope_right=5, midpoint=2)
ts_n = TwoSidedLinearTransformation(slope_left=4, slope_right=-5, midpoint=2)
bell = BellTransformation(center=c, sigma=2)
abs = AbsoluteTransformation()
tri = TriangularTransformation(cutoffs=(2, 8), peak=4)
log = LogarithmicTransformation()
exp = ExponentialTransformation()
pow_even = PowerTransformation(exponent=2)
pow_odd = PowerTransformation(exponent=3)
chain = ChainedTransformation([abs, clamp, aff])

# Transformed values
aff_0 = aff(to_tensor(0))
aff_1 = aff(to_tensor(1))
bell_off_1 = bell(to_tensor(bell.center + 1))
bell_off_01 = bell(to_tensor(bell.center + 0.1))


@pytest.mark.parametrize(
    ("transformation", "bounds", "expected"),
    [
        param(id, (None, None), (None, None), id="id_open"),
        param(id, (0, None), (0, None), id="id_half_open"),
        param(id, (0, 1), (0, 1), id="id_finite"),
        param(clamp, (0, 1), (2, 2), id="clamp_left_outside"),
        param(clamp, (0, 3), (2, 3), id="clamp_left_overlapping"),
        param(clamp, (3, 4), (3, 4), id="clamp_overlapping"),
        param(clamp, (4, 6), (4, 5), id="clamp_right_overlapping"),
        param(clamp, (6, 7), (5, 5), id="clamp_right_outside"),
        param(aff, (None, None), (None, None), id="affine_open"),
        param(aff, (0, None), (aff_0, None), id="affine_half_open"),
        param(aff, (0, 1), (aff_0, aff_1), id="affine_finite"),
        param(ts_v, (1, 3), (0, 5), id="ts_v_with_center"),
        param(ts_v, (3, 4), (5, 10), id="ts_v_no_center"),
        param(ts_l, (1, 3), (-2, 5), id="ts_l_with_center"),
        param(ts_l, (3, 4), (5, 10), id="ts_l_no_center"),
        param(ts_n, (1, 3), (-5, 0), id="ts_n_with_center"),
        param(ts_n, (3, 4), (-10, -5), id="ts_n_no_center"),
        param(bell, (None, None), (0, 1), id="bell_open"),
        param(bell, (c, None), (0, 1), id="bell_half_open"),
        param(bell, (c + 1, None), (0, bell_off_1), id="bell_half_open_reduced"),
        param(bell, (c + 0.1, c + 1), (bell_off_1, bell_off_01), id="bell_no_center"),
        param(bell, (c - 0.1, c + 1), (bell_off_1, 1), id="bell_with_center"),
        param(abs, (None, None), (0, None), id="abs_open"),
        param(abs, (0, None), (0, None), id="abs_half_open"),
        param(abs, (0, 1), (0, 1), id="abs_0_1"),
        param(abs, (0.1, 1), (0.1, 1), id="abs_0.1_1"),
        param(abs, (-0.1, 1), (0, 1), id="abs_-0.1_1"),
        param(tri, (None, None), (0, 1), id="tri_open"),
        param(tri, (0, 1), (0, 0), id="tri_left_outside"),
        param(tri, (0, 3), (0, 0.5), id="tri_left_overlapping"),
        param(tri, (3, 4), (0.5, 1), id="tri_overlapping"),
        param(tri, (5, 11), (0, 0.75), id="tri_right_overlapping"),
        param(tri, (11, 12), (0, 0), id="tri_right_outside"),
        param(log, (0, None), (None, None), id="log_open"),
        param(log, (1, np.exp(1)), (0, 1), id="log_unit"),
        param(exp, (None, None), (0, None), id="exp_open"),
        param(exp, (0, 1), (1, np.exp(1)), id="exp_unit"),
        param(pow_even, (1, None), (1, None), id="pow_even_open_right"),
        param(pow_even, (None, -1), (1, None), id="pow_even_open_left"),
        param(pow_even, (-2, 3), (0, 9), id="pow_even_with_center"),
        param(pow_even, (1, 3), (1, 9), id="pow_even_without_center"),
        param(pow_odd, (1, None), (1, None), id="pow_odd_open_right"),
        param(pow_odd, (None, -1), (None, -1), id="pow_odd_open_left"),
        param(pow_odd, (-2, 3), (-8, 27), id="pow_odd_with_center"),
        param(pow_odd, (1, 3), (1, 27), id="pow_odd_without_center"),
        param(chain, (-0.1, 3), (5, 7), id="chain"),
    ],
)
def test_image_computation(transformation, bounds, expected):
    """The image of a transformation is computed correctly."""
    bounds = (None, None) if bounds is None else bounds
    assert transformation.get_image(bounds) == Interval.create(expected)


def test_target_normalization():
    """Target normalization works as expected."""
    t = ModernTarget("t")
    with pytest.raises(IncompatibilityError, match="Only bounded targets"):
        t.normalize()
    assert t.clamp(-2, 4).get_image() == Interval(-2, 4)
    assert t.clamp(-2, 4).normalize().get_image() == Interval(0, 1)
