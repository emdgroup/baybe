"""Transformation tests."""

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_series_equal
from pytest import param

from baybe.targets.numerical import NumericalTarget
from baybe.transformations import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    CustomTransformation,
    ExponentialTransformation,
    IdentityTransformation,
    LogarithmicTransformation,
    PowerTransformation,
    TriangularTransformation,
    TwoSidedAffineTransformation,
)
from baybe.transformations.composite import (
    AdditiveTransformation,
    MultiplicativeTransformation,
)
from baybe.utils.dataframe import to_tensor
from baybe.utils.interval import Interval


def sample_input() -> pd.Series:
    return pd.Series(np.linspace(-10, 10, 20))


@pytest.fixture
def series() -> pd.Series:
    return sample_input()


@pytest.fixture
def tensor() -> torch.Tensor:
    return to_tensor(sample_input())


@pytest.mark.parametrize("chained_first", [True, False])
def test_transformation_chaining(chained_first):
    """Transformation chaining and flattening works as expected."""
    t1 = AffineTransformation()
    t2 = ClampingTransformation(0, 1)
    t3 = AbsoluteTransformation()
    c = ChainedTransformation([t1, t2])

    expected = ChainedTransformation([t1, t2, t3, t1, t2])
    if chained_first:
        actual = (c | t3) | c
    else:
        actual = c | (t3 | c)
    assert actual == expected


def test_unnesting_chained_transformations():
    """Chaining chained transformations flattens them."""
    t = PowerTransformation
    c = ChainedTransformation(
        [
            ChainedTransformation(
                [
                    ChainedTransformation([t(2), t(3)]),
                    ChainedTransformation([t(4), t(5)]),
                ]
            ),
            ChainedTransformation(
                [
                    ChainedTransformation([t(6), t(7)]),
                    ChainedTransformation([t(8), t(9)]),
                ]
            ),
        ],
    )
    assert c == ChainedTransformation([t(2), t(3), t(4), t(5), t(6), t(7), t(8), t(9)])


def test_affine_transformation_compression():
    """Compression of affine transformations works as expected."""
    t = IdentityTransformation()

    t1 = t * 2 + 3 | t
    assert t1 == AffineTransformation(factor=2, shift=3)

    t2 = (t + 3) * 2 | t
    assert t2 == AffineTransformation(factor=2, shift=3, shift_first=True)


def test_identity_transformation_chaining():
    """Chaining an identity transformation has no effect."""
    t1 = IdentityTransformation()
    t2 = ClampingTransformation(0, 1)

    assert t1.chain(t2) == t2
    assert t2.chain(t1) == t2


def test_generic_transformation(series):
    """Torch callables can be used to specify custom transformations."""
    t1 = NumericalTarget("t", AbsoluteTransformation())
    t2 = NumericalTarget("t", CustomTransformation(torch.abs))
    t3 = NumericalTarget("t", torch.abs)

    # explicitly trigger __torch_function__
    t4 = NumericalTarget("t", torch.abs(IdentityTransformation()))

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))
    assert_series_equal(transformed, t4.transform(series))


def test_transformation_addition(tensor):
    """Adding transformations results in chaining/shifting."""
    t1 = ChainedTransformation(
        [AbsoluteTransformation(), AffineTransformation(shift=1)]
    )
    t2 = AbsoluteTransformation() + 1
    t3 = AbsoluteTransformation() | AffineTransformation(shift=1)
    t4 = AbsoluteTransformation() - (-1)

    transformed = t1(tensor)
    assert torch.equal(transformed, t2(tensor))
    assert torch.equal(transformed, t3(tensor))
    assert torch.equal(transformed, t4(tensor))


def test_transformation_multiplication(tensor):
    """Multiplying transformations results in scaling."""
    t1 = ChainedTransformation(
        [AbsoluteTransformation(), AffineTransformation(factor=2)]
    )
    t2 = AbsoluteTransformation() * 2
    t3 = AbsoluteTransformation() | AffineTransformation(factor=2)
    t4 = AbsoluteTransformation() / 0.5

    transformed = t1(tensor)
    assert torch.equal(transformed, t2(tensor))
    assert torch.equal(transformed, t3(tensor))
    assert torch.equal(transformed, t4(tensor))


def test_torch_overloading(tensor):
    """Transformations can be passed to torch callables for chaining."""
    t1 = AffineTransformation(factor=2) | CustomTransformation(torch.abs)
    t2 = torch.abs(AffineTransformation(factor=2))
    torch.equal(t1(tensor), t2(tensor))


def test_invalid_torch_overloading():
    """Chaining torch callables works only with one-argument callables."""
    with pytest.raises(ValueError, match="enters as the only"):
        torch.add(AbsoluteTransformation(), 2)


# Constants
c = 1

# Transformations
id = IdentityTransformation()
clamp = ClampingTransformation(min=2, max=5)
aff = AffineTransformation(factor=2, shift=1)
ts_v = TwoSidedAffineTransformation(slope_left=-4, slope_right=5, midpoint=2)
ts_l = TwoSidedAffineTransformation(slope_left=2, slope_right=5, midpoint=2)
ts_n = TwoSidedAffineTransformation(slope_left=4, slope_right=-5, midpoint=2)
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
    bounds = Interval.create(bounds)
    assert transformation.get_image(bounds) == Interval.create(expected)


def test_affine_transformation_operator_order(series):
    """The alternative construction yields the correct equivalent transformation."""
    t1 = AffineTransformation(factor=2, shift=10, shift_first=False)
    t2 = AffineTransformation(factor=2, shift=5, shift_first=True)
    tensor = to_tensor(series.values)

    assert t1 == t2
    assert t1(tensor).equal(t2(tensor))


def test_identity_affine_transformation_chaining():
    """Chaining an identity affine transformation has no effect."""
    assert (
        AffineTransformation() | ExponentialTransformation()
        == ExponentialTransformation()
    )


def test_affine_transformation_chaining(series):
    """Chaining affine transformations compressed them."""
    t1 = AffineTransformation(factor=3, shift=2)
    t2 = AffineTransformation(factor=4, shift=3)

    chained = t1 | t2
    chained_factor = t2.factor * t1.factor
    chained_shift = t2.factor * t1.shift + t2.shift
    expected = AffineTransformation(factor=chained_factor, shift=chained_shift)

    tensor = to_tensor(series.values)
    assert chained == expected
    assert chained(tensor).equal(expected(tensor))


@pytest.mark.parametrize(
    ("t1", "t2"),
    [
        param(exp, ChainedTransformation([exp]), id="one-element-chain"),
        param(IdentityTransformation(), AffineTransformation(), id="affine-identity"),
    ],
)
def test_transformation_equality(t1, t2):
    """Length-one chained transformations are equivalent to their wrapped element."""
    assert t1 == t2
    assert t2 == t1


@pytest.mark.parametrize(
    "transformation",
    [
        param(AffineTransformation(factor=0.0), id="affine"),
        param(TwoSidedAffineTransformation(0, 0), id="two_sided"),
        param(ClampingTransformation(0, 0), id="clamping"),
    ],
)
def test_degenerate_transformations(transformation):
    """Degenerate transformations produce proper (non-nan) outputs."""
    assert transformation.get_image() == Interval(0, 0)
    assert transformation.get_image((20, None)) == Interval(0, 0)


def test_additive_transformation(tensor):
    """Additive transformations compute the sum of two transformations."""
    t1 = ExponentialTransformation()
    t2 = PowerTransformation(exponent=2)

    expected = t1(tensor) + t2(tensor)
    comp1 = AdditiveTransformation([t1, t2])
    comp2 = t1 + t2

    assert comp1 == comp2
    assert torch.equal(comp1(tensor), expected)
    assert torch.equal(comp2(tensor), expected)


def test_multiplicative_transformation(tensor):
    """Multiplicative transformations compute the product of two transformations."""
    t1 = ExponentialTransformation()
    t2 = PowerTransformation(exponent=2)

    expected = t1(tensor) * t2(tensor)
    comp1 = MultiplicativeTransformation([t1, t2])
    comp2 = t1 * t2

    assert comp1 == comp2
    assert torch.equal(comp1(tensor), expected)
    assert torch.equal(comp2(tensor), expected)
