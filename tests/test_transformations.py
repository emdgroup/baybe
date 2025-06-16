"""Transformation tests."""

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_series_equal

from baybe.targets.numerical import NumericalTarget as ModernTarget
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
    TwoSidedLinearTransformation,
)
from baybe.utils.dataframe import to_tensor


def sample_input() -> pd.Series:
    return pd.Series(np.linspace(-10, 10, 20))


@pytest.fixture
def series() -> pd.Series:
    return sample_input()


@pytest.mark.parametrize("chained_first", [True, False])
def test_transformation_chaining(chained_first):
    """Transformation chaining and flattening works as expected."""
    t1 = AffineTransformation()
    t2 = ClampingTransformation(0, 1)
    t3 = AbsoluteTransformation()
    c = ChainedTransformation([t1, t2])

    expected = ChainedTransformation([t1, t2, t3, t1, t2])
    if chained_first:
        actual = (c + t3) + c
    else:
        actual = c + (t3 + c)
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

    t1 = t * 2 + 3 + t
    assert t1 == AffineTransformation(factor=2, shift=3)

    t2 = (t + 3) * 2 + t
    assert t2 == AffineTransformation(factor=2, shift=3, shift_first=True)


def test_identity_transformation_chaining():
    """Chaining an identity transformation has no effect."""
    t1 = IdentityTransformation()
    t2 = ClampingTransformation(0, 1)

    assert t1.chain(t2) == t2
    assert t2.chain(t1) == t2


def test_generic_transformation(series):
    """Torch callables can be used to specify custom transformations."""
    t1 = ModernTarget("t", AbsoluteTransformation())
    t2 = ModernTarget("t", CustomTransformation(torch.abs))
    t3 = ModernTarget("t", torch.abs)
    t4 = ModernTarget(
        "t", torch.abs(IdentityTransformation())
    )  # explicitly trigger __torch_function__

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))
    assert_series_equal(transformed, t4.transform(series))


def test_transformation_addition(series):
    """Adding transformations results in chaining/shifting."""
    t1 = ModernTarget(
        "t",
        ChainedTransformation(
            [AbsoluteTransformation(), AffineTransformation(shift=1)]
        ),
    )
    t2 = ModernTarget("t", AbsoluteTransformation() + 1)
    t3 = ModernTarget("t", AbsoluteTransformation() + AffineTransformation(shift=1))

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))


def test_transformation_multiplication(series):
    """Multiplying transformations results in scaling."""
    t1 = ModernTarget(
        "t",
        ChainedTransformation(
            [AbsoluteTransformation(), AffineTransformation(factor=2)]
        ),
    )
    t2 = ModernTarget("t", AbsoluteTransformation() * 2)
    t3 = ModernTarget("t", AbsoluteTransformation() + AffineTransformation(factor=2))

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))


def test_torch_overloading(series):
    """Transformations can be passed to torch callables for chaining."""
    t1 = ModernTarget(
        "t", AffineTransformation(factor=2) + CustomTransformation(torch.abs)
    )
    t2 = ModernTarget("t", torch.abs(AffineTransformation(factor=2)))
    assert_series_equal(t1.transform(series), t2.transform(series))


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


def test_affine_transformation_operation_order(series):
    """The alternative construction yields the correct equivalent transformation."""
    t1 = AffineTransformation(factor=2, shift=10, shift_first=False)
    t2 = AffineTransformation(factor=2, shift=5, shift_first=True)
    tensor = to_tensor(series.values)

    assert t1 == t2
    assert t1(tensor).equal(t2(tensor))


def test_affine_transformation_chaining(series):
    """Chaining affine transformations compressed them."""
    t1 = AffineTransformation(factor=3, shift=2)
    t2 = AffineTransformation(factor=4, shift=3)

    chained = t1 + t2
    chained_factor = t2.factor * t1.factor
    chained_shift = t2.factor * t1.shift + t2.shift
    expected = AffineTransformation(factor=chained_factor, shift=chained_shift)

    tensor = to_tensor(series.values)
    assert chained == expected
    assert chained(tensor).equal(expected(tensor))
