"""Target tests."""

import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from baybe.exceptions import IncompatibilityError
from baybe.targets.numerical import NumericalTarget
from baybe.transformations.core import AffineTransformation
from baybe.utils.interval import Interval


def test_target_addition():
    """Target addition appends a shifting transformation."""
    t1 = NumericalTarget("t") + 1
    assert t1._transformation == AffineTransformation(shift=1)


def test_target_multiplication():
    """Target multiplication appends a scaling transformation."""
    t1 = NumericalTarget("t") * 2
    assert t1._transformation == AffineTransformation(scale=2)


def test_target_inversion():
    """Double inversion cancels out."""
    series = pd.Series([-2, 0, 3], dtype=float)
    t = NumericalTarget("t")
    ti = t.invert()
    tii = ti.invert()

    transformed = t.transform(series)
    assert tii == t
    assert_series_equal(transformed, -ti.transform(series))
    assert_series_equal(transformed, tii.transform(series))


def test_target_normalization():
    """Target normalization works as expected."""
    t = NumericalTarget("t")
    with pytest.raises(IncompatibilityError, match="Only bounded targets"):
        t.normalize()
    assert t.clamp(-2, 4).get_image() == Interval(-2, 4)
    assert t.clamp(-2, 4).normalize().get_image() == Interval(0, 1)
