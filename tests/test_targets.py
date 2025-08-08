"""Target tests."""

import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from pytest import param

from baybe.exceptions import IncompatibilityError
from baybe.targets.numerical import NumericalTarget
from baybe.transformations.basic import AffineTransformation, ClampingTransformation
from baybe.utils.interval import Interval


def test_target_addition():
    """Target addition appends a shifting transformation."""
    t1 = NumericalTarget("t") + 1
    assert t1._transformation == AffineTransformation(shift=1)


def test_target_multiplication():
    """Target multiplication appends a scaling transformation."""
    t1 = NumericalTarget("t") * 2
    assert t1._transformation == AffineTransformation(factor=2)


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


@pytest.mark.parametrize("codomain", [True, False])
def test_target_normalization(codomain: bool, monkeypatch):
    """Target normalization works as expected."""
    # We artificially create some transform whose codomain does not coincide with the
    # image but is twice as broad.
    monkeypatch.setattr(
        ClampingTransformation,
        "get_codomain",
        lambda self, interval=None: Interval.create(
            self.get_image(interval).to_ndarray() * 2
        ),
    )

    t = NumericalTarget("t")
    with pytest.raises(IncompatibilityError, match="Only bounded targets"):
        t.normalize(codomain)

    assert t.clamp(-2, 4).get_image() == Interval(-2, 4)
    assert t.clamp(-2, 4).get_codomain() == Interval(-4, 8)
    if codomain:
        assert t.clamp(-2, 4).normalize(codomain).get_image() != Interval(0, 1)
        assert t.clamp(-2, 4).normalize(codomain).get_codomain() == Interval(0, 1)
    else:
        assert t.clamp(-2, 4).normalize(codomain).get_image() == Interval(0, 1)
        assert t.clamp(-2, 4).normalize(codomain).get_codomain() != Interval(0, 1)


@pytest.mark.parametrize(
    "target",
    [
        param(NumericalTarget.match_bell("t", 2, 1), id="match"),
        param(NumericalTarget.match_power("t", 2, 2), id="power"),
        param(NumericalTarget.match_quadratic("t", 2), id="quadratic"),
        param(NumericalTarget.match_absolute("t", 2), id="absolute"),
        param(NumericalTarget.match_triangular("t", 2, width=40), id="triangular"),
    ],
)
def test_match_constructors(target):
    """Larger distance to match values yields smaller transformed values."""
    delta = [0.01, -0.02, 0.1, -0.2, 1, -2, 10, -20]
    match_value = 2

    transformed = target.transform(pd.Series(delta) + match_value)
    assert (transformed.diff().dropna() < 0).all()
