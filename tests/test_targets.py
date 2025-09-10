"""Target tests."""

import operator as op

import pandas as pd
import pytest
from attrs import evolve
from pandas.testing import assert_series_equal
from pytest import param

from baybe.exceptions import IncompatibilityError
from baybe.targets.numerical import NumericalTarget
from baybe.transformations.basic import (
    AffineTransformation,
    ClampingTransformation,
    ExponentialTransformation,
)
from baybe.utils.interval import Interval


def test_target_addition():
    """Target addition appends a shifting transformation."""
    t1 = NumericalTarget("t") + 1
    t2 = NumericalTarget("t") - (-1)
    assert t1 == t2
    assert t1.transformation == AffineTransformation(shift=1)


def test_target_multiplication():
    """Target multiplication appends a scaling transformation."""
    t1 = NumericalTarget("t") * 2
    t2 = NumericalTarget("t") / 0.5
    assert t1 == t2
    assert t1.transformation == AffineTransformation(factor=2)


def test_target_negation():
    """Double negation cancels out."""
    series = pd.Series([-2, 0, 3], dtype=float)
    t = NumericalTarget("t")
    ti = t.negate()
    tii = ti.negate()

    transformed = t.transform(series)
    assert tii == t
    assert_series_equal(transformed, -ti.transform(series))
    assert_series_equal(transformed, tii.transform(series))


@pytest.mark.parametrize("minimize", [True, False])
def test_target_normalization(monkeypatch, minimize):
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

    t_raw = NumericalTarget("t", minimize=minimize)
    with pytest.raises(IncompatibilityError, match="Only bounded targets"):
        t_raw.normalize()

    t = t_raw.clamp(-2, 4)
    assert not t.is_normalized
    assert t.get_image() == Interval(-2, 4)
    assert t.get_codomain() == Interval(-4, 8)
    assert t.normalize().is_normalized
    assert t.normalize().get_image() == Interval(0, 1)
    assert t.normalize().get_codomain() != Interval(0, 1)


@pytest.mark.parametrize(
    ("target", "transformed_value"),
    [
        param(NumericalTarget.match_bell("t", 2, 1), 1, id="bell"),
        param(NumericalTarget.match_power("t", 2, 2), 0, id="power"),
        param(NumericalTarget.match_quadratic("t", 2), 0, id="quadratic"),
        param(NumericalTarget.match_absolute("t", 2), 0, id="absolute"),
        param(NumericalTarget.match_triangular("t", 2, width=40), 1, id="triangular"),
    ],
)
def test_match_constructors(target, transformed_value):
    """Larger distance to match values yields "worse" transformed values."""
    delta = [0, 0.01, -0.02, 0.1, -0.2, 1, -2, 10, -20]
    match_value = 2
    series = pd.Series(delta) + match_value

    # On the target level, "worse" can mean "larger" or "smaller",
    # depending on the minimization flag
    t1 = target.transform(series)
    operator = op.gt if target.minimize else op.lt
    assert t1[0] == transformed_value
    assert operator(t1.diff().dropna(), 0).all()

    # Objectives, on the other hand, are always to be maximized.
    # Hence, "worse" means "smaller".
    t2 = target.to_objective().transform(series.to_frame(name=target.name)).squeeze()
    assert (t2.diff().dropna() < 0).all()


@pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])
def test_valid_combination(operator):
    """Targets with the same attributes (except transformation) can be combined."""
    t = NumericalTarget("t")
    operator(t, evolve(t, transformation=ExponentialTransformation()))
