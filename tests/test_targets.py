"""Target tests."""

import operator as op

import pandas as pd
import pytest
from attrs import evolve
from pandas.testing import assert_series_equal
from pytest import param

from baybe.exceptions import IncompatibilityError
from baybe.targets import MatchMode
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


@pytest.mark.parametrize("match_mode", MatchMode)
@pytest.mark.parametrize("mismatch_instead", [False, True], ids=["match", "mismatch"])
@pytest.mark.parametrize(
    ("constructor", "kwargs", "transformed_value"),
    [
        param("match_bell", {"sigma": 1}, 1, id="bell"),
        param("match_power", {"exponent": 3}, 0, id="power"),
        param("match_quadratic", {}, 0, id="quadratic"),
        param("match_absolute", {}, 0, id="absolute"),
        param("match_triangular", {"width": 40}, 1, id="triangular"),
    ],
)
def test_match_constructors(
    constructor, kwargs, transformed_value, mismatch_instead, match_mode
):
    """Distance to match values yields expected transformed values."""
    match_value = 2
    kwargs |= {"mismatch_instead": mismatch_instead, "match_mode": match_mode}
    target = getattr(NumericalTarget, constructor)("t", match_value, **kwargs)
    delta = [0, 0.01, -0.02, 0.1, -0.2, 1, -2, 10, -20]
    series = pd.Series(delta) + match_value

    # Ensure all expected points map to the expected transformed value
    if match_mode is MatchMode.eq:
        # For "=" mode, just the first entry should map to the transformed value
        idxs = pd.Index([0])
    else:
        # For the other modes, all entries on the match side should transform to the
        # match value
        idxs = series[
            series <= match_value
            if match_mode is MatchMode.le
            else series >= match_value
        ].index
    assert target.transform(series[idxs]).eq(transformed_value).all()

    # We now drop points that are on the match side (except for the exact match value).
    # The result is a sequence which should result in monotonous transformed values.
    # We can ensure the sequence always describes "worsening" (decreasing) values
    # by reversing the leftover sequence depending on the mismatch mode.
    series.drop(index=idxs.drop(0), inplace=True)
    if mismatch_instead:
        series = series[::-1]

    # On the target level, "worse" can mean "larger" or "smaller", depending on the
    # minimization flag
    t1 = target.transform(series)
    operator = op.gt if target.minimize else op.lt
    assert operator(t1.diff().dropna(), 0).all()

    # By contrast, objectives are always to be maximized, so "worse" means "smaller"
    t2 = target.to_objective().transform(series.to_frame(name=target.name)).squeeze()
    diffs_correct = t2.diff().dropna() < 0
    assert diffs_correct.all(), t2.diff()


@pytest.mark.parametrize("operator", [op.add, op.sub, op.mul])
def test_valid_combination(operator):
    """Targets with the same attributes (except transformation) can be combined."""
    t = NumericalTarget("t")
    operator(t, evolve(t, transformation=ExponentialTransformation()))
