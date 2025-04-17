"""Target transformation tests."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_series_equal
from pytest import param

from baybe.targets._deprecated import NumericalTarget as LegacyTarget
from baybe.targets.numerical import NumericalTarget as ModernTarget
from baybe.targets.transforms import (
    AbsoluteTransformation,
    AffineTransformation,
    ChainedTransformation,
    ClampingTransformation,
    GenericTransformation,
    bell_transform,
    linear_transform,
    triangular_transform,
)


def sample_input() -> pd.Series:
    return pd.Series(np.linspace(-10, 10, 20))


@pytest.fixture
def series() -> pd.Series:
    return sample_input()


@pytest.mark.parametrize("mode", ["MAX", "MIN"])
def test_constructor_equivalence_min_max(mode):
    """Calling the new target class with legacy arguments yields the legacy object."""
    groups = [
        (
            LegacyTarget("t", mode),
            ModernTarget("t", mode),
            LegacyTarget("t", mode=mode),
            ModernTarget("t", mode=mode),
            LegacyTarget(name="t", mode=mode),
            ModernTarget(name="t", mode=mode),
        ),
        (
            LegacyTarget("t", mode, (1, 2)),
            ModernTarget("t", mode, (1, 2)),
            LegacyTarget("t", mode, bounds=(1, 2)),
            ModernTarget("t", mode, bounds=(1, 2)),
            LegacyTarget("t", mode=mode, bounds=(1, 2)),
            ModernTarget("t", mode=mode, bounds=(1, 2)),
            LegacyTarget(name="t", mode=mode, bounds=(1, 2)),
            ModernTarget(name="t", mode=mode, bounds=(1, 2)),
            LegacyTarget("t", mode, (1, 2), "LINEAR"),
            ModernTarget("t", mode, (1, 2), "LINEAR"),
            LegacyTarget("t", mode, (1, 2), transformation="LINEAR"),
            ModernTarget("t", mode, (1, 2), transformation="LINEAR"),
            LegacyTarget("t", mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget("t", mode, bounds=(1, 2), transformation="LINEAR"),
            LegacyTarget("t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget("t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            LegacyTarget(name="t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget(name="t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
        ),
    ]
    for targets in groups:
        for t1, t2 in pairwise(targets):
            assert t1 == t2


@pytest.mark.parametrize("transformation", ["TRIANGULAR", "BELL"])
def test_constructor_equivalence_match(transformation):
    """Calling the new target class with legacy arguments yields the legacy object."""
    targets = (
        LegacyTarget("t", "MATCH", (1, 2), transformation),
        ModernTarget("t", "MATCH", (1, 2), transformation),
        LegacyTarget("t", "MATCH", (1, 2), transformation=transformation),
        ModernTarget("t", "MATCH", (1, 2), transformation=transformation),
        LegacyTarget("t", "MATCH", bounds=(1, 2), transformation=transformation),
        ModernTarget("t", "MATCH", bounds=(1, 2), transformation=transformation),
        LegacyTarget("t", mode="MATCH", bounds=(1, 2), transformation=transformation),
        ModernTarget("t", mode="MATCH", bounds=(1, 2), transformation=transformation),
        LegacyTarget(
            name="t", mode="MATCH", bounds=(1, 2), transformation=transformation
        ),
        ModernTarget(
            name="t", mode="MATCH", bounds=(1, 2), transformation=transformation
        ),
    )
    if transformation == "TRIANGULAR":
        targets += (
            LegacyTarget("t", "MATCH", (1, 2)),
            ModernTarget("t", "MATCH", (1, 2)),
            LegacyTarget("t", "MATCH", bounds=(1, 2)),
            ModernTarget("t", "MATCH", bounds=(1, 2)),
            LegacyTarget("t", mode="MATCH", bounds=(1, 2)),
            ModernTarget("t", mode="MATCH", bounds=(1, 2)),
            LegacyTarget(name="t", mode="MATCH", bounds=(1, 2)),
            ModernTarget(name="t", mode="MATCH", bounds=(1, 2)),
        )
    for t1, t2 in pairwise(targets):
        assert t1 == t2


@pytest.mark.parametrize(
    ("legacy", "modern", "expected"),
    [
        param(
            LegacyTarget("t", "MAX"),
            ModernTarget("t"),
            sample_input(),
            id="max",
        ),
        param(
            LegacyTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t", ClampingTransformation(min=0, max=1)),
            linear_transform(sample_input(), 0, 1, descending=False),
            id="max_clamped",
        ),
        param(
            LegacyTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget.clamped_affine("t", (2, 5)),
            linear_transform(sample_input(), 2, 5, descending=False),
            id="max_shifted_clamped",
        ),
        param(
            # NOTE: Minimization transformation without bounds is not possible with
            #   legacy interface."
            None,
            ModernTarget("t", AffineTransformation(factor=-1)),
            -sample_input(),
            id="min_no_bounds",
        ),
        param(
            # NOTE: Minimization transformation without bounds is not possible with
            #   legacy interface."
            None,
            ModernTarget("t", minimize=True),
            -sample_input(),
            id="min_no_bounds_with_flag",
        ),
        param(
            # NOTE: Minimization without bounds has no effect on the transformation
            #   of the legacy target since minimization is handled in the construction
            #   of the acquisition function.
            LegacyTarget("t", "MIN"),
            ModernTarget("t"),
            sample_input(),
            id="min",
        ),
        param(
            LegacyTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget.clamped_affine("t", (0, 1), descending=True),
            linear_transform(sample_input(), 0, 1, descending=True),
            id="min_clamped",
        ),
        param(
            LegacyTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget.clamped_affine("t", (2, 5), descending=True),
            linear_transform(sample_input(), 2, 5, descending=True),
            id="min_shifted_clamped",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget.match_bell("t", center=0, width=1),
            bell_transform(sample_input(), -1, 1),
            id="match_bell_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget.match_bell("t", center=2, width=1),
            bell_transform(sample_input(), 1, 3),
            id="match_bell_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget.match_bell("t", center=0, width=5),
            bell_transform(sample_input(), -5, 5),
            id="match_bell_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget.match_bell("t", center=4, width=2),
            bell_transform(sample_input(), 2, 6),
            id="match_bell_scaled_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (-1, 1)),
            triangular_transform(sample_input(), -1, 1),
            id="match_triangular_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (1, 3)),
            triangular_transform(sample_input(), 1, 3),
            id="match_triangular_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (-5, 5)),
            triangular_transform(sample_input(), -5, 5),
            id="match_triangular_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (2, 6)),
            triangular_transform(sample_input(), 2, 6),
            id="match_triangular_scaled_shifted",
        ),
    ],
)
def test_target_transformation(
    series, legacy: LegacyTarget, modern: ModernTarget, expected
):
    """The legacy and modern target variants transform equally."""
    transformed_modern = modern.transform(series)
    if legacy is not None:
        assert_series_equal(transformed_modern, legacy.transform(series))
    assert_series_equal(transformed_modern, pd.Series(expected))


def test_transformation_chaining():
    """Transformation chaining and flattening works as expected."""
    t1 = AffineTransformation()
    t2 = ClampingTransformation()
    t3 = AbsoluteTransformation()

    c = ChainedTransformation(t1, t2)
    t = c.append(t3).append(c)

    assert t == ChainedTransformation(t1, t2, t3, t1, t2)


def test_generic_transformation(series):
    """Torch callables can be used to specify generic transformations."""
    t1 = ModernTarget("t", AbsoluteTransformation())
    t2 = ModernTarget("t", GenericTransformation(torch.abs))
    t3 = ModernTarget("t", torch.abs)

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))


def test_transformation_addition(series):
    """Adding transformations results in chaining/shifting."""
    t1 = ModernTarget(
        "t",
        ChainedTransformation(AbsoluteTransformation(), AffineTransformation(shift=1)),
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
        ChainedTransformation(AbsoluteTransformation(), AffineTransformation(factor=2)),
    )
    t2 = ModernTarget("t", AbsoluteTransformation() * 2)
    t3 = ModernTarget("t", AbsoluteTransformation() + AffineTransformation(factor=2))

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))


def test_torch_overloading(series):
    """Transformations can be passed to torch callables for chaining."""
    t1 = ModernTarget(
        "t", AffineTransformation(factor=2) + GenericTransformation(torch.abs)
    )
    t2 = ModernTarget("t", torch.abs(AffineTransformation(factor=2)))
    assert_series_equal(t1.transform(series), t2.transform(series))


def test_invalid_torch_overloading():
    """Chaining torch callables works only with one-argument callables."""
    with pytest.raises(ValueError, match="enters as the only"):
        torch.add(AbsoluteTransformation(), 2)
