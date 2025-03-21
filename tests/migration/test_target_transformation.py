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
    transformed_modern = modern.transform(series)
    if legacy is not None:
        assert_series_equal(transformed_modern, legacy.transform(series))
    assert_series_equal(transformed_modern, pd.Series(expected))


def test_transformation_chaining():
    t1 = AffineTransformation()
    t2 = ClampingTransformation()
    t3 = AbsoluteTransformation()

    c = ChainedTransformation(t1, t2)
    t = c.append(t3).append(c)

    assert t == ChainedTransformation(t1, t2, t3, t1, t2)


def test_generic_transformation(series):
    t1 = ModernTarget("t", AbsoluteTransformation())
    t2 = ModernTarget("t", GenericTransformation(torch.abs))
    t3 = ModernTarget("t", torch.abs)

    transformed = t1.transform(series)
    assert_series_equal(transformed, t2.transform(series))
    assert_series_equal(transformed, t3.transform(series))


def test_transformation_addition(series):
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
    t1 = ModernTarget(
        "t", AffineTransformation(factor=2) + GenericTransformation(torch.abs)
    )
    t2 = ModernTarget("t", torch.abs(AffineTransformation(factor=2)))
    assert_series_equal(t1.transform(series), t2.transform(series))


def test_invalid_torch_overloading():
    with pytest.raises(ValueError, match="enters as the only"):
        torch.add(AbsoluteTransformation(), 2)
