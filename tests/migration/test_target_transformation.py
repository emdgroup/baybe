"""Target transformation tests."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_series_equal
from pytest import param

from baybe.targets.numerical import NumericalTarget as ModernTarget
from baybe.targets.transforms import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    ExponentialTransformation,
    GenericTransformation,
    IdentityTransformation,
    LogarithmicTransformation,
    PowerTransformation,
    bell_transform,
    linear_transform,
    triangular_transform,
)
from baybe.utils.interval import Interval


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
            ModernTarget("t", mode),
            ModernTarget("t", mode=mode),
            ModernTarget(name="t", mode=mode),
            # ------------
            # Modern style
            ModernTarget("t", minimize=mode == "MIN"),
        ),
        (
            ModernTarget("t", mode, (1, 2)),
            ModernTarget("t", mode, bounds=(1, 2)),
            ModernTarget("t", mode=mode, bounds=(1, 2)),
            ModernTarget(name="t", mode=mode, bounds=(1, 2)),
            ModernTarget("t", mode, (1, 2), "LINEAR"),
            ModernTarget("t", mode, (1, 2), transformation="LINEAR"),
            ModernTarget("t", mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget("t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            ModernTarget(name="t", mode=mode, bounds=(1, 2), transformation="LINEAR"),
            # ------------
            # Modern style
            ModernTarget.ramp(name="t", cutoffs=(1, 2), descending=mode == "MIN"),
        ),
    ]
    for targets in groups:
        for t1, t2 in pairwise(targets):
            assert t1 == t2


@pytest.mark.parametrize("transformation", ["TRIANGULAR", "BELL"])
def test_constructor_equivalence_match(transformation):
    """Calling the new target class with legacy arguments yields the legacy object."""
    targets = (
        ModernTarget("t", "MATCH", (1, 2), transformation),
        ModernTarget("t", "MATCH", (1, 2), transformation=transformation),
        ModernTarget("t", "MATCH", bounds=(1, 2), transformation=transformation),
        ModernTarget("t", mode="MATCH", bounds=(1, 2), transformation=transformation),
        ModernTarget(
            name="t", mode="MATCH", bounds=(1, 2), transformation=transformation
        ),
    )
    if transformation == "TRIANGULAR":
        targets += (
            ModernTarget("t", "MATCH", (1, 2)),
            ModernTarget("t", "MATCH", bounds=(1, 2)),
            ModernTarget("t", mode="MATCH", bounds=(1, 2)),
            ModernTarget(name="t", mode="MATCH", bounds=(1, 2)),
        )

    # ------------
    # Modern style
    if transformation == "BELL":
        targets += (ModernTarget.match_bell("t", center=1.5, width=0.5),)
    else:
        targets += (ModernTarget.match_triangular("t", (1, 2)),)

    for t1, t2 in pairwise(targets):
        assert t1 == t2


def test_target_deserialization():
    """Deserialization also works from legacy arguments."""
    actual = ModernTarget.from_dict({"name": "t", "mode": "MATCH", "bounds": (1, 2)})
    expected = ModernTarget("t", "MATCH", (1, 2))
    assert actual == expected


@pytest.mark.parametrize(
    ("legacy", "modern", "expected"),
    [
        param(
            ModernTarget("t", "MAX"),
            ModernTarget("t"),
            sample_input(),
            id="max",
        ),
        param(
            ModernTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t", ClampingTransformation(min=0, max=1)),
            linear_transform(sample_input(), 0, 1, descending=False),
            id="max_clamped",
        ),
        param(
            ModernTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget.ramp("t", (2, 5)),
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
            ModernTarget("t", "MIN"),
            ModernTarget("t"),
            sample_input(),
            id="min",
        ),
        param(
            ModernTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget.ramp("t", (0, 1), descending=True),
            linear_transform(sample_input(), 0, 1, descending=True),
            id="min_clamped",
        ),
        param(
            ModernTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget.ramp("t", (2, 5), descending=True),
            linear_transform(sample_input(), 2, 5, descending=True),
            id="min_shifted_clamped",
        ),
        param(
            ModernTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget.match_bell("t", center=0, width=1),
            bell_transform(sample_input(), -1, 1),
            id="match_bell_unit_centered",
        ),
        param(
            ModernTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget.match_bell("t", center=2, width=1),
            bell_transform(sample_input(), 1, 3),
            id="match_bell_unit_shifted",
        ),
        param(
            ModernTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget.match_bell("t", center=0, width=5),
            bell_transform(sample_input(), -5, 5),
            id="match_bell_scaled_centered",
        ),
        param(
            ModernTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget.match_bell("t", center=4, width=2),
            bell_transform(sample_input(), 2, 6),
            id="match_bell_scaled_shifted",
        ),
        param(
            ModernTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (-1, 1)),
            triangular_transform(sample_input(), -1, 1),
            id="match_triangular_unit_centered",
        ),
        param(
            ModernTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (1, 3)),
            triangular_transform(sample_input(), 1, 3),
            id="match_triangular_unit_shifted",
        ),
        param(
            ModernTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (-5, 5)),
            triangular_transform(sample_input(), -5, 5),
            id="match_triangular_scaled_centered",
        ),
        param(
            ModernTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget.match_triangular("t", (2, 6)),
            triangular_transform(sample_input(), 2, 6),
            id="match_triangular_scaled_shifted",
        ),
    ],
)
def test_target_transformation(
    series, legacy: ModernTarget, modern: ModernTarget, expected
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

    c = ChainedTransformation([t1, t2])
    t = c.append(t3).append(c)

    assert t == ChainedTransformation([t1, t2, t3, t1, t2])


@pytest.mark.parametrize(
    "transformations",
    [
        (AbsoluteTransformation(),),
        (AbsoluteTransformation(), IdentityTransformation()),
    ],
    ids=["single", "single_with_identity"],
)
def test_invalid_transformation_chaining(transformations):
    """A chaining transformation requires at least two involved transformations."""
    with pytest.raises(ValueError, match="must be >= 2: 1"):
        ChainedTransformation(transformations)


def test_identity_transformation_chaining():
    """Chaining an identity transformation has no effect."""
    t1 = IdentityTransformation()
    t2 = ClampingTransformation()

    assert t1.append(t2) == t2
    assert t2.append(t1) == t2


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
        "t", AffineTransformation(factor=2) + GenericTransformation(torch.abs)
    )
    t2 = ModernTarget("t", torch.abs(AffineTransformation(factor=2)))
    assert_series_equal(t1.transform(series), t2.transform(series))


def test_invalid_torch_overloading():
    """Chaining torch callables works only with one-argument callables."""
    with pytest.raises(ValueError, match="enters as the only"):
        torch.add(AbsoluteTransformation(), 2)


c = 1

id = IdentityTransformation()
clamp = ClampingTransformation(min=2, max=5)
aff = AffineTransformation(factor=2, shift=1)
bell = BellTransformation(center=c, width=2)
abs = AbsoluteTransformation()
log = LogarithmicTransformation()
exp = ExponentialTransformation()
pow = PowerTransformation(0.5)
chain = ChainedTransformation([abs, clamp, aff])

aff_0 = aff(torch.tensor(0))
aff_1 = aff(torch.tensor(1))
bell_off_1 = bell(torch.tensor(bell.center + 1))
bell_off_01 = bell(torch.tensor(bell.center + 0.1))


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
        param(log, (0, None), (None, None), id="log_open"),
        param(log, (1, np.exp(1)), (0, 1), id="log_unit"),
        param(exp, (None, None), (0, None), id="exp_open"),
        param(exp, (0, 1), (1, np.exp(1)), id="exp_unit"),
        param(pow, (0, None), (0, None), id="pow_open"),
        param(pow, (1, 4), (1, 2), id="pow_finite"),
        param(chain, (-0.1, 3), (5, 7), id="chain"),
    ],
)
def test_image_computation(transformation, bounds, expected):
    """The image of a transformation is computed correctly."""
    bounds = (None, None) if bounds is None else bounds
    assert transformation.get_image(bounds) == Interval.create(expected)
