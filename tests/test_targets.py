"""Target tests."""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import torch
from pandas.testing import assert_series_equal
from pytest import param

from baybe.exceptions import IncompatibilityError
from baybe.targets._deprecated import (
    LegacyTarget,
    bell_transform,
    linear_transform,
    triangular_transform,
)
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
from baybe.utils.interval import Interval


def sample_input() -> pd.Series:
    return pd.Series(np.linspace(-10, 10, 20))


@pytest.fixture
def series() -> pd.Series:
    return sample_input()


@pytest.mark.parametrize("mode", ["MAX", "MIN"])
def test_constructor_equivalence_min_max(mode):
    """
    Calling the new target class with legacy arguments yields an object equivalent
    to the legacy object.
    """  # noqa
    groups = [
        (
            # ------------
            # Legacy style
            ModernTarget("t", mode),
            ModernTarget("t", mode=mode),
            ModernTarget(name="t", mode=mode),
            # ------------
            # Modern style
            ModernTarget("t", minimize=mode == "MIN"),
        ),
        (
            # ------------
            # Legacy style
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
            ModernTarget.normalize_ramp(
                name="t", cutoffs=(1, 2), descending=mode == "MIN"
            ),
        ),
    ]
    for targets in groups:
        for t1, t2 in pairwise(targets):
            assert t1 == t2


@pytest.mark.parametrize("transformation", ["TRIANGULAR", "BELL"])
def test_constructor_equivalence_match(transformation):
    """
    Calling the new target class with legacy arguments yields an object equivalent
    to the legacy object.
    """  # noqa
    # ------------
    # Legacy style
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
        targets += (ModernTarget.match_bell("t", match_value=1.5, sigma=0.5),)
    else:
        targets += (
            ModernTarget.match_triangular("t", cutoffs=(1, 2)),
            ModernTarget.match_triangular("t", match_value=1.5, width=1),
            ModernTarget.match_triangular("t", match_value=1.5, margins=(0.5, 0.5)),
        )

    for t1, t2 in pairwise(targets):
        assert t1 == t2


@pytest.mark.parametrize(
    ("legacy", "deprecation", "modern", "expected"),
    [
        param(
            LegacyTarget("t", "MAX"),
            ModernTarget("t", "MAX"),
            ModernTarget("t"),
            sample_input(),
            id="max",
        ),
        param(
            LegacyTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t", "MAX", (0, 1), "LINEAR"),
            ModernTarget("t").clamp(0, 1),
            linear_transform(sample_input(), 0, 1, descending=False),
            id="max_clamped",
        ),
        param(
            LegacyTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget("t", "MAX", (2, 5), "LINEAR"),
            ModernTarget.normalize_ramp("t", (2, 5)),
            linear_transform(sample_input(), 2, 5, descending=False),
            id="max_shifted_clamped",
        ),
        param(
            # NOTE: Minimization transformation without bounds is not possible with
            #   legacy interface."
            None,
            None,
            ModernTarget("t", AffineTransformation(factor=-1)),
            -sample_input(),
            id="min_no_bounds",
        ),
        param(
            # NOTE: Minimization transformation without bounds is not possible with
            #   legacy interface."
            None,
            None,
            ModernTarget("t", minimize=True),
            -sample_input(),
            id="min_no_bounds_with_flag",
        ),
        param(
            # NOTE: Minimization without bounds had no effect on the transformation
            #   of the legacy target since minimization was handled in the construction
            #   of the acquisition function:
            #   * https://github.com/emdgroup/baybe/pull/462
            #   * https://github.com/emdgroup/baybe/issues/460
            None,  # should be `LegacyTarget("t", "MIN")` but see explanation above
            ModernTarget("t", "MIN"),
            ModernTarget("t", minimize=True),
            -sample_input(),
            id="min",
        ),
        param(
            LegacyTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget("t", "MIN", (0, 1), "LINEAR"),
            ModernTarget.normalize_ramp("t", (0, 1), descending=True),
            linear_transform(sample_input(), 0, 1, descending=True),
            id="min_clamped",
        ),
        param(
            LegacyTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget("t", "MIN", (2, 5), "LINEAR"),
            ModernTarget.normalize_ramp("t", (2, 5), descending=True),
            linear_transform(sample_input(), 2, 5, descending=True),
            id="min_shifted_clamped",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget("t", "MATCH", (-1, 1), "BELL"),
            ModernTarget.match_bell("t", match_value=0, sigma=1),
            bell_transform(sample_input(), -1, 1),
            id="match_bell_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget("t", "MATCH", (1, 3), "BELL"),
            ModernTarget.match_bell("t", match_value=2, sigma=1),
            bell_transform(sample_input(), 1, 3),
            id="match_bell_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget("t", "MATCH", (-5, 5), "BELL"),
            ModernTarget.match_bell("t", match_value=0, sigma=5),
            bell_transform(sample_input(), -5, 5),
            id="match_bell_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget("t", "MATCH", (2, 6), "BELL"),
            ModernTarget.match_bell("t", match_value=4, sigma=2),
            bell_transform(sample_input(), 2, 6),
            id="match_bell_scaled_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (-1, 1), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(-1, 1)),
            triangular_transform(sample_input(), -1, 1),
            id="match_triangular_unit_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (1, 3), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(1, 3)),
            triangular_transform(sample_input(), 1, 3),
            id="match_triangular_unit_shifted",
        ),
        param(
            LegacyTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (-5, 5), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(-5, 5)),
            triangular_transform(sample_input(), -5, 5),
            id="match_triangular_scaled_centered",
        ),
        param(
            LegacyTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget("t", "MATCH", (2, 6), "TRIANGULAR"),
            ModernTarget.match_triangular("t", cutoffs=(2, 6)),
            triangular_transform(sample_input(), 2, 6),
            id="match_triangular_scaled_shifted",
        ),
    ],
)
def test_target_transformation(
    series,
    legacy: LegacyTarget,
    deprecation: ModernTarget,
    modern: ModernTarget,
    expected,
):
    """The legacy and modern target variants transform equally."""
    expected = pd.Series(expected)
    if legacy is not None:
        assert_series_equal(legacy.transform(series), expected)
    if deprecation is not None:
        assert_series_equal(deprecation.transform(series), expected)
    assert_series_equal(modern.transform(series), expected)


@pytest.mark.parametrize("chained_first", [True, False])
def test_transformation_chaining(chained_first):
    """Transformation chaining and flattening works as expected."""
    t1 = AffineTransformation()
    t2 = ClampingTransformation()
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
    t2 = ClampingTransformation()

    assert t1.append(t2) == t2
    assert t2.append(t1) == t2


def test_generic_transformation(series):
    """Torch callables can be used to specify generic transformations."""
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
ts_v = TwoSidedLinearTransformation(slope_left=-4, slope_right=5, center=2)
ts_l = TwoSidedLinearTransformation(slope_left=2, slope_right=5, center=2)
ts_n = TwoSidedLinearTransformation(slope_left=4, slope_right=-5, center=2)
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
