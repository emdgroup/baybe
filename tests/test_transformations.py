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
    SigmoidTransformation,
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


def _assert_transformation_values(transformation, inputs, expected):
    """Assert that a transformation maps the given inputs to the expected values."""
    torch.testing.assert_close(
        transformation(to_tensor(np.array(inputs))),
        to_tensor(np.array(expected)),
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("function", "inputs", "expected"),
    [
        param(torch.neg, [-2.5, 0.5, 3.0], [2.5, -0.5, -3.0], id="negation"),
        param(torch.square, [-2.0, 0.0, 3.0], [4.0, 0.0, 9.0], id="square"),
    ],
)
def test_custom_transformation_values(function, inputs, expected):
    """Custom transformations apply the wrapped torch callable."""
    _assert_transformation_values(CustomTransformation(function), inputs, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        param([-3.5, 0.25, 2.0], [-3.5, 0.25, 2.0], id="mixed-values"),
        param([-1e6, 0.5, 2.5e6], [-1e6, 0.5, 2.5e6], id="large-magnitudes"),
    ],
)
def test_identity_transformation_values(inputs, expected):
    """Identity transformations return their inputs unchanged."""
    _assert_transformation_values(IdentityTransformation(), inputs, expected)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            ClampingTransformation(min=-1.0, max=1.0),
            [-3.0, 0.25, 1.5],
            [-1.0, 0.25, 1.0],
            id="two-sided",
        ),
        param(
            ClampingTransformation(min=2.0),
            [-4.0, 2.0, 5.0],
            [2.0, 2.0, 5.0],
            id="lower-only",
        ),
        param(
            ClampingTransformation(max=2.0),
            [-5.0, 0.5, 3.0],
            [-5.0, 0.5, 2.0],
            id="upper-only",
        ),
    ],
)
def test_clamping_transformation_values(transformation, inputs, expected):
    """Clamping transformations limit values according to their configured cutoffs."""
    _assert_transformation_values(transformation, inputs, expected)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            AffineTransformation(),
            [-3.0, 0.5, 2.0],
            [-3.0, 0.5, 2.0],
            id="identity",
        ),
        param(
            AffineTransformation(factor=2.0),
            [-3.0, 0.5, 2.0],
            [-6.0, 1.0, 4.0],
            id="scale",
        ),
        param(
            AffineTransformation(shift=3.0),
            [-3.0, 0.5, 2.0],
            [0.0, 3.5, 5.0],
            id="shift",
        ),
        param(
            AffineTransformation(factor=-1.0, shift=1.0),
            [-3.0, 0.5, 2.0],
            [4.0, 0.5, -1.0],
            id="negative-scale",
        ),
        param(
            AffineTransformation(factor=2.0, shift=3.0, shift_first=True),
            [-3.0, 0.5, 2.0],
            [0.0, 7.0, 10.0],
            id="shift-then-scale",
        ),
        param(
            AffineTransformation(factor=0.0, shift=3.0),
            [-np.inf, -3.0, 0.5, 2.0, np.inf],
            [3.0, 3.0, 3.0, 3.0, 3.0],
            id="zero-factor",
        ),
    ],
)
def test_affine_transformation_values(transformation, inputs, expected):
    """Affine transformations scale and shift their inputs as configured."""
    _assert_transformation_values(transformation, inputs, expected)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            TwoSidedAffineTransformation(slope_left=-1.0, slope_right=1.0),
            [-2.5, 0.25, 1.5],
            [2.5, 0.25, 1.5],
            id="absolute",
        ),
        param(
            TwoSidedAffineTransformation(
                slope_left=2.0, slope_right=-1.0, midpoint=1.0
            ),
            [-1.0, 0.5, 2.5],
            [-4.0, -1.0, -1.5],
            id="positive-left-negative-right",
        ),
        param(
            TwoSidedAffineTransformation(slope_left=0.0, slope_right=1.0, midpoint=1.0),
            [-np.inf, -1.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 2.0],
            id="zero-left-slope",
        ),
        param(
            TwoSidedAffineTransformation(slope_left=1.0, slope_right=0.0, midpoint=1.0),
            [-3.0, 1.0, 3.0, np.inf],
            [-4.0, 0.0, 0.0, 0.0],
            id="zero-right-slope",
        ),
    ],
)
def test_two_sided_affine_transformation_values(transformation, inputs, expected):
    """Two-sided affine transformations apply the configured per-side slopes."""
    _assert_transformation_values(transformation, inputs, expected)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            BellTransformation(center=0.0, sigma=1.0),
            [-2.5, 0.0, 0.5],
            [np.exp(-3.125), 1.0, np.exp(-0.125)],
            id="standard",
        ),
        param(
            BellTransformation(center=1.0, sigma=1.0),
            [-1.5, 1.0, 1.5],
            [np.exp(-3.125), 1.0, np.exp(-0.125)],
            id="shift-only",
        ),
        param(
            BellTransformation(center=0.0, sigma=0.5),
            [-1.0, 0.0, 0.25],
            [np.exp(-2.0), 1.0, np.exp(-0.125)],
            id="scale-only",
        ),
        param(
            BellTransformation(center=1.0, sigma=2.0),
            [-3.0, 1.0, 2.0],
            [np.exp(-2.0), 1.0, np.exp(-0.125)],
            id="shift-and-scale",
        ),
    ],
)
def test_bell_transformation_values(transformation, inputs, expected):
    """Bell transformations peak at their center and decay with their width."""
    _assert_transformation_values(transformation, inputs, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        param(
            [-2.5, -1.0, 0.25, 1.5],
            [2.5, 1.0, 0.25, 1.5],
            id="mixed-signs",
        ),
        param(
            [-1.5e6, 1e-3, 2.5e6],
            [1.5e6, 1e-3, 2.5e6],
            id="large-magnitudes",
        ),
    ],
)
def test_absolute_transformation_values(inputs, expected):
    """Absolute transformations return the distance from zero."""
    _assert_transformation_values(AbsoluteTransformation(), inputs, expected)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            TriangularTransformation(cutoffs=(-1.0, 1.0), peak=0.0),
            [-2.0, -1.0, -0.75, 0.0, 0.25, 1.0, 2.5],
            [0.0, 0.0, 0.25, 1.0, 0.75, 0.0, 0.0],
            id="centered",
        ),
        param(
            TriangularTransformation(cutoffs=(0.0, 4.0), peak=1.0),
            [-1.0, 0.0, 0.5, 1.0, 3.0, 4.0, 5.0],
            [0.0, 0.0, 0.5, 1.0, 1 / 3, 0.0, 0.0],
            id="peak-left",
        ),
        param(
            TriangularTransformation(cutoffs=(0.0, 4.0), peak=3.0),
            [-1.0, 0.0, 1.5, 3.0, 3.5, 4.0, 5.0],
            [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0],
            id="peak-right",
        ),
    ],
)
def test_triangular_transformation_values(transformation, inputs, expected):
    """Triangular transformations interpolate between cutoffs and peak."""
    _assert_transformation_values(transformation, inputs, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        param([0.25, 1.0, np.exp(1.0)], [np.log(0.25), 0.0, 1.0], id="positive"),
        param([-1.0, 0.0], [np.nan, -np.inf], id="outside-domain"),
    ],
)
def test_logarithmic_transformation_values(inputs, expected):
    """Logarithmic transformations map positive inputs to their natural logarithm."""
    _assert_transformation_values(LogarithmicTransformation(), inputs, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        param(
            [-2.0, 0.5, 1.0],
            [np.exp(-2.0), np.exp(0.5), np.exp(1.0)],
            id="finite",
        ),
        param(
            [-np.inf, 0.5, np.inf],
            [0.0, np.exp(0.5), np.inf],
            id="limits",
        ),
    ],
)
def test_exponential_transformation_values(inputs, expected):
    """Exponential transformations map inputs to their natural exponential."""
    _assert_transformation_values(ExponentialTransformation(), inputs, expected)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            SigmoidTransformation(),
            [-2.0, 0.5, 1.5],
            [
                1 / (1 + np.exp(2.0)),
                1 / (1 + np.exp(-0.5)),
                1 / (1 + np.exp(-1.5)),
            ],
            id="standard",
        ),
        param(
            SigmoidTransformation(center=1.0),
            [-1.5, 1.0, 2.5],
            [
                1 / (1 + np.exp(2.5)),
                0.5,
                1 / (1 + np.exp(-1.5)),
            ],
            id="shift-only",
        ),
        param(
            SigmoidTransformation(slope=0.5),
            [-2.0, 0.5, 1.5],
            [
                1 / (1 + np.exp(4.0)),
                1 / (1 + np.exp(-1.0)),
                1 / (1 + np.exp(-3.0)),
            ],
            id="scale-only",
        ),
        param(
            SigmoidTransformation(center=1.0, slope=1.5),
            [-1.0, 1.0, 2.5],
            [
                1 / (1 + np.exp(12.0)),
                0.5,
                1 / (1 + np.exp(-9.0)),
            ],
            id="shift-and-scale",
        ),
        param(
            SigmoidTransformation(center=1.0, slope=-1.0),
            [-1.0, 1.0, 2.5],
            [
                1 / (1 + np.exp(-8.0)),
                0.5,
                1 / (1 + np.exp(6.0)),
            ],
            id="decreasing",
        ),
        param(
            SigmoidTransformation(slope=0.0),
            [-1e6, 0.5, 2.5e6],
            [0.5, 0.5, 0.5],
            id="zero-slope",
        ),
    ],
)
def test_sigmoid_transformation_values(transformation, inputs, expected):
    """Sigmoid transformations map values according to their center and slope."""
    _assert_transformation_values(transformation, inputs, expected)


@pytest.mark.parametrize("slope", [-2.0, -0.25, 0.25, 1.0, 3.0])
def test_sigmoid_transformation_slope_matches_derivative(slope):
    """The slope parameter equals the derivative of the sigmoid at its center."""
    center = 1.5
    transformation = SigmoidTransformation(center=center, slope=slope)
    x = torch.tensor([center], dtype=torch.float64, requires_grad=True)

    (gradient,) = torch.autograd.grad(transformation(x), x)

    torch.testing.assert_close(gradient, torch.tensor([slope], dtype=torch.float64))


@pytest.mark.parametrize(
    ("anchors", "expected_slope_sign"),
    [
        param([(-2.0, 0.1), (5.0, 0.6)], 1, id="increasing"),
        param([(-2.0, 0.9), (5.0, 0.4)], -1, id="decreasing"),
    ],
)
def test_sigmoid_transformation_from_anchors(anchors, expected_slope_sign):
    """Sigmoid transformations pass through their anchors in the expected direction."""
    transformation = SigmoidTransformation.from_anchors(anchors)

    assert np.sign(transformation.slope) == expected_slope_sign

    x = np.array([x for x, _ in anchors])
    y = np.array([y for _, y in anchors])
    torch.testing.assert_close(transformation(to_tensor(x)), to_tensor(y))


@pytest.mark.parametrize(
    ("anchors", "match"),
    [
        param([(0.0, 0.1)], "convertible to two pairs", id="wrong-number"),
        param([("a", 0.1), (5.0, 0.6)], "convertible to two pairs", id="non-numeric"),
        param([(0.0, 0.0), (1.0, 0.5)], "open interval", id="lower-bound"),
        param([(0.0, 0.5), (1.0, 1.0)], "open interval", id="upper-bound"),
        param([(0.0, -0.1), (1.0, 0.5)], "open interval", id="below-interval"),
        param([(0.0, 0.5), (1.0, 1.1)], "open interval", id="above-interval"),
        param([(0.0, 0.1), (0.0, 0.6)], "abscissas", id="duplicate-abscissa"),
        param([(0.0, 0.5), (1.0, 0.5)], "ordinates", id="duplicate-ordinate"),
    ],
)
def test_sigmoid_transformation_from_anchors_rejects_invalid_anchors(anchors, match):
    """Sigmoid transformations reject malformed or underdetermined anchors."""
    with pytest.raises(ValueError, match=match):
        SigmoidTransformation.from_anchors(anchors)


@pytest.mark.parametrize(
    ("transformation", "inputs", "expected"),
    [
        param(
            PowerTransformation(exponent=2),
            [-2.0, -0.5, 0.0, 1.5, 3.0],
            [4.0, 0.25, 0.0, 2.25, 9.0],
            id="even",
        ),
        param(
            PowerTransformation(exponent=3),
            [-2.0, -0.5, 0.0, 1.5, 3.0],
            [-8.0, -0.125, 0.0, 3.375, 27.0],
            id="odd",
        ),
        param(
            PowerTransformation(exponent=1.5),
            [0.0, 0.25, 4.0],
            [0.0, 0.125, 8.0],
            id="non-integer",
        ),
        param(
            PowerTransformation(exponent=-1),
            [0.5, 1.0, 4.0],
            [2.0, 1.0, 0.25],
            id="negative",
        ),
        param(
            PowerTransformation(exponent=0),
            [-3.0, 0.5, 2.0],
            [1.0, 1.0, 1.0],
            id="zero",
        ),
    ],
)
def test_power_transformation_values(transformation, inputs, expected):
    """Power transformations raise their inputs to the configured exponent."""
    _assert_transformation_values(transformation, inputs, expected)


def test_power_transformation_rejects_negative_input_for_non_integer_exponent():
    """Non-integer power transformations reject negative inputs."""
    transformation = PowerTransformation(exponent=1.5)

    with pytest.raises(RuntimeError, match="must contain non-negative elements"):
        transformation(to_tensor(np.array([-0.25, 1.0])))


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


def test_three_distinct_transformations_chaining() -> None:
    """Verify three piped transformations apply in left-to-right order."""
    t1 = AffineTransformation(factor=2, shift=1)
    t2 = PowerTransformation(exponent=2)
    t3 = ClampingTransformation(min=0, max=10)
    x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

    chained = t1 | t2 | t3
    grouped = (t1 | t2) | t3
    nested_output = t3(t2(t1(x)))

    assert chained == grouped == ChainedTransformation([t1, t2, t3])
    assert torch.equal(chained(x), nested_output)
    assert torch.equal(grouped(x), nested_output)


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
    t1b = (t * 2).vshift(3)
    assert t1 == t1b == AffineTransformation(factor=2, shift=3)

    t2 = (t + 3) * 2 | t
    t2b = t.vshift(3) * 2
    assert t2 == t2b == AffineTransformation(factor=2, shift=3, shift_first=True)


@pytest.mark.parametrize(
    "t, expected",
    [
        param(
            BellTransformation(center=0, sigma=1),
            BellTransformation(center=2, sigma=1),
            id="bell",
        ),
        param(
            TwoSidedAffineTransformation(midpoint=0, slope_left=-4, slope_right=2),
            TwoSidedAffineTransformation(midpoint=2, slope_left=-4, slope_right=2),
            id="2sided_affine",
        ),
        param(
            TriangularTransformation(peak=0, cutoffs=(-2, 1)),
            TriangularTransformation(peak=2, cutoffs=(0, 3)),
            id="triangular_cutoffs",
        ),
        param(
            TriangularTransformation.from_margins(peak=0.0, margins=(2, 1)),
            TriangularTransformation(peak=2, cutoffs=(0, 3)),
            id="triangular_margins",
        ),
        param(
            TriangularTransformation.from_width(peak=0.0, width=4),
            TriangularTransformation(peak=2, cutoffs=(0, 4)),
            id="triangular_width",
        ),
    ],
)
def test_positional_shift_transformation_compression(t, expected):
    """Simple input shifts are compressed correctly."""
    t1 = AffineTransformation(shift=-2) | t
    t2 = t.hshift(2)
    assert t1 == t2 == expected


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
sig_inc = SigmoidTransformation(center=2, slope=1)
sig_dec = SigmoidTransformation(center=2, slope=-1)
sig_const = SigmoidTransformation(slope=0)
pow_even = PowerTransformation(exponent=2)
pow_odd = PowerTransformation(exponent=3)
chain = ChainedTransformation([abs, clamp, aff])

# Transformed values
aff_0 = aff(to_tensor(0))
aff_1 = aff(to_tensor(1))
bell_off_1 = bell(to_tensor(bell.center + 1))
bell_off_01 = bell(to_tensor(bell.center + 0.1))
sig_inc_3 = sig_inc(to_tensor(3))
sig_dec_3 = sig_dec(to_tensor(3))


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
        param(sig_inc, (None, None), (0, 1), id="sig_inc_open"),
        param(sig_inc, (3, None), (sig_inc_3, 1), id="sig_inc_half_open_reduced"),
        param(sig_dec, (None, None), (0, 1), id="sig_dec_open"),
        param(sig_dec, (3, None), (0, sig_dec_3), id="sig_dec_half_open_reduced"),
        param(sig_const, (None, None), (0.5, 0.5), id="sig_const_open"),
        param(sig_const, (-1, 1), (0.5, 0.5), id="sig_const_finite"),
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
    ("transformation", "expected"),
    [
        param(AffineTransformation(factor=0.0), Interval(0, 0), id="affine"),
        param(TwoSidedAffineTransformation(0, 0), Interval(0, 0), id="two_sided"),
        param(ClampingTransformation(0, 0), Interval(0, 0), id="clamping"),
        param(SigmoidTransformation(slope=0), Interval(0.5, 0.5), id="sigmoid"),
    ],
)
def test_degenerate_transformations(transformation, expected):
    """Degenerate transformations produce proper (non-nan) outputs."""
    assert transformation.get_image() == expected
    assert transformation.get_image((20, None)) == expected


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
