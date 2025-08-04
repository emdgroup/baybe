"""Hypothesis strategies for transformations."""

import hypothesis
import hypothesis.strategies as st
from hypothesis import assume

from baybe.transformations import (
    AbsoluteTransformation,
    AffineTransformation,
    BellTransformation,
    ChainedTransformation,
    ClampingTransformation,
    ExponentialTransformation,
    IdentityTransformation,
    LogarithmicTransformation,
    PowerTransformation,
    TriangularTransformation,
    TwoSidedAffineTransformation,
)
from tests.hypothesis_strategies.utils import intervals

from ..hypothesis_strategies.basic import finite_floats, positive_finite_floats

identity_transformations = lambda: st.just(IdentityTransformation())  # noqa: E731
absolute_transformations = lambda: st.just(AbsoluteTransformation())  # noqa: E731
logarithmic_transformations = lambda: st.just(LogarithmicTransformation())  # noqa: E731
exponential_transformations = lambda: st.just(ExponentialTransformation())  # noqa: E731


@st.composite
def clamping_transformations(draw: st.DrawFn) -> ClampingTransformation:
    """Generate :class:`baybe.transformations.basic.ClampingTransformation`."""
    min = draw(st.floats(allow_nan=False, max_value=float("inf"), exclude_max=True))
    max = draw(st.floats(allow_nan=False, min_value=min, exclude_min=True))
    return ClampingTransformation(min, max)


@st.composite
def affine_transformations(draw: st.DrawFn) -> AffineTransformation:  # type: ignore[return]
    """Generate :class:`baybe.transformations.basic.AffineTransformation`."""
    factor = draw(finite_floats())
    shift = draw(finite_floats())
    shift_first = draw(st.booleans())

    try:
        return AffineTransformation(factor, shift, shift_first=shift_first)
    except OverflowError:
        assume(False)


@st.composite
def two_sided_affine_transformations(draw: st.DrawFn) -> TwoSidedAffineTransformation:
    """Generate :class:`baybe.transformations.basic.TwoSidedAffineTransformation`."""
    slope_left = draw(finite_floats())
    slope_right = draw(finite_floats())
    center = draw(finite_floats())
    return TwoSidedAffineTransformation(slope_left, slope_right, center)


@st.composite
def bell_transformations(draw: st.DrawFn) -> BellTransformation:
    """Generate :class:`baybe.transformations.basic.BellTransformation`."""
    center = draw(finite_floats())
    sigma = draw(positive_finite_floats())
    return BellTransformation(center=center, sigma=sigma)


@st.composite
def triangular_transformations(draw: st.DrawFn) -> TriangularTransformation:  # type: ignore[return]
    """Generate :class:`baybe.transformations.basic.TriangularTransformation`."""
    cutoffs = draw(intervals(exclude_fully_unbounded=True, exclude_half_bounded=True))
    try:
        peak = draw(
            st.floats(
                min_value=cutoffs.lower,
                max_value=cutoffs.upper,
                exclude_min=True,
                exclude_max=True,
            )
        )
    # Unclear how to avoid these situations upfront, so dropping sample if they occur
    except hypothesis.errors.InvalidArgument:
        # This situation happens when the cutoffs are extreme so that there exists no
        # floating point number between them for the peak
        assume(False)

    try:
        return TriangularTransformation(cutoffs, peak)
    except OverflowError:
        # The cutoffs/peak must be chosen such that the derived slopes do not overflow
        assume(False)


@st.composite
def power_transformations(draw: st.DrawFn) -> PowerTransformation:
    """Generate :class:`baybe.transformations.basic.PowerTransformation`."""
    exponent = draw(st.integers(min_value=2))
    return PowerTransformation(exponent=exponent)


@st.composite
def chained_transformations(
    draw: st.DrawFn, min_size: int = 2, max_size: int = 3
) -> ChainedTransformation:
    """Generate :class:`baybe.transformations.basic.ChainedTransformation`."""
    transformations = draw(
        st.lists(single_transformations, min_size=min_size, max_size=max_size)
    )
    try:
        return ChainedTransformation(transformations)
    except OverflowError:
        # Combining affine transformations while chaining can lead to overflow
        assume(False)


single_transformations = st.one_of(
    [
        identity_transformations(),
        absolute_transformations(),
        logarithmic_transformations(),
        exponential_transformations(),
        clamping_transformations(),
        affine_transformations(),
        two_sided_affine_transformations(),
        bell_transformations(),
        triangular_transformations(),
        power_transformations(),
    ]
)
"""A strategy that generates single transformations."""
