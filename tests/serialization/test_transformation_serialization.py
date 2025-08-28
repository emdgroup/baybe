"""Test serialization of transformations."""

import hypothesis.strategies as st
import pytest
from hypothesis import given
from pytest import param

from baybe.transformations import (
    Transformation,
)

from ..hypothesis_strategies.transformations import (
    absolute_transformations,
    affine_transformations,
    bell_transformations,
    chained_transformations,
    clamping_transformations,
    exponential_transformations,
    identity_transformations,
    logarithmic_transformations,
    power_transformations,
    triangular_transformations,
    two_sided_affine_transformations,
)


@pytest.mark.parametrize(
    "transformation_strategy",
    [
        param(identity_transformations(), id="IdentityTransformation"),
        param(absolute_transformations(), id="AbsoluteTransformation"),
        param(logarithmic_transformations(), id="LogarithmicTransformation"),
        param(exponential_transformations(), id="ExponentialTransformation"),
        param(clamping_transformations(), id="ClampingTransformation"),
        param(affine_transformations(), id="AffineTransformation"),
        param(
            two_sided_affine_transformations(),
            id="TwoSidedAffineTransformation",
        ),
        param(bell_transformations(), id="BellTransformation"),
        param(triangular_transformations(), id="TriangularTransformation"),
        param(power_transformations(), id="PowerTransformation"),
        param(chained_transformations(), id="ChainedTransformation"),
    ],
)
@given(data=st.data())
def test_single_transformation_roundtrip(transformation_strategy, data):
    """A serialization roundtrip yields an equivalent object."""
    transformation = data.draw(transformation_strategy)
    string = transformation.to_json()
    transformation2 = Transformation.from_json(string)
    assert transformation == transformation2, (transformation, transformation2)
