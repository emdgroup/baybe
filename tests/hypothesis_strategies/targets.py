"""Hypothesis strategies for targets."""

import hypothesis.strategies as st

from baybe.targets import NumericalTarget
from baybe.targets.binary import BinaryTarget
from baybe.transformations.basic import ClampingTransformation
from tests.hypothesis_strategies.metadata import measurable_metadata
from tests.hypothesis_strategies.transformations import (
    chained_transformations,
    single_transformations,
)

target_names = st.text(min_size=1)
"""A strategy that generates target names."""


@st.composite
def numerical_targets(draw: st.DrawFn, normalized: bool = False):
    """Generate :class:`baybe.targets.numerical.NumericalTarget`."""
    name = draw(target_names)
    if normalized:
        transformation = ClampingTransformation(0, 1)
    else:
        transformation = draw(
            st.one_of(single_transformations, chained_transformations())
        )
    minimize = draw(st.booleans())
    metadata = draw(measurable_metadata())
    return NumericalTarget(name, transformation, minimize=minimize, metadata=metadata)


choice_values = st.one_of(
    [st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()]
)
"""A strategy that generates choice values."""


@st.composite
def binary_targets(draw: st.DrawFn):
    """A strategy that generates binary targets."""
    name = draw(target_names)
    choices = draw(st.lists(choice_values, min_size=2, max_size=2, unique=True))
    target_metadata = draw(measurable_metadata())
    return BinaryTarget(
        name,
        success_value=choices[0],
        failure_value=choices[1],
        metadata=target_metadata,
    )


targets = st.one_of([binary_targets(), numerical_targets()])
"""A strategy that generates targets."""
