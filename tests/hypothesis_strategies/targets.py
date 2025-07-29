"""Hypothesis strategies for targets."""

import hypothesis.strategies as st

from baybe.targets import NumericalTarget
from baybe.targets.binary import BinaryTarget

from ..hypothesis_strategies.transformations import (
    chained_transformations,
    single_transformations,
)

target_names = st.text(min_size=1)
"""A strategy that generates target names."""


@st.composite
def numerical_targets(draw: st.DrawFn):
    """Generate :class:`baybe.targets.numerical.NumericalTarget`."""
    name = draw(target_names)
    transformation = draw(st.one_of(single_transformations, chained_transformations()))
    minimize = draw(st.booleans())
    return NumericalTarget(name, transformation, minimize=minimize)


choice_values = st.one_of(
    [st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()]
)
"""A strategy that generates choice values."""


@st.composite
def binary_targets(draw: st.DrawFn):
    """A strategy that generates binary targets."""
    name = draw(target_names)
    choices = draw(st.lists(choice_values, min_size=2, max_size=2, unique=True))
    return BinaryTarget(name, success_value=choices[0], failure_value=choices[1])


targets = st.one_of([binary_targets(), numerical_targets()])
"""A strategy that generates targets."""
