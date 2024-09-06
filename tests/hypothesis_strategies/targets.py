"""Hypothesis strategies for targets."""

import hypothesis.strategies as st

from baybe.targets.binary import BinaryTarget
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import _VALID_TRANSFORMATIONS, NumericalTarget
from baybe.utils.interval import Interval

from .utils import intervals as st_intervals

target_name = st.text(min_size=1)
"""A strategy that generates target names."""


@st.composite
def numerical_targets(
    draw: st.DrawFn, bounds_strategy: st.SearchStrategy[Interval] | None = None
):
    """Generate :class:`baybe.targets.numerical.NumericalTarget`.

    Args:
        draw: Hypothesis draw object.
        bounds_strategy: An optional strategy for generating the target bounds.

    Returns:
        _type_: _description_
    """
    name = draw(target_name)
    mode = draw(st.sampled_from(TargetMode))
    if bounds_strategy is None:
        bounds_strategy = st_intervals(
            exclude_half_bounded=True, exclude_fully_unbounded=mode is TargetMode.MATCH
        )
    bounds = draw(bounds_strategy)
    transformation = draw(st.sampled_from(_VALID_TRANSFORMATIONS[mode]))

    return NumericalTarget(
        name=name, mode=mode, bounds=bounds, transformation=transformation
    )


choice_values = st.one_of(
    [st.booleans(), st.integers(), st.floats(allow_nan=False), st.text()]
)
"""A strategy that generates choice values."""


@st.composite
def binary_targets(draw: st.DrawFn):
    """A strategy that generates binary targets."""
    name = draw(target_name)
    choices = draw(st.lists(choice_values, min_size=2, max_size=2, unique=True))
    return BinaryTarget(name, success_value=choices[0], failure_value=choices[1])


targets = st.one_of([binary_targets(), numerical_targets()])
"""A strategy that generates targets."""
