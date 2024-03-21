"""Hypothesis strategies for targets."""

import hypothesis.strategies as st

from baybe.targets.enum import TargetMode
from baybe.targets.numerical import _VALID_TRANSFORMATIONS, NumericalTarget

from .utils import intervals

target_name = st.text(min_size=1)
"""A strategy that generates target names."""


@st.composite
def numerical_targets(draw: st.DrawFn):
    """Generate :class:`baybe.targets.numerical.NumericalTarget`."""
    name = draw(target_name)
    mode = draw(st.sampled_from(TargetMode))
    bounds = draw(
        intervals(
            exclude_half_bounded=True, exclude_fully_unbounded=mode is TargetMode.MATCH
        )
    )
    transformation = draw(st.sampled_from(_VALID_TRANSFORMATIONS[mode]))

    return NumericalTarget(
        name=name, mode=mode, bounds=bounds, transformation=transformation
    )


targets = numerical_targets()
"""A strategy that generates targets."""
