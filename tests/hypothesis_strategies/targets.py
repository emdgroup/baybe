"""Hypothesis strategies for targets."""

from typing import Optional

import hypothesis.strategies as st

from baybe.targets.enum import TargetMode
from baybe.targets.numerical import _VALID_TRANSFORMATIONS, NumericalTarget
from baybe.utils.interval import Interval

from .utils import intervals as st_intervals

target_name = st.text(min_size=1)
"""A strategy that generates target names."""


@st.composite
def numerical_targets(
    draw: st.DrawFn, bounds_strategy: Optional[st.SearchStrategy[Interval]] = None
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


targets = numerical_targets()
"""A strategy that generates targets."""
