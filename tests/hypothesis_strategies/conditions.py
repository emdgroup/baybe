"""Hypothesis strategies for conditions."""

from typing import Any

import hypothesis.strategies as st

from baybe.constraints import SubSelectionCondition, ThresholdCondition
from baybe.constraints.conditions import (
    _threshold_operators,
    _valid_tolerance_operators,
)
from tests.hypothesis_strategies.basic import finite_floats, positive_finite_floats


def sub_selection_conditions(superset: list[Any] | None = None):
    """Generate :class:`baybe.constraints.conditions.SubSelectionCondition`."""
    if superset is None:
        element_strategy = st.text()
    else:
        element_strategy = st.sampled_from(superset)
    return st.builds(
        SubSelectionCondition, st.lists(element_strategy, unique=True, min_size=1)
    )


@st.composite
def threshold_conditions(draw: st.DrawFn):
    """Generate :class:`baybe.constraints.conditions.ThresholdCondition`."""
    threshold = draw(finite_floats())
    operator = draw(st.sampled_from(list(_threshold_operators.keys())))
    if operator in _valid_tolerance_operators:
        tolerance = draw(positive_finite_floats())
    else:
        tolerance = None
    return ThresholdCondition(
        threshold=threshold, operator=operator, tolerance=tolerance
    )
