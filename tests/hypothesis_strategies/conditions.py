"""Hypothesis strategies for conditions."""

from typing import Any

import hypothesis.strategies as st

from baybe.constraints import SubSelectionCondition, ThresholdCondition
from baybe.constraints.conditions import _THRESHOLD_OPERATORS, _TOLERANCE_OPERATORS
from tests.hypothesis_strategies.basic import finite_floats, positive_finite_floats


@st.composite
def sub_selection_conditions(draw: st.DrawFn, superset: list[Any] | None = None):
    """Generate :class:`baybe.constraints.conditions.SubSelectionCondition`."""
    if superset is None:
        element_strategy = draw(
            st.sampled_from([st.text(), finite_floats(), st.booleans()])
        )
    else:
        # A valid superset implies homogeneous, supported entries, so every
        # selection drawn from it is constructible.
        SubSelectionCondition(superset)
        element_strategy = st.sampled_from(superset)

    return SubSelectionCondition(
        draw(st.lists(element_strategy, unique=True, min_size=1))
    )


@st.composite
def threshold_conditions(draw: st.DrawFn):
    """Generate :class:`baybe.constraints.conditions.ThresholdCondition`."""
    threshold = draw(finite_floats())
    operator = draw(st.sampled_from(list(_THRESHOLD_OPERATORS.keys())))
    if operator in _TOLERANCE_OPERATORS:
        tolerance = draw(positive_finite_floats())
    else:
        tolerance = None
    return ThresholdCondition(
        threshold=threshold, operator=operator, tolerance=tolerance
    )
