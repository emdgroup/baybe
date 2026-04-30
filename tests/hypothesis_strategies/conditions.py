"""Hypothesis strategies for conditions."""

from typing import Any

import hypothesis.strategies as st

from baybe.constraints import SubSelectionCondition, ThresholdCondition
from tests.hypothesis_strategies.basic import finite_floats


def sub_selection_conditions(superset: list[Any] | None = None):
    """Generate :class:`baybe.constraints.conditions.SubSelectionCondition`."""
    if superset is None:
        element_strategy = st.text()
    else:
        element_strategy = st.sampled_from(superset)
    return st.builds(
        SubSelectionCondition, st.lists(element_strategy, unique=True, min_size=1)
    )


def threshold_conditions():
    """Generate :class:`baybe.constraints.conditions.ThresholdCondition`."""
    return st.builds(ThresholdCondition, threshold=finite_floats())
