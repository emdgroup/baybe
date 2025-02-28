"""Hypothesis strategies for objectives."""

import hypothesis.strategies as st

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.targets import numerical_targets
from ..hypothesis_strategies.utils import intervals as st_intervals

_intervals = st_intervals(exclude_fully_unbounded=True, exclude_half_bounded=True)

_targets = st.lists(
    numerical_targets(_intervals), min_size=2, unique_by=lambda t: t.name
)


def single_target_objectives():
    """Generate :class:`baybe.objectives.single.SingleTargetObjective`."""
    return st.builds(SingleTargetObjective, target=numerical_targets())


@st.composite
def desirability_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.desirability.DesirabilityObjective`."""
    targets = draw(_targets)
    weights = draw(
        st.lists(
            finite_floats(min_value=0.0, exclude_min=True),
            min_size=len(targets),
            max_size=len(targets),
        )
    )
    scalarizer = draw(st.sampled_from(Scalarizer))
    return DesirabilityObjective(targets, weights, scalarizer)


@st.composite
def pareto_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.pareto.ParetoObjective`."""
    return ParetoObjective(draw(_targets))
