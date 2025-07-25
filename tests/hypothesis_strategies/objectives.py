"""Hypothesis strategies for objectives."""

import hypothesis.strategies as st

from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.pareto import ParetoObjective
from baybe.objectives.single import SingleTargetObjective

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.targets import numerical_targets

_target_lists = st.lists(numerical_targets(), min_size=2, unique_by=lambda t: t.name)


def single_target_objectives():
    """Generate :class:`baybe.objectives.single.SingleTargetObjective`."""
    return st.builds(SingleTargetObjective, target=numerical_targets())


@st.composite
def desirability_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.desirability.DesirabilityObjective`."""
    targets = draw(_target_lists)
    weights = draw(
        st.lists(
            finite_floats(min_value=0.0, exclude_min=True),
            min_size=len(targets),
            max_size=len(targets),
        )
    )
    scalarizer = draw(st.sampled_from(Scalarizer))
    if require_normalization := draw(st.booleans()):
        targets = [t.clamp(0, 1) for t in targets]
    return DesirabilityObjective(targets, weights, scalarizer, require_normalization)


@st.composite
def pareto_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.pareto.ParetoObjective`."""
    return ParetoObjective(draw(_target_lists))
