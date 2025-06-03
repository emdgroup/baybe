"""Hypothesis strategies for objectives."""

from hypothesis import strategies as st

from baybe.objectives.chimera import ChimeraObjective, ThresholdType
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.single import SingleTargetObjective

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.targets import linear_numerical_targets, numerical_targets
from ..hypothesis_strategies.utils import intervals as st_intervals


def single_target_objectives():
    """Generate :class:`baybe.objectives.single.SingleTargetObjective`."""
    return st.builds(SingleTargetObjective, target=numerical_targets())


@st.composite
def desirability_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.desirability.DesirabilityObjective`."""
    intervals = st_intervals(exclude_fully_unbounded=True, exclude_half_bounded=True)
    targets = draw(
        st.lists(numerical_targets(intervals), min_size=2, unique_by=lambda t: t.name)
    )
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
def chimera_objectives(draw: st.DrawFn):
    """Generate :class:`baybe.objectives.chimera.chimera_objectives`."""
    # 1) Draw 3–4 distinct NumericalTargets
    # (all guaranteed LINEAR, MIN/MAX, and width ≥1e-6)
    bounded_intervals = st_intervals(
        exclude_bounded=False,
        exclude_half_bounded=True,
        exclude_fully_unbounded=True,
    )
    targets = draw(
        st.lists(
            linear_numerical_targets(bounded_intervals),
            min_size=2,
            max_size=100,
            unique_by=lambda t: t.name,
        )
    )

    n_targets = len(targets)

    # 3) Draw threshold_types (only FRACTION or ABSOLUTE for now)
    threshold_types = draw(
        st.lists(
            st.sampled_from([ThresholdType.FRACTION]),  # , ThresholdType.ABSOLUTE]),
            min_size=n_targets,
            max_size=n_targets,
        )
    )

    # 4) For each threshold_type, draw a threshold_value in the correct range
    threshold_values: list[float] = []
    for i, tt in enumerate(threshold_types):
        lo, hi = targets[i].bounds.lower, targets[i].bounds.upper
        if tt is ThresholdType.ABSOLUTE:
            val = draw(
                finite_floats(
                    min_value=lo,
                    max_value=hi,
                    exclude_min=True,
                    exclude_max=True,
                )
            )
        else:  # FRACTION
            val = draw(
                finite_floats(
                    min_value=0.0,
                    max_value=1.0,
                    exclude_min=True,
                    exclude_max=True,
                )
            )
        threshold_values.append(val)

    # 5) Fix softness = 0.0 for now (#TODO: draw later)
    softness = 0.0

    # 6) Build and return the ChimeraObjective
    int_chimera = ChimeraObjective(
        targets=tuple(targets),
        threshold_values=tuple(threshold_values),
        threshold_types=tuple(tt.value for tt in threshold_types),
        softness=softness,
    )
    return int_chimera
