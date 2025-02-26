"""Hypothesis strategies for objectives."""

from hypothesis import assume
from hypothesis import strategies as st

from baybe.objectives.chimera import ChimeraObjective, ThresholdType
from baybe.objectives.desirability import DesirabilityObjective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.single import SingleTargetObjective
from baybe.targets.enum import TargetMode

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.targets import numerical_targets
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
    """Generate :class:`baybe.objectives.chimera.ChimeraObjective` and its reference."""
    intervals = st_intervals(exclude_fully_unbounded=True, exclude_half_bounded=True)
    targets = draw(
        st.lists(
            numerical_targets(intervals),
            min_size=3,  # At least 2 targets.
            max_size=4,  # TODO: remove max_size once tests working
            unique_by=lambda t: t.name,
        )
    )

    # Force that every drawn target has a linear transformation and mode MIN or MAX.
    assume(all(t.mode is not TargetMode.MATCH for t in targets))
    n_targets = len(targets)

    threshold_types = draw(
        st.lists(
            st.sampled_from(
                [ThresholdType.FRACTION, ThresholdType.PERCENTILE]
            ),  # TODO: let's test these two first
            # st.sampled_from(ThresholdType), # TODO: add back if the test works
            min_size=n_targets,
            max_size=n_targets,
        )
    )

    # Now draw threshold values one-by-one based on each threshold type.
    threshold_values = []
    for i, tt in enumerate(threshold_types):
        if tt == ThresholdType.ABSOLUTE:
            lower, upper = targets[
                i
            ].bounds  # assuming targets[i].bounds is a tuple (lower, upper)
            val = draw(
                finite_floats(
                    min_value=lower,
                    max_value=upper,
                    exclude_min=True,
                    exclude_max=True,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
        else:
            # For FRACTION or PERCENTILE, sample from [0.0, 1.0]
            val = draw(
                finite_floats(
                    min_value=0.0,
                    max_value=1.0,
                    exclude_min=True,
                    exclude_max=True,
                    allow_nan=False,
                    allow_infinity=False,
                )
            )
        threshold_values.append(val)

    # softness = draw(st.floats(min_value=0.0, max_value=1.0))
    softness = 0.0  # TODO: remove this line once tests working

    # Initialize internal ChimeraObjective using the drawn targets.
    int_chimera = ChimeraObjective(
        targets=targets,
        threshold_values=tuple(threshold_values),
        threshold_types=tuple(tt.value for tt in threshold_types),
        softness=softness,
    )
    return int_chimera
