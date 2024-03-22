"""Hypothesis strategies for constraints."""

from functools import partial
from typing import Any, List, Optional, Type, Union

import hypothesis.strategies as st

from baybe.constraints.conditions import (
    SubSelectionCondition,
    ThresholdCondition,
    _valid_logic_combiners,
)
from baybe.constraints.continuous import (
    ContinuousLinearEqualityConstraint,
    ContinuousLinearInequalityConstraint,
)
from baybe.constraints.discrete import (
    DiscreteExcludeConstraint,
    DiscreteLinkedParametersConstraint,
    DiscreteNoLabelDuplicatesConstraint,
    DiscreteProductConstraint,
    DiscreteSumConstraint,
)
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalDiscreteParameter

from .parameters import discrete_parameters

_disc_params = st.lists(discrete_parameters, min_size=1, unique_by=lambda p: p.name)


def sub_selection_conditions(superset: Optional[List[Any]] = None):
    """Generate :class:`baybe.constraints.conditions.SubSelectionCondition`."""
    if superset is None:
        element_strategy = st.text()
    else:
        element_strategy = st.sampled_from(superset)
    return st.builds(SubSelectionCondition, st.lists(element_strategy, unique=True))


def threshold_conditions():
    """Generate :class:`baybe.constraints.conditions.ThresholdCondition`."""
    return st.builds(
        ThresholdCondition, threshold=st.floats(allow_infinity=False, allow_nan=False)
    )


@st.composite
def discrete_excludes_constraints(
    draw: st.DrawFn, parameters: Optional[List[DiscreteParameter]] = None
):
    """Generate :class:`baybe.constraints.discrete.DiscreteExcludeConstraint`."""
    if parameters is None:
        parameters = draw(_disc_params)

    parameter_names = [p.name for p in parameters]

    # Threshold conditions only make sense for numerical parameters
    conditions = [
        draw(st.one_of([sub_selection_conditions(p.values), threshold_conditions()]))
        if isinstance(p, NumericalDiscreteParameter)
        else draw(sub_selection_conditions(p.values))
        for p in parameters
    ]

    combiner = draw(st.sampled_from(list(_valid_logic_combiners)))
    return DiscreteExcludeConstraint(parameter_names, conditions, combiner)


def _discrete_constraints(
    constraint_type: Union[
        Type[DiscreteSumConstraint],
        Type[DiscreteProductConstraint],
        Type[DiscreteNoLabelDuplicatesConstraint],
        Type[DiscreteLinkedParametersConstraint],
    ],
    parameter_names: Optional[List[str]] = None,
):
    """Generate discrete constraints."""
    if parameter_names is None:
        parameters = st.lists(st.text(), unique=True, min_size=1)
    else:
        assert len(parameter_names) > 0
        assert len(parameter_names) == len(set(parameter_names))
        parameters = st.just(parameter_names)

    if constraint_type in [DiscreteSumConstraint, DiscreteProductConstraint]:
        return st.builds(constraint_type, parameters, threshold_conditions())
    else:
        return st.builds(constraint_type, parameters)


discrete_sum_constraints = partial(_discrete_constraints, DiscreteSumConstraint)
"""Generate :class:`baybe.constraints.discrete.DiscreteSumConstraint`."""

discrete_product_constraints = partial(_discrete_constraints, DiscreteProductConstraint)
"""Generate :class:`baybe.constraints.discrete.DiscreteProductConstraint`."""

discrete_no_label_duplicates_constraints = partial(
    _discrete_constraints, DiscreteNoLabelDuplicatesConstraint
)
"""Generate :class:`baybe.constraints.discrete.DiscreteNoLabelDuplicatesConstraint`."""

discrete_linked_parameters_constraints = partial(
    _discrete_constraints, DiscreteLinkedParametersConstraint
)
"""Generate :class:`baybe.constraints.discrete.DiscreteLinkedParametersConstraint`."""


@st.composite
def continuous_linear_equality_constraints(
    draw: st.DrawFn, parameters: Optional[List[DiscreteParameter]] = None
):
    """Generate :class:`baybe.constraints.continuous.ContinuousLinearEqualityConstraint`."""  # noqa:E501
    if parameters is None:
        parameters = draw(_disc_params)

    parameter_names = [p.name for p in parameters]
    coefficients = draw(
        st.lists(
            st.floats(allow_nan=False),
            min_size=len(parameter_names),
            max_size=len(parameter_names),
        )
    )
    rhs = draw(st.floats(allow_nan=False))
    return ContinuousLinearEqualityConstraint(parameter_names, coefficients, rhs)


@st.composite
def continuous_linear_inequality_constraints(
    draw: st.DrawFn, parameters: Optional[List[DiscreteParameter]] = None
):
    """Generate :class:`baybe.constraints.continuous.ContinuousLinearInequalityConstraint`."""  # noqa:E501
    if parameters is None:
        parameters = draw(_disc_params)

    parameter_names = [p.name for p in parameters]
    coefficients = draw(
        st.lists(
            st.floats(allow_nan=False),
            min_size=len(parameter_names),
            max_size=len(parameter_names),
        )
    )
    rhs = draw(st.floats(allow_nan=False))
    return ContinuousLinearInequalityConstraint(parameter_names, coefficients, rhs)