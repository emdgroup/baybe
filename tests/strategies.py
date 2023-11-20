"""Hypothesis strategies."""
import random

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume

from baybe.exceptions import NumericalUnderflowError
from baybe.parameters.categorical import (
    CategoricalEncoding,
    CategoricalParameter,
    TaskParameter,
)
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.utils import DTypeFloatNumpy

_largest_lower_interval = np.nextafter(
    np.nextafter(np.inf, 0, dtype=DTypeFloatNumpy), 0, dtype=DTypeFloatNumpy
)
"""
The largest possible value for the lower end of a continuous interval such that there
still exists a larger but finite number for the upper interval end.
"""

parameter_name = st.text(min_size=1)
"""A strategy that creates parameter names."""

categories = st.lists(st.text(min_size=1), min_size=2, unique=True)
"""A strategy that creates parameter categories."""


@st.composite
def numerical_discrete_parameter(  # pylint: disable=inconsistent-return-statements
    draw: st.DrawFn,
):
    """Generates class:`baybe.parameters.numerical.NumericalDiscreteParameter`."""
    name = draw(parameter_name)
    values = draw(
        st.lists(
            st.one_of(
                st.integers(),
                st.floats(allow_infinity=False, allow_nan=False),
            ),
            min_size=2,
            unique=True,
        )
    )

    # Reject examples where the tolerance validator cannot be satisfied
    try:
        return NumericalDiscreteParameter(name=name, values=values)
    except NumericalUnderflowError:
        assume(False)


@st.composite
def numerical_continuous_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.numerical.NumericalContinuousParameter`."""
    name = draw(parameter_name)
    lower = draw(st.floats(max_value=_largest_lower_interval, allow_infinity=False))
    upper = draw(st.floats(min_value=lower, exclude_min=True, allow_infinity=False))
    return NumericalContinuousParameter(name=name, bounds=(lower, upper))


@st.composite
def categorical_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.categorical.CategoricalParameter`."""
    name = draw(parameter_name)
    values = draw(categories)
    encoding = draw(st.sampled_from(CategoricalEncoding))
    return CategoricalParameter(name=name, values=values, encoding=encoding)


@st.composite
def task_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.categorical.TaskParameter`."""
    name = draw(parameter_name)
    values = draw(categories)
    active_values = random.sample(values, random.randint(0, len(values)))
    return TaskParameter(name=name, values=values, active_values=active_values)


parameter = st.one_of(
    [
        numerical_discrete_parameter(),
        numerical_continuous_parameter(),
        categorical_parameter(),
        task_parameter(),
    ]
)
"""A strategy that creates parameters."""
