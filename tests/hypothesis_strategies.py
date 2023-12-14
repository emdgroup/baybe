"""Hypothesis strategies."""

import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.pandas import columns, data_frames

from baybe.parameters.categorical import (
    CategoricalEncoding,
    CategoricalParameter,
    TaskParameter,
)
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.substance import SubstanceEncoding, SubstanceParameter
from baybe.utils.chemistry import get_canonical_smiles
from baybe.utils.numeric import DTypeFloatNumpy

_largest_lower_interval = np.nextafter(
    np.nextafter(np.inf, 0, dtype=DTypeFloatNumpy), 0, dtype=DTypeFloatNumpy
)
"""
The largest possible value for the lower end of a continuous interval such that there
still exists a larger but finite number for the upper interval end.
"""

decorrelation = st.one_of(
    st.booleans(),
    st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
)
"""A strategy that generates decorrelation settings."""

parameter_name = st.text(min_size=1)
"""A strategy that generates parameter names."""

categories = st.lists(st.text(min_size=1), min_size=2, unique=True)
"""A strategy that generates parameter categories."""


@st.composite
def smiles(draw: st.DrawFn):
    """Generate short SMILES strings."""
    n_atoms = draw(st.integers(min_value=0, max_value=19))
    string = "C"
    for _ in range(n_atoms):
        next_atom = draw(st.sampled_from("CNO")) if string[-1] == "C" else "C"
        string += next_atom
    return string


@st.composite
def substance_data(draw: st.DrawFn):
    """Generate data for :class:`baybe.parameters.substance.SubstanceParameter`."""
    names = draw(st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))
    substances = draw(
        st.lists(smiles(), min_size=len(names), max_size=len(names), unique=True)
    )
    substances = list(set(get_canonical_smiles(s) for s in substances))
    return dict(zip(names, substances))


@st.composite
def custom_descriptors(draw: st.DrawFn):
    """Generate data for :class:`baybe.parameters.custom.CustomDiscreteParameter`."""
    index = st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True)
    cols = columns(
        names_or_number=10,
        elements=st.floats(allow_nan=False, allow_infinity=False),
        unique=True,
        dtype=DTypeFloatNumpy,
    )
    return draw(data_frames(index=index, columns=cols))


@st.composite
def numerical_discrete_parameter(
    draw: st.DrawFn,
):
    """Generate :class:`baybe.parameters.numerical.NumericalDiscreteParameter`."""
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
    max_tolerance = np.diff(np.sort(values)).min() / 2
    if max_tolerance == 0.0:
        tolerance = 0.0
    else:
        tolerance = draw(
            st.floats(
                min_value=0.0,
                max_value=max_tolerance,
                allow_nan=False,
                exclude_max=True,
            )
        )
    return NumericalDiscreteParameter(name=name, values=values, tolerance=tolerance)


@st.composite
def numerical_continuous_parameter(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.numerical.NumericalContinuousParameter`."""
    name = draw(parameter_name)
    lower = draw(st.floats(max_value=_largest_lower_interval, allow_infinity=False))
    upper = draw(st.floats(min_value=lower, exclude_min=True, allow_infinity=False))
    return NumericalContinuousParameter(name=name, bounds=(lower, upper))


@st.composite
def categorical_parameter(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.categorical.CategoricalParameter`."""
    name = draw(parameter_name)
    values = draw(categories)
    encoding = draw(st.sampled_from(CategoricalEncoding))
    return CategoricalParameter(name=name, values=values, encoding=encoding)


@st.composite
def task_parameter(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.categorical.TaskParameter`."""
    name = draw(parameter_name)
    values = draw(categories)
    active_values = draw(
        st.lists(st.sampled_from(values), min_size=1, max_size=len(values), unique=True)
    )
    return TaskParameter(name=name, values=values, active_values=active_values)


@st.composite
def substance_parameter(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.substance.SubstanceParameter`."""
    name = draw(parameter_name)
    data = draw(substance_data())
    decorrelate = draw(decorrelation)
    encoding = draw(st.sampled_from(SubstanceEncoding))
    return SubstanceParameter(
        name=name, data=data, decorrelate=decorrelate, encoding=encoding
    )


@st.composite
def custom_parameter(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.custom.CustomDiscreteParameter`."""
    name = draw(parameter_name)
    data = draw(custom_descriptors())
    decorrelate = draw(decorrelation)
    return CustomDiscreteParameter(name=name, data=data, decorrelate=decorrelate)


parameter = st.one_of(
    [
        numerical_discrete_parameter(),
        numerical_continuous_parameter(),
        categorical_parameter(),
        task_parameter(),
        substance_parameter(),
        custom_parameter(),
    ]
)
"""A strategy that generates parameters."""
