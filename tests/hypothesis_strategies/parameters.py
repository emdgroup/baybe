"""Hypothesis strategies for parameters."""

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
from baybe.utils.numerical import DTypeFloatNumpy

from ..hypothesis_strategies.basic import finite_floats
from .utils import intervals

decorrelations = st.one_of(
    st.booleans(),
    finite_floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
)
"""A strategy that generates decorrelation settings."""

parameter_names = st.text(min_size=1)
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
    from baybe.utils.chemistry import get_canonical_smiles

    names = draw(st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))
    substances = draw(
        st.lists(
            smiles().map(get_canonical_smiles),
            min_size=len(names),
            max_size=len(names),
            unique=True,
        )
    )
    return dict(zip(names, substances))


@st.composite
def custom_descriptors(draw: st.DrawFn):
    """Generate data for :class:`baybe.parameters.custom.CustomDiscreteParameter`."""
    index = st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True)
    cols = columns(
        names_or_number=10,
        elements=finite_floats(),
        unique=True,
        dtype=DTypeFloatNumpy,
    )
    return draw(data_frames(index=index, columns=cols))


@st.composite
def numerical_discrete_parameters(
    draw: st.DrawFn,
    min_value: float | None = None,
    max_value: float | None = None,
):
    """Generate :class:`baybe.parameters.numerical.NumericalDiscreteParameter`."""
    name = draw(parameter_names)
    values = draw(
        st.lists(
            finite_floats(
                min_value=min_value,
                max_value=max_value,
            ),
            min_size=2,
            unique=True,
        )
    )
    max_tolerance = np.diff(np.sort(values)).min() / 2
    if (max_tolerance == 0.0) or (max_tolerance != DTypeFloatNumpy(max_tolerance)):
        tolerance = 0.0
    else:
        tolerance = draw(
            finite_floats(
                min_value=0.0,
                max_value=max_tolerance,
                exclude_max=True,
            )
        )
    return NumericalDiscreteParameter(name=name, values=values, tolerance=tolerance)


@st.composite
def numerical_continuous_parameters(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.numerical.NumericalContinuousParameter`."""
    name = draw(parameter_names)
    bounds = draw(intervals(exclude_half_bounded=True, exclude_fully_unbounded=True))
    return NumericalContinuousParameter(name=name, bounds=bounds)


@st.composite
def categorical_parameters(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.categorical.CategoricalParameter`."""
    name = draw(parameter_names)
    values = draw(categories)
    encoding = draw(st.sampled_from(CategoricalEncoding))
    return CategoricalParameter(name=name, values=values, encoding=encoding)


@st.composite
def task_parameters(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.categorical.TaskParameter`."""
    name = draw(parameter_names)
    values = draw(categories)
    active_values = draw(
        st.lists(st.sampled_from(values), min_size=1, max_size=len(values), unique=True)
    )
    return TaskParameter(name=name, values=values, active_values=active_values)


@st.composite
def substance_parameters(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.substance.SubstanceParameter`."""
    name = draw(parameter_names)
    data = draw(substance_data())
    decorrelate = draw(decorrelations)
    encoding = draw(st.sampled_from(SubstanceEncoding))
    return SubstanceParameter(
        name=name, data=data, decorrelate=decorrelate, encoding=encoding
    )


@st.composite
def custom_parameters(draw: st.DrawFn):
    """Generate :class:`baybe.parameters.custom.CustomDiscreteParameter`."""
    name = draw(parameter_names)
    data = draw(custom_descriptors())
    decorrelate = draw(decorrelations)
    return CustomDiscreteParameter(name=name, data=data, decorrelate=decorrelate)


parameters = st.one_of(
    [
        numerical_discrete_parameters(),
        numerical_continuous_parameters(),
        categorical_parameters(),
        task_parameters(),
        substance_parameters(),
        custom_parameters(),
    ]
)
"""A strategy that generates parameters."""


discrete_parameters = st.one_of(
    [
        numerical_discrete_parameters(),
        categorical_parameters(),
        task_parameters(),
        substance_parameters(),
        custom_parameters(),
    ]
)
"""A strategy that generates discrete parameters."""


continuous_parameters = numerical_continuous_parameters
"""A strategy that generates continuous parameters."""
