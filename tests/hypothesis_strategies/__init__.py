"""Hypothesis strategies."""

from hypothesis import strategies as st

from tests.hypothesis_strategies.acquisition import acquisition_functions
from tests.hypothesis_strategies.constraints import constraints
from tests.hypothesis_strategies.kernels import kernels
from tests.hypothesis_strategies.objectives import objectives
from tests.hypothesis_strategies.parameters import parameters
from tests.hypothesis_strategies.priors import priors
from tests.hypothesis_strategies.targets import targets

baybe_objects = st.one_of(
    [
        parameters,
        targets,
        objectives,
        kernels(),
        priors(),
        acquisition_functions,
        constraints,
    ]
)
"""A strategy that generates BayBE objects."""
