"""Hypothesis strategies for priors."""

import hypothesis.strategies as st

from baybe.kernels.priors import GammaPrior

gamma_priors = st.builds(
    GammaPrior,
    st.floats(min_value=0, exclude_min=True),
    st.floats(min_value=0, exclude_min=True),
)
"""A strategy that generates Gamma priors."""

priors = st.one_of(
    [
        gamma_priors,
    ]
)
"""A strategy that generates priors."""
