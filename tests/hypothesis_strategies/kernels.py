"""Hypothesis strategies for kernels."""

import hypothesis.strategies as st

from baybe.kernels import MaternKernel

from ..hypothesis_strategies.priors import priors

matern_kernels = st.builds(
    MaternKernel,
    nu=st.sampled_from((0.5, 1.5, 2.5)),
    lengthscale_prior=st.one_of(st.none(), priors),
)
"""A strategy that generates Matern kernels."""
