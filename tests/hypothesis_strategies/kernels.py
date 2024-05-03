"""Hypothesis strategies for kernels."""

import hypothesis.strategies as st

from baybe.kernels import MaternKernel, ScaleKernel

from ..hypothesis_strategies.priors import priors

matern_kernels = st.builds(
    MaternKernel,
    nu=st.sampled_from((0.5, 1.5, 2.5)),
    lengthscale_prior=st.one_of(st.none(), priors),
)
"""A strategy that generates Matern kernels."""

base_kernels = matern_kernels
"""A strategy that generates base kernels to be used within more complex kernels."""

scale_kernels = st.builds(
    ScaleKernel,
    base_kernel=base_kernels,
    outputscale_prior=st.one_of(st.none(), priors),
)
"""A strategy that generates scale kernels."""

kernels = st.one_of([base_kernels, scale_kernels])
"""A strategy that generates kernels."""
