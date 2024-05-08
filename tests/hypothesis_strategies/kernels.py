"""Hypothesis strategies for kernels."""

import hypothesis.strategies as st

from baybe.kernels import MaternKernel, ScaleKernel

from ..hypothesis_strategies.priors import priors

matern_kernels = st.builds(
    MaternKernel,
    nu=st.sampled_from((0.5, 1.5, 2.5)),
    lengthscale_prior=st.one_of(st.none(), priors),
    lengthscale_initial_value=st.floats(min_value=0, exclude_min=True),
)
"""A strategy that generates matern kernels."""


base_kernels = st.one_of([matern_kernels])
"""A strategy that generates base kernels to be used within more complex kernels."""


@st.composite
def scale_kernels(draw: st.DrawFn):
    """Generate :class:`baybe.kernels.basic.ScaleKernel`."""
    base_kernel = draw(base_kernels)
    outputscale_priors = draw(
        st.one_of(st.none(), priors),
    )
    outputscale_initial_value = draw(
        st.floats(min_value=0, exclude_min=True),
    )
    return ScaleKernel(
        base_kernel=base_kernel,
        outputscale_prior=outputscale_priors,
        outputscale_initial_value=outputscale_initial_value,
    )


kernels = st.one_of([base_kernels, scale_kernels()])
"""A strategy that generates kernels."""
