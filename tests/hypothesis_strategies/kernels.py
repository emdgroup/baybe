"""Hypothesis strategies for kernels."""

from enum import Enum

import hypothesis.strategies as st

from baybe.kernels.basic import (
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
)
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel

from ..hypothesis_strategies.basic import positive_finite_floats
from ..hypothesis_strategies.priors import priors


class KernelType(Enum):
    """Taxonomy of kernel types."""

    SINGLE = "SINGLE"
    ADDITIVE = "ADDITIVE"
    PRODUCT = "PRODUCT"


linear_kernels = st.builds(
    LinearKernel,
    variance_prior=st.one_of(st.none(), priors()),
    variance_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates linear kernels."""

matern_kernels = st.builds(
    MaternKernel,
    nu=st.sampled_from((0.5, 1.5, 2.5)),
    lengthscale_prior=st.one_of(st.none(), priors()),
    lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates Matern kernels."""

periodic_kernels = st.builds(
    PeriodicKernel,
    lengthscale_prior=st.one_of(st.none(), priors()),
    lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
    period_length_prior=st.one_of(st.none(), priors()),
    period_length_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates periodic kernels."""

piecewise_polynomial_kernels = st.builds(
    PiecewisePolynomialKernel,
    q=st.integers(min_value=0, max_value=3),
    lengthscale_prior=st.one_of(st.none(), priors()),
    lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates piecewise polynomial kernels."""

polynomial_kernels = st.builds(
    PolynomialKernel,
    power=st.integers(min_value=0),
    offset_prior=st.one_of(st.none(), priors()),
    offset_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates polynomial kernels."""

rbf_kernels = st.builds(
    RBFKernel,
    lengthscale_prior=st.one_of(st.none(), priors()),
    lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates radial basis function (RBF) kernels."""

rff_kernels = st.builds(
    RFFKernel,
    num_samples=st.integers(min_value=1),
    lengthscale_prior=st.one_of(st.none(), priors()),
    lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates random Fourier features (RFF) kernels."""

rq_kernels = st.builds(
    RQKernel,
    lengthscale_prior=st.one_of(st.none(), priors()),
    lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
)
"""A strategy that generates rational quadratic (RQ) kernels."""

base_kernels = st.one_of(
    [
        matern_kernels,  # on top because it is the default for many use cases
        linear_kernels,
        rbf_kernels,
        rq_kernels,
        rff_kernels,
        piecewise_polynomial_kernels,
        polynomial_kernels,
        periodic_kernels,
    ]
)
"""A strategy that generates base kernels to be used within more complex kernels."""


@st.composite
def single_kernels(draw: st.DrawFn):
    """Generate single kernels (i.e., without kernel arithmetic, except scaling)."""
    base_kernel = draw(base_kernels)
    add_scale = draw(st.booleans())
    if add_scale:
        return ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=draw(st.one_of(st.none(), priors())),
            outputscale_initial_value=draw(
                st.one_of(st.none(), positive_finite_floats()),
            ),
        )
    else:
        return base_kernel


@st.composite
def kernels(draw: st.DrawFn):
    """Generate :class:`baybe.kernels.base.Kernel`."""
    kernel_type = draw(st.sampled_from(KernelType))

    if kernel_type is KernelType.SINGLE:
        return draw(single_kernels())

    base_kernels = draw(st.lists(single_kernels(), min_size=2))
    if kernel_type is KernelType.ADDITIVE:
        return AdditiveKernel(base_kernels)
    if kernel_type is KernelType.PRODUCT:
        return ProductKernel(base_kernels)
