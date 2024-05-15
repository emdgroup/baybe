"""Hypothesis strategies for kernels."""

from enum import Enum

import hypothesis.strategies as st

from baybe.kernels.basic import MaternKernel
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel

from ..hypothesis_strategies.basic import finite_floats
from ..hypothesis_strategies.priors import priors


class KernelType(Enum):
    """Taxonomy of kernel types."""

    SINGLE = "SINGLE"
    ADDITIVE = "ADDITIVE"
    PRODUCT = "PRODUCT"


matern_kernels = st.builds(
    MaternKernel,
    nu=st.sampled_from((0.5, 1.5, 2.5)),
    lengthscale_prior=st.one_of(st.none(), priors),
    lengthscale_initial_value=st.one_of(st.none(), finite_floats()),
)
"""A strategy that generates Matern kernels."""


base_kernels = st.one_of([matern_kernels])
"""A strategy that generates base kernels to be used within more complex kernels."""


@st.composite
def single_kernels(draw: st.DrawFn):
    """Generate simple kernels without kernel arithmetic (except scaling)."""
    base_kernel = draw(base_kernels)
    add_scale = draw(st.booleans())
    if add_scale:
        return ScaleKernel(
            base_kernel=base_kernel,
            outputscale_prior=draw(st.one_of(st.none(), priors)),
            outputscale_initial_value=draw(
                st.one_of(st.none(), finite_floats()),
            ),
        )
    else:
        return base_kernel


@st.composite
def kernels(draw: st.DrawFn):
    """Generate :class:`baybe.kernels.basic.Kernel`."""
    kernel_type = draw(st.sampled_from(KernelType))

    if kernel_type is KernelType.SINGLE:
        return draw(single_kernels())

    base_kernels = draw(st.lists(single_kernels()))
    if kernel_type is KernelType.ADDITIVE:
        return AdditiveKernel(base_kernels)
    if kernel_type is KernelType.PRODUCT:
        return ProductKernel(base_kernels)
