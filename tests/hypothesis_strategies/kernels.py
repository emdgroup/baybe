"""Hypothesis strategies for kernels."""

from collections.abc import Sequence
from enum import Enum

import hypothesis.strategies as st

from baybe.kernels.basic import (
    IndexKernel,
    LinearKernel,
    MaternKernel,
    PeriodicKernel,
    PiecewisePolynomialKernel,
    PolynomialKernel,
    PositiveIndexKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
)
from baybe.kernels.composite import AdditiveKernel, ProductKernel, ScaleKernel
from tests.hypothesis_strategies.basic import positive_finite_floats
from tests.hypothesis_strategies.parameters import parameter_names
from tests.hypothesis_strategies.priors import priors


class KernelType(Enum):
    """Taxonomy of kernel types."""

    SINGLE = "SINGLE"
    ADDITIVE = "ADDITIVE"
    PRODUCT = "PRODUCT"


def active_parameter_names(names: Sequence[str] | None = None):
    """A strategy generating optional parameter names for kernels to operate on."""
    if names is None:
        return st.one_of(
            st.none(), st.lists(parameter_names, min_size=1, max_size=5, unique=True)
        )
    return st.just(names)


def linear_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates linear kernels."""
    return st.builds(
        LinearKernel,
        parameter_names=active_parameter_names(parameter_names),
        variance_prior=st.one_of(st.none(), priors()),
        variance_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def matern_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates Matern kernels."""
    return st.builds(
        MaternKernel,
        parameter_names=active_parameter_names(parameter_names),
        nu=st.sampled_from((0.5, 1.5, 2.5)),
        lengthscale_prior=st.one_of(st.none(), priors()),
        lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def periodic_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates periodic kernels."""
    return st.builds(
        PeriodicKernel,
        parameter_names=active_parameter_names(parameter_names),
        lengthscale_prior=st.one_of(st.none(), priors()),
        lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
        period_length_prior=st.one_of(st.none(), priors()),
        period_length_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def piecewise_polynomial_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates piecewise polynomial kernels."""
    return st.builds(
        PiecewisePolynomialKernel,
        parameter_names=active_parameter_names(parameter_names),
        q=st.integers(min_value=0, max_value=3),
        lengthscale_prior=st.one_of(st.none(), priors()),
        lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def polynomial_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates polynomial kernels."""
    return st.builds(
        PolynomialKernel,
        parameter_names=active_parameter_names(parameter_names),
        power=st.integers(min_value=0),
        offset_prior=st.one_of(st.none(), priors()),
        offset_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def rbf_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates radial basis function (RBF) kernels."""
    return st.builds(
        RBFKernel,
        parameter_names=active_parameter_names(parameter_names),
        lengthscale_prior=st.one_of(st.none(), priors()),
        lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def rff_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates random Fourier features (RFF) kernels."""
    return st.builds(
        RFFKernel,
        parameter_names=active_parameter_names(parameter_names),
        num_samples=st.integers(min_value=1),
        lengthscale_prior=st.one_of(st.none(), priors()),
        lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


def rq_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates rational quadratic (RQ) kernels."""
    return st.builds(
        RQKernel,
        parameter_names=active_parameter_names(parameter_names),
        lengthscale_prior=st.one_of(st.none(), priors()),
        lengthscale_initial_value=st.one_of(st.none(), positive_finite_floats()),
    )


@st.composite
def index_kernels(
    draw: st.DrawFn,
    parameter_names: Sequence[str] | None = None,
):
    """A strategy that generates index kernels."""
    num_tasks = draw(st.integers(min_value=2, max_value=5))
    rank = draw(st.integers(min_value=1, max_value=num_tasks))
    names = draw(active_parameter_names(parameter_names))
    if draw(st.booleans()):
        return PositiveIndexKernel(
            parameter_names=names,
            num_tasks=num_tasks,
            rank=rank,
        )
    return IndexKernel(parameter_names=names, num_tasks=num_tasks, rank=rank)


def base_kernels(parameter_names: Sequence[str] | None = None):
    """A strategy that generates base kernels to be used within more complex kernels."""
    return st.one_of(
        [
            matern_kernels(parameter_names),  # on top because it is the default
            linear_kernels(parameter_names),
            rbf_kernels(parameter_names),
            rq_kernels(parameter_names),
            rff_kernels(parameter_names),
            index_kernels(parameter_names=parameter_names),
            piecewise_polynomial_kernels(parameter_names),
            polynomial_kernels(parameter_names),
            periodic_kernels(parameter_names),
        ]
    )


@st.composite
def single_kernels(
    draw: st.DrawFn,
    parameter_names: Sequence[str] | None = None,
):
    """Generate single kernels (i.e., without kernel arithmetic, except scaling)."""
    base_kernel = draw(base_kernels(parameter_names))
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
def kernels(
    draw: st.DrawFn,
    parameter_names: Sequence[str] | None = None,
):
    """Generate :class:`baybe.kernels.base.Kernel`."""
    kernel_type = draw(st.sampled_from(KernelType))

    if kernel_type is KernelType.SINGLE:
        return draw(single_kernels(parameter_names=parameter_names))

    base_kernels = draw(
        st.lists(single_kernels(parameter_names=parameter_names), min_size=2)
    )
    if kernel_type is KernelType.ADDITIVE:
        return AdditiveKernel(base_kernels)
    if kernel_type is KernelType.PRODUCT:
        return ProductKernel(base_kernels)
