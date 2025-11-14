"""Composite kernels (that is, kernels composed of other kernels)."""

import gc
from functools import reduce
from operator import add, mul

import numpy as np
from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_iterable, ge, gt, instance_of, min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.kernels.base import CompositeKernel, Kernel
from baybe.priors.base import Prior
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float


@define(frozen=True)
class ScaleKernel(CompositeKernel):
    """A kernel for decorating existing kernels with an outputscale."""

    base_kernel: Kernel = field(validator=instance_of(Kernel))
    """The base kernel that is being decorated."""

    outputscale_prior: Prior | None = field(
        default=None, validator=optional_v(instance_of(Prior))
    )
    """An optional prior on the output scale."""

    outputscale_initial_value: float | None = field(
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0.0)]),
    )
    """An optional initial value for the output scale."""

    @override
    def to_gpytorch(self, *args, **kwargs):
        import torch

        from baybe.utils.torch import DTypeFloatTorch

        gpytorch_kernel = super().to_gpytorch(*args, **kwargs)
        if (initial_value := self.outputscale_initial_value) is not None:
            gpytorch_kernel.outputscale = torch.tensor(
                initial_value, dtype=DTypeFloatTorch
            )
        return gpytorch_kernel


@define(frozen=True)
class AdditiveKernel(CompositeKernel):
    """A kernel representing the sum of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be summed."""

    @override
    def to_gpytorch(self, *args, **kwargs):
        return reduce(add, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


@define(frozen=True)
class ProductKernel(CompositeKernel):
    """A kernel representing the product of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be multiplied."""

    @override
    def to_gpytorch(self, *args, **kwargs):
        return reduce(mul, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


_field_n_projections = field(
    default=None, validator=optional_v([instance_of(int), ge(0)]), kw_only=True
)
"""Attrs field for :attr:`baybe.kernels.ProjectionKernel.n_projections`."""

_field_projection_matrix = field(
    default=None, converter=optional_c(np.asarray), kw_only=True
)
"""Attrs field for :attr:`baybe.kernels.ProjectionKernel.projection_matrix`."""

_field_learn_projection = field(
    default=False, validator=instance_of(bool), kw_only=True
)
"""Attrs field for :attr:`baybe.kernels.ProjectionKernel.learn_projection`."""


@define(frozen=True)
class ProjectionKernel(CompositeKernel):
    """A random projection kernel for dimensionality reduction."""

    base_kernel: Kernel = field(validator=instance_of(Kernel))
    """The kernel to apply after projection."""

    n_projections: int | None = _field_n_projections
    """The number of projections used (i.e. dimensionality of the projection space).

    Must be provided if no projection matrix is specified.
    """

    projection_matrix: np.ndarray | None = _field_projection_matrix
    """A pre-specified projection matrix.

    Must be provided if no number of projections is specified.
    """

    learn_projection: bool = _field_learn_projection
    """Boolean specifying if the projection matrix should be learned.

    If a projection matrix is provided and learning is activated, the provided matrix
    is used as initial value.
    """

    @projection_matrix.validator
    def _validate_projection_matrix(self, attribute: field, value: np.ndarray | None):
        if value is None:
            if self.n_projections is None:
                raise ValueError(
                    "Either a projection matrix or the number of projections "
                    "must be specified."
                )
            return
        if value.ndim != 2:
            raise ValueError(
                f"The projection matrix must be 2-dimensional, "
                f"but has shape {value.shape}."
            )

    @override
    def to_gpytorch(self, **kwargs):
        from baybe.kernels._gpytorch import ProjectionKernel as GPytorchProjectionKernel

        gpytorch_kernel = self.base_kernel.to_gpytorch(ard_num_dims=self.n_projections)
        return GPytorchProjectionKernel(
            gpytorch_kernel,
            n_projections=self.n_projections,
            projection_matrix=self.projection_matrix,
            learn_projection=self.learn_projection,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
