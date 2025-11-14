"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol

import numpy as np
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.composite import (
    ProjectionKernel,
    _field_learn_projection,
    _field_n_projections,
    _field_projection_matrix,
)
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor


def to_kernel_factory(x: Kernel | KernelFactory, /) -> KernelFactory:
    """Wrap a kernel into a plain kernel factory (with factory passthrough)."""
    return x.to_factory() if isinstance(x, Kernel) else x


class KernelFactory(Protocol):
    """A protocol defining the interface expected for kernel factories."""

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        """Create a :class:`baybe.kernels.base.Kernel` for the given DOE context."""
        ...


@define(frozen=True)
class PlainKernelFactory(KernelFactory, SerialMixin):
    """A trivial factory that returns a fixed pre-defined kernel upon request."""

    kernel: Kernel = field(validator=instance_of(Kernel))
    """The fixed kernel to be returned by the factory."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return self.kernel


@define(frozen=True)
class ProjectionKernelFactory(KernelFactory, SerialMixin):
    """A factory producing projected kernels."""

    base_kernel_factory: KernelFactory = field(
        alias="kernel_or_factory", converter=to_kernel_factory
    )

    n_projections: int | None = _field_n_projections
    """See :class:`baybe.kernels.ProjectionKernel`."""

    projection_matrix: np.ndarray | None = _field_projection_matrix
    """See :class:`baybe.kernels.ProjectionKernel`."""

    learn_projection: bool = _field_learn_projection
    """See :class:`baybe.kernels.ProjectionKernel`."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        base_kernel = self.base_kernel_factory(searchspace, train_x, train_y)
        return ProjectionKernel(
            base_kernel=base_kernel,
            n_projections=self.n_projections,
            projection_matrix=self.projection_matrix,
            learn_projection=self.learn_projection,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
