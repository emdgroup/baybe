"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol

from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor


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


# Internal kernel
@define(frozen=True)
class ProjectionAdapterFactory(KernelFactory, SerialMixin):
    """A wrapper object for calling Projection Kernel."""

    default_factory: KernelFactory
    proj_dim: int
    learn_projection: bool = False

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        """Wrap a Projection Kernel for lazy eval in GP."""
        from baybe.kernels.base import ProjectionKernel

        base_kernel = self.default_factory(searchspace, train_x, train_y)

        proj_kernel = ProjectionKernel(
            base_kernel=base_kernel,
            proj_dim=self.proj_dim,
            learn_projection=self.learn_projection,
        )

        return proj_kernel


def to_kernel_factory(x: Kernel | KernelFactory, /) -> KernelFactory:
    """Wrap a kernel into a plain kernel factory (with factory passthrough)."""
    return x.to_factory() if isinstance(x, Kernel) else x


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
