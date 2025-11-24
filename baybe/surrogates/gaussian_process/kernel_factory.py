"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol

from attrs import define, field
from attrs.validators import ge, instance_of
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.kernels.composite import ProjectionKernel
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.surrogates.gaussian_process.presets.default import DefaultKernelFactory

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
        alias="kernel_or_factory",
        factory=DefaultKernelFactory,
        converter=to_kernel_factory,
    )
    """The factory producing the base kernel applied to the projected inputs."""

    n_projections: int = field(validator=[instance_of(int), ge(1)])
    """The number of projections to be used."""

    learn_projection: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """Boolean specifying if the projection matrix should be learned."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        base_kernel = self.base_kernel_factory(searchspace, train_x, train_y)
        return ProjectionKernel(
            base_kernel=base_kernel,
            projection_matrix=projection_matrix,
            learn_projection=self.learn_projection,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
