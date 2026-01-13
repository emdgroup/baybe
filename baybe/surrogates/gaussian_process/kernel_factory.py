"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Protocol

from attr.converters import optional as optional_c
from attrs import define, field
from attrs.validators import instance_of
from attrs.validators import optional as optional_v
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


class DiscreteFidelityKernelFactory(KernelFactory):
    """A protocol for discrete fidelity kernel factories."""

    num_tasks: int = field(converter=int, validator=instance_of(int))

    rank: int | None = field(
        converter=optional_c(int), validator=optional_v(instance_of(int))
    )


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


def to_kernel_factory(x: Kernel | KernelFactory, /) -> KernelFactory:
    """Wrap a kernel into a plain kernel factory (with factory passthrough)."""
    return x.to_factory() if isinstance(x, Kernel) else x


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
