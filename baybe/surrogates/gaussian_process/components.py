"""Component factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.searchspace import SearchSpace
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from torch import Tensor


Component = Kernel
_T = TypeVar("_T", bound=Component)


class ComponentFactory(Protocol, Generic[_T]):
    """A protocol defining the interface expected for GP component factories."""

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> _T:
        """Create a GP component for the given recommendation context."""


@define(frozen=True)
class PlainComponentFactory(ComponentFactory[_T], SerialMixin):
    """A trivial factory that returns a fixed pre-defined component upon request."""

    component: _T = field(validator=instance_of(Component))
    """The fixed component to be returned by the factory."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> _T:
        return self.component


def to_component_factory(x: Component | ComponentFactory, /) -> ComponentFactory:
    """Wrap a component into a plain component factory (with factory passthrough)."""
    return x.to_factory() if isinstance(x, Component) else x


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
