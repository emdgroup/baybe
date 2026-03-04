"""Component factories for the Gaussian process surrogate."""

from __future__ import annotations

import gc
import sys
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeAlias, TypeVar

from attrs import Attribute, define, field
from typing_extensions import override

from baybe.kernels.base import Kernel
from baybe.searchspace import SearchSpace
from baybe.serialization.core import block_serialization_hook, converter
from baybe.serialization.mixin import SerialMixin

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean
    from torch import Tensor

    Component: TypeAlias = Kernel | GPyTorchKernel | GPyTorchMean | GPyTorchLikelihood
else:
    # At runtime, we use only the BayBE types for serialization compatibility
    Component: TypeAlias = Kernel

_T_co = TypeVar("_T_co", bound=Component, covariant=True)


def _is_gpytorch_component_class(obj: Any, /) -> bool:
    """Check if a class is a GPyTorch component class using lazy loading."""
    if sys.modules.get("gpytorch") is None:
        return False
    from gpytorch.kernels import Kernel as GPyTorchKernel
    from gpytorch.likelihoods import Likelihood as GPyTorchLikelihood
    from gpytorch.means import Mean as GPyTorchMean

    return issubclass(obj, (GPyTorchKernel, GPyTorchMean, GPyTorchLikelihood))


def _validate_component(instance: Any, attribute: Attribute, value: Any) -> None:
    """Validate that an object is a BayBE or a GPyTorch GP component."""
    if isinstance(value, Kernel) or _is_gpytorch_component_class(type(value)):
        return

    raise TypeError(
        f"The object provided for '{attribute.alias}' of "
        f"'{instance.__class__.__name__}' must be a BayBE or a GPyTorch GP component. "
        f"Got: {type(value)}"
    )


class ComponentFactory(Protocol, Generic[_T_co]):
    """A protocol defining the interface expected for GP component factories."""

    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> _T_co:
        """Create a GP component for the given recommendation context."""


@define(frozen=True)
class PlainComponentFactory(ComponentFactory[_T_co], SerialMixin):
    """A trivial factory that returns a fixed pre-defined component upon request."""

    component: _T_co = field(validator=_validate_component)
    """The fixed component to be returned by the factory."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> _T_co:
        return self.component


def to_component_factory(x: Component | ComponentFactory, /) -> ComponentFactory:
    """Wrap a component into a plain component factory (with factory passthrough)."""
    if isinstance(x, Component) or _is_gpytorch_component_class(type(x)):
        return PlainComponentFactory(x)
    return x


# Block serialization of GPyTorch kernel classes since not yet supported
converter.register_unstructure_hook_factory(
    _is_gpytorch_component_class,
    lambda _: block_serialization_hook,
)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
