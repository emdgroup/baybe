"""Base classes for all kernels."""

from abc import ABC

from attrs import define

from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin


@define(frozen=True)
class Kernel(ABC, SerialMixin):
    """Abstract base class for all kernels."""

    def to_gpytorch(self, *args, **kwargs):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels

        kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)

        # Get the fields of the actual class via unstructure_base
        fields_dict = unstructure_base(self)
        # Filter the type out
        fields = {field: fields_dict[field] for field in fields_dict if field != "type"}
        # Update kwargs to contain class-specific parameters
        kwargs.update(fields)

        return kernel_cls(*args, **kwargs)


# Register de-/serialization hooks
converter.register_structure_hook(Kernel, get_base_structure_hook(Kernel))
converter.register_unstructure_hook(Kernel, unstructure_base)
