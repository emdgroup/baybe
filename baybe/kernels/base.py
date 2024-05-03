"""Base classes for all kernels."""

from abc import ABC

from attrs import define

from baybe.kernels.priors.base import Prior
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.utils.basic import filter_attributes


@define(frozen=True)
class Kernel(ABC, SerialMixin):
    """Abstract base class for all kernels."""

    def to_gpytorch(self, *args, **kwargs):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels

        kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)
        fields_dict = filter_attributes(object=self, callable_=kernel_cls.__init__)

        prior_dict = {
            key: fields_dict[key].to_gpytorch()
            for key in fields_dict
            if isinstance(fields_dict[key], Prior)
        }

        kernel_dict = {
            key: fields_dict[key].to_gpytorch(*args, **kwargs)
            for key in fields_dict
            if isinstance(fields_dict[key], Kernel)
        }

        fields_dict.update(kernel_dict)
        fields_dict.update(prior_dict)
        kwargs.update(fields_dict)

        return kernel_cls(*args, **kwargs)


# Register de-/serialization hooks
converter.register_structure_hook(Kernel, get_base_structure_hook(Kernel))
converter.register_unstructure_hook(Kernel, unstructure_base)
