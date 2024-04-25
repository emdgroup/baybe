"""Base classes for all kernels."""

from abc import ABC
from typing import Optional

from attrs import define, field

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

    lengthscale_prior: Optional[Prior] = field(default=None, kw_only=True)
    """An optional prior on the kernel lengthscale."""

    def to_gpytorch(self, *args, **kwargs):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels

        kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)
        fields_dict = filter_attributes(object=self, callable_=kernel_cls.__init__)

        # If a lengthscale_prior was chosen, we manually add it to the dictionary since
        # it is not discovered by the code above
        if self.lengthscale_prior is not None:
            fields_dict["lengthscale_prior"] = self.lengthscale_prior

        # If there are any priors in the attributes, we transform them using our
        # to_gpytorch function. Passing of args and kwargs is not necessary here.
        prior_dict = {
            key: fields_dict[key].to_gpytorch()
            for key in fields_dict
            if isinstance(fields_dict[key], Prior)
        }

        # If there are any kernels in the attributes, we transform them using our
        # to_gpytorch function. It is necessary to pass the args and kwargs here.
        kernel_dict = {
            key: fields_dict[key].to_gpytorch(*args, **kwargs)
            for key in fields_dict
            if isinstance(fields_dict[key], Kernel)
        }

        # Update fields_dict
        fields_dict.update(kernel_dict)
        fields_dict.update(prior_dict)
        # Update kwargs to contain class-specific attributes
        kwargs.update(fields_dict)

        return kernel_cls(*args, **kwargs)


# Register de-/serialization hooks
converter.register_structure_hook(Kernel, get_base_structure_hook(Kernel))
converter.register_unstructure_hook(Kernel, unstructure_base)
