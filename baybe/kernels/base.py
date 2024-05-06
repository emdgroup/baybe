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
from baybe.utils.basic import filter_attributes, get_parent_classes


@define(frozen=True)
class Kernel(ABC, SerialMixin):
    """Abstract base class for all kernels."""

    def to_gpytorch(self, *args, **kwargs):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels
        from torch import Tensor

        # Due to the weird way that gpytorch does the inheritance for kernels, it is not
        # sufficient to just check for the fields of the actual class, but also for the
        # ones of the basic "Kernel" class and combine them.
        kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)
        parent_classes = get_parent_classes(kernel_cls)
        fields_dict = {}
        for parent_class in parent_classes:
            fields_dict.update(
                filter_attributes(object=self, callable_=parent_class.__init__)
            )

        # Since the args and kwargs passed in this function are kernel-specific, we do
        # not need to pass them when converting the priors but when converting the
        # kernels.
        prior_dict = {
            key: fields_dict[key].to_gpytorch()
            for key in fields_dict
            if isinstance(fields_dict[key], Prior)
        }

        # NOTE: Our coretest actually behave the same if we do not pass *args and
        # **kwargs here. This suggests that it might be the case that gyptorch itself
        # already passes theses. It might be worthwhile investigating this in more
        # detail later.
        kernel_dict = {
            key: fields_dict[key].to_gpytorch(*args, **kwargs)
            for key in fields_dict
            if isinstance(fields_dict[key], Kernel)
        }

        fields_dict.update(kernel_dict)
        fields_dict.update(prior_dict)
        kwargs.update(fields_dict)

        gpytorch_kernel = kernel_cls(*args, **kwargs)

        # Since the initial values for the priors can only be set after initialization
        # and not using args or kwargs and since the naming is inconsistent, we need to
        # do some hacky checking and setting here
        if hasattr(self, "lengthscale_prior_initial_value"):
            initial_value = self.lengthscale_prior_initial_value
            if initial_value is not None:
                gpytorch_kernel.lengthscale = Tensor([initial_value])
        if hasattr(self, "outputscale_prior_initial_value"):
            initial_value = self.outputscale_prior_initial_value
            if initial_value is not None:
                gpytorch_kernel.outputscale = Tensor([initial_value])
        return gpytorch_kernel


# Register de-/serialization hooks
converter.register_structure_hook(Kernel, get_base_structure_hook(Kernel))
converter.register_unstructure_hook(Kernel, unstructure_base)
