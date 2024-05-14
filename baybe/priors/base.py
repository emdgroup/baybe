"""Base class for all priors."""

from abc import ABC

from attrs import define

from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.utils.basic import filter_attributes


@define(frozen=True)
class Prior(ABC, SerialMixin):
    """Abstract base class for all priors."""

    def to_gpytorch(self, *args, **kwargs):
        """Create the gpytorch representation of the prior."""
        import gpytorch.priors

        prior_cls = getattr(gpytorch.priors, self.__class__.__name__)
        fields_dict = filter_attributes(object=self, callable_=prior_cls.__init__)

        # Update kwargs to contain class-specific attributes
        kwargs.update(fields_dict)

        return prior_cls(*args, **kwargs)


# Register de-/serialization hooks
converter.register_structure_hook(Prior, get_base_structure_hook(Prior))
converter.register_unstructure_hook(Prior, unstructure_base)
