"""Base class for all priors."""

import gc
from abc import ABC

from attrs import define

from baybe.serialization.mixin import SerialMixin
from baybe.settings import active_settings
from baybe.utils.basic import match_attributes


@define(frozen=True)
class Prior(ABC, SerialMixin):
    """Abstract base class for all priors."""

    def to_gpytorch(self, *args, **kwargs):
        """Create the gpytorch representation of the prior."""
        import gpytorch.priors
        import torch

        # TODO: This is only a temporary workaround. A proper solution requires
        #   modifying the torch import procedure using the built-in tools of importlib
        #   so that the dtype is set whenever torch is lazily loaded.
        torch.set_default_dtype(active_settings.DTypeFloatTorch)

        prior_cls = getattr(gpytorch.priors, self.__class__.__name__)
        fields_dict = match_attributes(self, prior_cls.__init__)[0]

        # Update kwargs to contain class-specific attributes
        kwargs.update(fields_dict)

        return prior_cls(*args, **kwargs)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
