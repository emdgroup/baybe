"""Base classes for all kernels."""

from __future__ import annotations

import gc
from abc import ABC
from typing import TYPE_CHECKING, Any

from attrs import define

from baybe.exceptions import UnmatchedAttributeError
from baybe.priors.base import Prior
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.utils.basic import get_baseclasses, match_attributes

if TYPE_CHECKING:
    import torch

    from baybe.surrogates.gaussian_process.kernel_factory import PlainKernelFactory


@define(frozen=True)
class Kernel(ABC, SerialMixin):
    """Abstract base class for all kernels."""

    def to_factory(self) -> PlainKernelFactory:
        """Wrap the kernel in a :class:`baybe.surrogates.gaussian_process.kernel_factory.PlainKernelFactory`."""  # noqa: E501
        from baybe.surrogates.gaussian_process.kernel_factory import PlainKernelFactory

        return PlainKernelFactory(self)

    def to_gpytorch(
        self,
        *,
        ard_num_dims: int | None = None,
        batch_shape: torch.Size | None = None,
        active_dims: tuple[int, ...] | None = None,
    ):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels

        # Extract keywords with non-default values. This is required since gpytorch
        # makes use of kwargs, i.e. differentiates if certain keywords are explicitly
        # passed or not. For instance, `ard_num_dims = kwargs.get("ard_num_dims", 1)`
        # fails if we explicitly pass `ard_num_dims=None`.
        kw: dict[str, Any] = dict(
            ard_num_dims=ard_num_dims, batch_shape=batch_shape, active_dims=active_dims
        )
        kw = {k: v for k, v in kw.items() if v is not None}

        # Get corresponding gpytorch kernel class and its base classes
        kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)
        base_classes = get_baseclasses(kernel_cls, abstract=True)

        # Fetch the necessary gpytorch constructor parameters of the kernel.
        # NOTE: In gpytorch, some attributes (like the kernel lengthscale) are handled
        #   via the `gpytorch.kernels.Kernel` base class. Hence, it is not sufficient to
        #   just check the fields of the actual class, but also those of its base
        #   classes.
        kernel_attrs: dict[str, Any] = {}
        unmatched_attrs: dict[str, Any] = {}
        for cls in [kernel_cls, *base_classes]:
            matched, unmatched = match_attributes(self, cls.__init__, strict=False)
            kernel_attrs.update(matched)
            unmatched_attrs.update(unmatched)

        # Sanity check: all attributes of the BayBE kernel need a corresponding match
        # in the gpytorch kernel (otherwise, the BayBE kernel class is misconfigured).
        # Exception: initial values are not used during construction but are set
        # on the created object (see code at the end of the method).
        missing = set(unmatched) - set(kernel_attrs)
        if leftover := {m for m in missing if not m.endswith("_initial_value")}:
            raise UnmatchedAttributeError(leftover)

        # Convert specified priors to gpytorch, if provided
        prior_dict = {
            key: value.to_gpytorch()
            for key, value in kernel_attrs.items()
            if isinstance(value, Prior)
        }

        # Convert specified inner kernels to gpytorch, if provided
        kernel_dict = {
            key: value.to_gpytorch(**kw)
            for key, value in kernel_attrs.items()
            if isinstance(value, Kernel)
        }

        # Create the kernel with all its inner gpytorch objects
        kernel_attrs.update(kernel_dict)
        kernel_attrs.update(prior_dict)
        gpytorch_kernel = kernel_cls(**kernel_attrs, **kw)

        # If the kernel has a lengthscale, set its initial value
        if kernel_cls.has_lengthscale:
            import torch

            from baybe.utils.torch import DTypeFloatTorch

            # We can ignore mypy here and simply assume that the corresponding BayBE
            # kernel class has the necessary lengthscale attribute defined. This is
            # safer than using a `hasattr` check in the above if-condition since for
            # the latter the code would silently fail when forgetting to add the
            # attribute to a new kernel class / misspelling it.
            if (initial_value := self.lengthscale_initial_value) is not None:  # type: ignore[attr-defined]
                gpytorch_kernel.lengthscale = torch.tensor(
                    initial_value, dtype=DTypeFloatTorch
                )

        return gpytorch_kernel


@define(frozen=True)
class BasicKernel(Kernel, ABC):
    """Abstract base class for all basic kernels."""


@define(frozen=True)
class CompositeKernel(Kernel, ABC):
    """Abstract base class for all composite kernels."""


# Register de-/serialization hooks
converter.register_structure_hook(Kernel, get_base_structure_hook(Kernel))
converter.register_unstructure_hook(Kernel, unstructure_base)

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
