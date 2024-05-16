"""Base classes for all kernels."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Optional

from attrs import define

from baybe.priors.base import Prior
from baybe.serialization.core import (
    converter,
    get_base_structure_hook,
    unstructure_base,
)
from baybe.serialization.mixin import SerialMixin
from baybe.utils.basic import filter_attributes, get_baseclasses

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
        ard_num_dims: Optional[int] = None,
        batch_shape: Optional[torch.Size] = None,
        active_dims: Optional[tuple[int, ...]] = None,
    ):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels

        # Fetch the necessary gpytorch constructor parameters of the kernel.
        # NOTE: In gpytorch, some attributes (like the kernel lengthscale) are handled
        # via the `gpytorch.kernels.Kernel` base class. Hence, it is not sufficient to
        # just check the fields of the actual class, but also those of the base class.
        kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)
        base_classes = get_baseclasses(kernel_cls, abstract=True)
        fields_dict = {}
        for cls in [kernel_cls, *base_classes]:
            fields_dict.update(filter_attributes(object=self, callable_=cls.__init__))

        # Convert specified priors to gpytorch, if provided
        prior_dict = {
            key: value.to_gpytorch()
            for key, value in fields_dict.items()
            if isinstance(value, Prior)
        }

        # Convert specified inner kernels to gpytorch, if provided
        kernel_dict = {
            key: value.to_gpytorch(
                ard_num_dims=ard_num_dims,
                batch_shape=batch_shape,
                active_dims=active_dims,
            )
            for key, value in fields_dict.items()
            if isinstance(value, Kernel)
        }

        # Create the kernel with all its inner gpytorch objects
        fields_dict.update(kernel_dict)
        fields_dict.update(prior_dict)
        gpytorch_kernel = kernel_cls(**fields_dict)

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


# Register de-/serialization hooks
converter.register_structure_hook(Kernel, get_base_structure_hook(Kernel))
converter.register_unstructure_hook(Kernel, unstructure_base)
