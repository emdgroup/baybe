"""Base classes for all kernels."""

from __future__ import annotations

import gc
from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from attrs import define

from baybe.exceptions import UnmatchedAttributeError
from baybe.priors.base import Prior
from baybe.serialization.mixin import SerialMixin
from baybe.settings import active_settings
from baybe.utils.basic import get_baseclasses, match_attributes

if TYPE_CHECKING:
    import torch

    from baybe.surrogates.gaussian_process.components.kernel import PlainKernelFactory


@define(frozen=True)
class Kernel(ABC, SerialMixin):
    """Abstract base class for all kernels."""

    def __add__(self, other: Any) -> Kernel:
        """Create a sum kernel from two kernels.

        Flattens nested sums so that ``(a + b) + c`` yields
        ``SumKernel([a, b, c])`` instead of ``SumKernel([SumKernel([a, b]), c])``.
        """
        if isinstance(other, Kernel):
            from baybe.kernels.composite import SumKernel

            left = self.base_kernels if isinstance(self, SumKernel) else (self,)
            right = other.base_kernels if isinstance(other, SumKernel) else (other,)
            return SumKernel([*left, *right])
        return NotImplemented

    def __radd__(self, other: Any) -> Kernel:
        """Support right-hand addition for kernel objects."""
        return self.__add__(other)

    def __mul__(self, other: Any) -> Kernel:
        """Create a product kernel or scale kernel.

        When multiplied with another kernel, a product kernel is created. Nested
        products are flattened so that ``(a * b) * c`` yields
        ``ProductKernel([a, b, c])``. When multiplied with a numeric constant, a scale
        kernel with a fixed (non-trainable) output scale is created.
        """
        if isinstance(other, Kernel):
            from baybe.kernels.composite import ProductKernel

            left = self.base_kernels if isinstance(self, ProductKernel) else (self,)
            right = other.base_kernels if isinstance(other, ProductKernel) else (other,)
            return ProductKernel([*left, *right])
        if isinstance(other, (int, float)):
            from baybe.kernels.composite import ScaleKernel

            return ScaleKernel(
                base_kernel=self,
                outputscale_initial_value=float(other),
                outputscale_trainable=False,
            )
        return NotImplemented

    def __rmul__(self, other: Any) -> Kernel:
        """Support right-hand multiplication, enabling ``constant * kernel``."""
        return self.__mul__(other)

    def to_factory(self) -> PlainKernelFactory:
        """Wrap the kernel in a :class:`baybe.surrogates.gaussian_process.components.PlainKernelFactory`."""  # noqa: E501
        from baybe.surrogates.gaussian_process.components.kernel import (
            PlainKernelFactory,
        )

        return PlainKernelFactory(self)

    def to_gpytorch(
        self,
        *,
        ard_num_dims: int | None = None,
        batch_shape: torch.Size | None = None,
        active_dims: Sequence[int] | None = None,
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
        try:
            kernel_cls = getattr(gpytorch.kernels, self.__class__.__name__)
        except AttributeError:
            import botorch.models.kernels.positive_index

            kernel_cls = getattr(
                botorch.models.kernels.positive_index, self.__class__.__name__
            )
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
        # Exceptions: initial values and trainability flags are not used during
        # construction but are set on the created object after construction.
        missing = set(unmatched) - set(kernel_attrs)
        if leftover := {
            m
            for m in missing
            if not m.endswith("_initial_value") and not m.endswith("_trainable")
        }:
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

            # We can ignore mypy here and simply assume that the corresponding BayBE
            # kernel class has the necessary lengthscale attribute defined. This is
            # safer than using a `hasattr` check in the above if-condition since for
            # the latter the code would silently fail when forgetting to add the
            # attribute to a new kernel class / misspelling it.
            if (initial_value := self.lengthscale_initial_value) is not None:  # type: ignore[attr-defined]
                gpytorch_kernel.lengthscale = torch.tensor(
                    initial_value, dtype=active_settings.DTypeFloatTorch
                )

        return gpytorch_kernel


@define(frozen=True)
class BasicKernel(Kernel, ABC):
    """Abstract base class for all basic kernels."""


@define(frozen=True)
class CompositeKernel(Kernel, ABC):
    """Abstract base class for all composite kernels."""


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
