"""Base classes for all kernels."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from itertools import chain
from typing import TYPE_CHECKING, Any

from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_iterable, instance_of
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.exceptions import UnmatchedAttributeError
from baybe.priors.base import Prior
from baybe.searchspace.core import SearchSpace
from baybe.serialization.mixin import SerialMixin
from baybe.settings import active_settings
from baybe.utils.basic import classproperty, get_baseclasses, match_attributes

if TYPE_CHECKING:
    import torch

    from baybe.surrogates.gaussian_process.components.kernel import PlainKernelFactory


@define(frozen=True)
class Kernel(ABC, SerialMixin):
    """Abstract base class for all kernels."""

    @classproperty
    def _whitelisted_attributes(cls) -> frozenset[str]:
        """Attribute names to exclude from gpytorch matching."""
        return frozenset()

    def to_factory(self) -> PlainKernelFactory:
        """Wrap the kernel in a :class:`baybe.surrogates.gaussian_process.components.PlainKernelFactory`."""  # noqa: E501
        from baybe.surrogates.gaussian_process.components.kernel import (
            PlainKernelFactory,
        )

        return PlainKernelFactory(self)

    @abstractmethod
    def _get_dimensions(self, searchspace: SearchSpace) -> tuple[tuple[int, ...], int]:
        """Get the active dimensions and the number of ARD dimensions."""

    def to_gpytorch(
        self,
        searchspace: SearchSpace,
        *,
        batch_shape: torch.Size | None = None,
    ):
        """Create the gpytorch representation of the kernel."""
        import gpytorch.kernels

        active_dims, ard_num_dims = self._get_dimensions(searchspace)

        # Extract keywords with non-default values. This is required since gpytorch
        # makes use of kwargs, i.e. differentiates if certain keywords are explicitly
        # passed or not. For instance, `ard_num_dims = kwargs.get("ard_num_dims", 1)`
        # fails if we explicitly pass `ard_num_dims=None`.
        kw: dict[str, Any] = dict(batch_shape=batch_shape, active_dims=active_dims)
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
        # Exception: initial values are not used during construction but are set
        # on the created object (see code at the end of the method).
        missing = set(unmatched) - set(kernel_attrs) - self._whitelisted_attributes
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
            key: value.to_gpytorch(searchspace, **kw)
            for key, value in kernel_attrs.items()
            if isinstance(value, Kernel)
        }

        # Create the kernel with all its inner gpytorch objects
        kernel_attrs.update(kernel_dict)
        kernel_attrs.update(prior_dict)
        gpytorch_kernel = kernel_cls(**kernel_attrs, ard_num_dims=ard_num_dims, **kw)

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

    parameter_names: tuple[str, ...] | None = field(
        default=None,
        converter=optional_c(tuple),
        validator=optional_v(deep_iterable(member_validator=instance_of(str))),
        kw_only=True,
    )
    """An optional set of names specifiying the parameters the kernel should act on."""

    @override
    @classproperty
    def _whitelisted_attributes(cls) -> frozenset[str]:
        return frozenset({"parameter_names"})

    def _get_dimensions(self, searchspace):
        if self.parameter_names is None:
            active_dims = None
        else:
            active_dims = list(
                chain(
                    *[
                        searchspace.get_comp_rep_parameter_indices(name)
                        for name in self.parameter_names
                    ]
                )
            )

        # We use automatic relevance determination for all kernels
        ard_num_dims = (
            len(active_dims)
            if active_dims is not None
            else len(searchspace.comp_rep_columns)
        )
        return active_dims, ard_num_dims


@define(frozen=True)
class CompositeKernel(Kernel, ABC):
    """Abstract base class for all composite kernels."""

    def _get_dimensions(self, searchspace):
        return None, None


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
