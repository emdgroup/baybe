"""Composite kernels (that is, kernels composed of other kernels)."""

from __future__ import annotations

import gc
from functools import reduce
from operator import add, mul
from typing import TYPE_CHECKING, Any

from attrs import Attribute, define, evolve, field
from attrs.converters import optional as optional_c
from attrs.validators import and_, deep_iterable, gt, instance_of, min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.kernels.base import CompositeKernel, Kernel
from baybe.priors.base import Prior
from baybe.settings import active_settings
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gpytorch.kernels import Kernel as GPyTorchKernel

    from baybe.searchspace.core import SearchSpace


def _is_vanishing(kernel: Kernel, /) -> bool:
    """Check whether a kernel vanishes under kernel arithmetic."""
    from baybe.kernels.basic import VanishingKernel

    return isinstance(kernel, VanishingKernel)


def _validate_not_vanishing(instance: Any, attribute: Attribute, value: Any) -> None:  # noqa: DOC101, DOC103
    """Reject a vanishing kernel in a single kernel slot.

    Raises:
        ValueError: If the kernel vanishes under kernel arithmetic.
    """
    if _is_vanishing(value):
        raise ValueError(f"'{type(instance).__name__}' cannot wrap a vanishing kernel.")


def _validate_not_all_vanishing(  # noqa: DOC101, DOC103
    instance: Any, attribute: Attribute, value: Any
) -> None:
    """Require at least one non-vanishing kernel in a composite.

    Raises:
        ValueError: If all kernels vanish under kernel arithmetic.
    """
    if all(_is_vanishing(k) for k in value):
        raise ValueError(
            f"'{type(instance).__name__}' must contain at least one kernel that does "
            f"not vanish under kernel arithmetic."
        )


def _to_gpytorch_factors(
    base_kernels: Iterable[Kernel], searchspace: SearchSpace
) -> list[GPyTorchKernel]:
    """Convert base kernels to GPyTorch, dropping the vanishing ones.

    Args:
        base_kernels: The base kernels of a composite kernel.
        searchspace: The search space the kernels operate on.

    Returns:
        The GPyTorch kernels of the non-vanishing base kernels.
    """
    factors = [k.to_gpytorch(searchspace) for k in base_kernels if not _is_vanishing(k)]
    # Guaranteed by the composite kernel validators
    assert factors
    return factors


def _reduce_base_kernels(
    kernel: AdditiveKernel | ProductKernel, name: str, searchspace: SearchSpace, /
) -> Kernel | None:
    """Remove a parameter from all base kernels of an additive or product kernel.

    Vanishing base kernels are dropped along the way.

    Args:
        kernel: The additive or product kernel whose base kernels are to be reduced.
        name: The name of the parameter to remove.
        searchspace: The search space the kernel operates on.

    Returns:
        The reduced kernel of the same type, the sole remaining base kernel, or
        ``None`` if no base kernel remains.
    """
    remaining = tuple(
        reduced
        for k in kernel.base_kernels
        if (reduced := k._without_parameter(name, searchspace)) is not None
        and not _is_vanishing(reduced)
    )
    if not remaining:
        return None
    if len(remaining) == 1:
        return remaining[0]
    return evolve(kernel, base_kernels=remaining)


@define(frozen=True)
class ScaleKernel(CompositeKernel):
    """A kernel for decorating existing kernels with an outputscale."""

    base_kernel: Kernel = field(
        validator=[instance_of(Kernel), _validate_not_vanishing]
    )
    """The base kernel that is being decorated."""

    outputscale_prior: Prior | None = field(
        default=None, validator=optional_v(instance_of(Prior))
    )
    """An optional prior on the output scale."""

    outputscale_initial_value: float | None = field(
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0.0)]),
    )
    """An optional initial value for the output scale."""

    outputscale_trainable: bool = field(default=True, validator=instance_of(bool))
    """Boolean flag indicating whether the output scale is trainable.

    If ``False``, the output scale is frozen at its initial value and excluded from
    optimization."""

    @override
    def _without_parameter(
        self, name: str, searchspace: SearchSpace, /
    ) -> Kernel | None:
        stripped = self.base_kernel._without_parameter(name, searchspace)
        return None if stripped is None else evolve(self, base_kernel=stripped)

    @override
    def _scope_to_parameter(self, name: str | None, /) -> Kernel:
        return evolve(self, base_kernel=self.base_kernel._scope_to_parameter(name))

    @override
    def to_gpytorch(self, *args, **kwargs):
        import torch

        gpytorch_kernel = super().to_gpytorch(*args, **kwargs)
        if (initial_value := self.outputscale_initial_value) is not None:
            gpytorch_kernel.outputscale = torch.tensor(
                initial_value, dtype=active_settings.DTypeFloatTorch
            )
        if not self.outputscale_trainable:
            gpytorch_kernel.raw_outputscale.requires_grad_(False)
        return gpytorch_kernel


@define(frozen=True)
class AdditiveKernel(CompositeKernel):
    """A kernel representing the sum of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel),
            iterable_validator=and_(min_len(2), _validate_not_all_vanishing),
        ),
    )
    """The individual kernels to be summed."""

    @override
    def _without_parameter(
        self, name: str, searchspace: SearchSpace, /
    ) -> Kernel | None:
        return _reduce_base_kernels(self, name, searchspace)

    @override
    def _scope_to_parameter(self, name: str | None, /) -> Kernel:
        return evolve(
            self,
            base_kernels=tuple(k._scope_to_parameter(name) for k in self.base_kernels),
        )

    @override
    def to_gpytorch(self, *args, **kwargs):
        return reduce(add, _to_gpytorch_factors(self.base_kernels, *args, **kwargs))


@define(frozen=True)
class ProductKernel(CompositeKernel):
    """A kernel representing the product of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel),
            iterable_validator=and_(min_len(2), _validate_not_all_vanishing),
        ),
    )
    """The individual kernels to be multiplied."""

    @override
    def _without_parameter(
        self, name: str, searchspace: SearchSpace, /
    ) -> Kernel | None:
        return _reduce_base_kernels(self, name, searchspace)

    @override
    def _scope_to_parameter(self, name: str | None, /) -> Kernel:
        return evolve(
            self,
            base_kernels=tuple(k._scope_to_parameter(name) for k in self.base_kernels),
        )

    @override
    def to_gpytorch(self, *args, **kwargs):
        return reduce(mul, _to_gpytorch_factors(self.base_kernels, *args, **kwargs))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
