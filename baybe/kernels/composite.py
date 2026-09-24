"""Composite kernels (that is, kernels composed of other kernels)."""

from __future__ import annotations

import gc
from functools import reduce
from operator import add, mul
from typing import TYPE_CHECKING

from attrs import define, evolve, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_iterable, gt, instance_of, min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.kernels.base import CompositeKernel, Kernel
from baybe.priors.base import Prior
from baybe.settings import active_settings
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace


def _reduce_base_kernels(
    kernel: AdditiveKernel | ProductKernel, name: str, searchspace: SearchSpace, /
) -> Kernel | None:
    """Remove a parameter from all base kernels of a composite kernel.

    Args:
        kernel: The composite kernel whose base kernels are to be reduced.
        name: The name of the parameter to remove.
        searchspace: The search space the kernel operates on.

    Returns:
        The reduced composite kernel, the sole remaining base kernel, or ``None``
        if no base kernel remains.
    """
    remaining = tuple(
        reduced
        for k in kernel.base_kernels
        if (reduced := k._without_parameter(name, searchspace)) is not None
    )
    if not remaining:
        return None
    if len(remaining) == 1:
        return remaining[0]
    return evolve(kernel, base_kernels=remaining)


@define(frozen=True)
class ScaleKernel(CompositeKernel):
    """A kernel for decorating existing kernels with an outputscale."""

    base_kernel: Kernel = field(validator=instance_of(Kernel))
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
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
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
        return reduce(add, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


@define(frozen=True)
class ProductKernel(CompositeKernel):
    """A kernel representing the product of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
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
        return reduce(mul, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
