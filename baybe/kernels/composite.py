"""Composite kernels (that is, kernels composed of other kernels)."""

import gc
from functools import reduce
from operator import add, mul

from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_iterable, gt, instance_of, min_len
from attrs.validators import optional as optional_v

from baybe.kernels.base import CompositeKernel, Kernel
from baybe.priors.base import Prior
from baybe.utils.validation import finite_float


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

    def to_gpytorch(self, *args, **kwargs):  # noqa: D102
        # See base class.
        import torch

        from baybe.utils.torch import DTypeFloatTorch

        gpytorch_kernel = super().to_gpytorch(*args, **kwargs)
        if (initial_value := self.outputscale_initial_value) is not None:
            gpytorch_kernel.outputscale = torch.tensor(
                initial_value, dtype=DTypeFloatTorch
            )
        return gpytorch_kernel


@define(frozen=True)
class AdditiveKernel(CompositeKernel):
    """A kernel representing the sum of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be summed."""

    def to_gpytorch(self, *args, **kwargs):  # noqa: D102
        # See base class.

        return reduce(add, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


@define(frozen=True)
class ProductKernel(CompositeKernel):
    """A kernel representing the product of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be multiplied."""

    def to_gpytorch(self, *args, **kwargs):  # noqa: D102
        # See base class.

        return reduce(mul, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
