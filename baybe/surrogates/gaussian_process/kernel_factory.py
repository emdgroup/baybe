"""Kernel factories for the Gaussian process surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from baybe.kernels.base import Kernel
from baybe.surrogates.gaussian_process.components import (
    ComponentFactory,
    PlainComponentFactory,
)

if TYPE_CHECKING:
    from gpytorch.kernels import Kernel as GPyTorchKernel

    KernelFactory = ComponentFactory[Kernel | GPyTorchKernel]
    PlainKernelFactory = PlainComponentFactory[Kernel | GPyTorchKernel]
else:
    # At runtime, we use only the BayBE type for serialization compatibility
    KernelFactory = ComponentFactory[Kernel]
    PlainKernelFactory = PlainComponentFactory[Kernel]
