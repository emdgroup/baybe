"""Kernel factories for the Gaussian process surrogate."""

from baybe.kernels.base import Kernel
from baybe.surrogates.gaussian_process.components import (
    ComponentFactory,
    PlainComponentFactory,
)

KernelFactory = ComponentFactory[Kernel]
PlainKernelFactory = PlainComponentFactory[Kernel]
