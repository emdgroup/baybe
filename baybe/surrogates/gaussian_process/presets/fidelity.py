"""Kernels for Gaussian process fidelity surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.kernels.basic import IndexKernel
from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace


@define
class IndependentFidelityKernelFactory(KernelFactory):
    """Rank 0 index kernel treating fidelities as independent."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return IndexKernel(
            num_tasks=searchspace.n_fidelities,
            active_dims=searchspace.fidelity_idx,
            rank=0,
        )


@define
class IndexFidelityKernelFactory(KernelFactory):
    """Full rank index kernel modelling dependent fidelities."""

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        return IndexKernel(
            num_tasks=searchspace.n_fidelities,
            active_dims=searchspace.fidelity_idx,
            rank=searchspace.n_fidelities,
        )


DefaultFidelityKernelFactory = IndexFidelityKernelFactory

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
