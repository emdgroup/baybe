"""Adaptive Prior proposed in the paper: 
Guanming Chen, Maximilian Fleck, Thijs Stuyver. Leveraging Hidden-Space Representations Effectively in Bayesian Optimization for Experiment Design through Dimension-Aware Hyperpriors. ChemRxiv. 09 February 2026.
DOI: https://doi.org/10.26434/chemrxiv.10001986/v2"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from typing_extensions import override

from baybe.kernels.basic import MaternKernel
from baybe.kernels.composite import ScaleKernel
from baybe.parameters import TaskParameter
from baybe.priors.basic import GammaPrior
from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory

if TYPE_CHECKING:
    from torch import Tensor

    from baybe.kernels.base import Kernel
    from baybe.searchspace.core import SearchSpace

import math

@define
class CHENKernelFactory(KernelFactory):
    """ Surrogate model with an adaptive hyperprior proposed in the paper: 
        Guanming Chen, Maximilian Fleck, Thijs Stuyver. Leveraging Hidden-Space Representations Effectively in Bayesian Optimization for Experiment Design through Dimension-Aware Hyperpriors. ChemRxiv. 09 February 2026.
        DOI: https://doi.org/10.26434/chemrxiv.10001986/v2
    """

    @override
    def __call__(
        self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor
    ) -> Kernel:
        effective_dims = train_x.shape[-1] - len(
            [p for p in searchspace.parameters if isinstance(p, TaskParameter)]
        )

        x = math.sqrt(effective_dims)
        l_mean = 0.4 * x + 4.0
        
        lengthscale_prior = GammaPrior(2.0*l_mean, 2.0)
        lengthscale_initial_value = l_mean
        outputscale_prior = GammaPrior(1.0*l_mean, 1.0) 
        outputscale_initial_value = l_mean

        return ScaleKernel(
            MaternKernel(
                nu=2.5,
                lengthscale_prior=lengthscale_prior,
                lengthscale_initial_value=lengthscale_initial_value,
            ),
            outputscale_prior=outputscale_prior,
            outputscale_initial_value=outputscale_initial_value,
        )

# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
