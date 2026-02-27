from __future__ import annotations

import math

from abc import ABC, abstractmethod
from attrs import define, field
from attrs.validators import instance_of
from contextlib import nullcontext
from copy import deepcopy
import numpy as np

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.exceptions import UnsupportedError
from botorch.exceptions.warnings import legacy_ei_numerics_warning
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.model import Model
from botorch.utils.constants import get_constants_like
from botorch.utils.probability import MVNXPB
from botorch.utils.probability.utils import (
    compute_log_prob_feas_from_bounds,
    log_ndtr as log_Phi,
    log_phi,
    ndtr as Phi,
    phi,
)
from botorch.utils.safe_math import log1mexp, logmeanexp
from botorch.utils.transforms import (
    t_batch_mode_transform,
)
from gpytorch.likelihoods.gaussian_likelihood import FixedNoiseGaussianLikelihood
from torch import Tensor
from torch.nn.functional import pad

from itertools import product as iter_product

@define
class CostAwareAcquisitionFunction(AcquisitionFunction, ABC):
    """Abstract base class for acquisition functions with cost-aware wrapping over a base acquisition function"""

    # Jordan MHS: check the type here!
    # Jordan MHS: alias base_acqf for user-defined ICWAF.
    base_acqf: AcquisitionFunction = field(validator=instance_of(AcquisitionFunction))

    fidelities: dict[int, tuple[float, ...]]

    costs: dict[int, tuple[float, ...]]

    # @abstractmethod
    # def cost_model(self):
    #     ...

    @abstractmethod
    def forward(self, X):
        ...

    @abstractmethod
    def __getattr__(self, name):
        ...

@define
class InverseCostWeightedAcquisitionFunction(CostAwareAcquisitionFunction):
    """Cost aware acquisition function which divides an acquisition value by the corresponding cost on forward"""
    
    # def cost_model(self):
    #     return self._cost_model
    
    def forward(self, X):
        return self.base_model.forward(X) / self.cost_model(X)
    
    def __getattr__(self, name):
        return getattr(self.base_acqf, name)