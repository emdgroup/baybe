from __future__ import annotations

import math

from abc import ABC, abstractmethod
from contextlib import nullcontext
from copy import deepcopy
import numpy as np

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
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

# the following two numbers are needed for _log_ei_helper
_neg_inv_sqrt2 = -(2**-0.5)
_log_sqrt_pi_div_2 = math.log(math.pi / 2) / 2

class MultiFidelityUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Two-stage Multi Fidelity Upper Confidence Bound (UCB), based on Kandasamy (2016).

    Analytic upper confidence bound that comprises of the posterior mean plus two
    additional terms: the posterior standard deviation weighted by a trade-off
    parameter, `beta`; and a fidelity-based tolerance parameter (Jordan MHS: BLAH). 
    Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x, m) = mu(x, m) + sqrt(beta) * sigma(x, m) + zeta(m)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively, and `zeta(m)` is the maximum absolute discrepancy between
    fidelity `m` and the highest fidelity `M`.

    `MFUCB(x) = softmin_m(UCB(x, m))` where `softmin_m(v_1, ..., v_m) = (sum_{i=1}^m v_i exp(v_i/T))/(sum_{i=1}^m exp(v_i)`.
    """

    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        fidelities: dict[int, tuple[float, ...]],
        costs: dict[int, tuple[float, ...]],                                   # Jordan MHS TODO, let this be a callable
        zetas: dict[int, tuple[float, ...]],
        softmin_temperature: float = 1e-2,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.
        
        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            fidelities: Computational representation of fidelity values.
            costs: Cost of querying each . Has structure {fidelity_col_idx, costs}.
            zetas: maximum absolute discrepancy between each fidelity and the higest 
                fidelity output.
            softmin_temperature: smoothing parameter for gradient based optimisation
                of design.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("softmin_temperature", torch.as_tensor(softmin_temperature))
        
        fidelity_indices = torch.tensor(list(fidelities.keys()), dtype=torch.long)
        
        # Cartesian product of fidelity values over the indices
        
        fidelity_combos_product = list(iter_product(*fidelities.values()))
        fidelity_combos_tensor = torch.tensor(fidelity_combos_product, dtype=torch.double)
        
        self.register_buffer("fidelity_columns", fidelity_indices)
        self.register_buffer("fidelities_comb", fidelity_combos_tensor)
        
        # Jordan MHS: use a fidelity parameter-based heuristic for this. 
        if zetas is None: 
            zetas = {fid_col: torch.tensor((0.0) * len(fid_vals)) for fid_col, fid_vals in fidelities.items()}
        
        zetas_product = list(iter_product(*zetas.values()))
        zetas_tensor = torch.tensor(zetas_product, dtype=torch.double)
        
        self.register_buffer("zetas_comb", torch.as_tensor(zetas_tensor))
        
        costs_product = list(iter_product(*costs.values()))
        costs_tensor = torch.tensor(costs_product, dtype=torch.double)
        
        self.register_buffer("costs_comb", torch.as_tensor(costs_tensor))
        
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""First optimisation stage: choose optimal design design to query.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design/fidelity points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design and fidelity points `X`.
        """

        batch_size, q, d = X.shape
        
        n_comb, k = self.fidelities_comb.shape

        X_extended = X.clone().unsqueeze(1).repeat(1, n_comb, 1, 1)
        X_extended[..., :, self.fidelity_columns] = self.fidelities_comb.view(1, n_comb, 1, k)
        
        zetas_comb_sum = self.zetas_comb.sum(dim=-1)
        zetas_comb_sum = zetas_comb_sum.view(1, n_comb, 1, 1)
        zetas_extended = zetas_comb_sum.expand(batch_size, n_comb, q, 1)
        
        X_eval = X_extended.reshape(batch_size * n_comb, q, d)
        means, sigmas = self._mean_and_sigma(X_eval)
        
        means = means.view(batch_size, n_comb, q, 1)
        sigmas = sigmas.view(batch_size, n_comb, q, 1)
        
        sign = 1 if self.maximize else -1
        indiv_ucbs = sign * means + (self.beta ** 0.5) * sigmas + zetas_extended

        min_indiv_ucb = torch.min(indiv_ucbs)
        ucb_mins, _ = indiv_ucbs.min(dim=1, keepdim=True)

        T = self.softmin_temperature
        
        acq_values = (-T * torch.log(torch.sum(torch.exp(-(indiv_ucbs - ucb_mins)/T), dim=1)) + ucb_mins.squeeze(-1)).squeeze(-1).squeeze(-1)
        
        return acq_values

    def optimize_stage_two(self, X: Tensor) -> Tensor:
        r"""Second optimisation stage: choose optimal fidelity to query."""
        
        # Jordan MHS TODO: consider heteroskedastic noise between fidelities. 
        aleatoric_uncertainty = torch.sqrt(self.model.likelihood.noise)
        
        found_suitable_lower_fid = False
        best_fid_idx = None
        optimal_X_cost = None

        prev_fids = None 
        prev_cost = None
        prev_zeta = None

        total_costs_comb = self.costs_comb.sum(dim =-1)
        increasing_cost_order = torch.argsort(total_costs_comb)

        for i in increasing_cost_order: 
            curr_fids = self.fidelities_comb[i].clone()
            curr_cost = self.costs_comb.sum(dim =-1)[i]
            curr_zeta = self.zetas_comb.sum(dim =-1)[i]

            X_curr_fid = X.clone()
            X_curr_fid[:, self.fidelity_columns] = curr_fids

            _, curr_posterior_uncertainty = self._mean_and_sigma(X_curr_fid)

            if prev_cost is not None:

                if (self.beta ** 0.5) * prev_posterior_uncertainty >= (aleatoric_uncertainty + prev_zeta) * torch.sqrt(prev_cost / curr_cost):
                    found_suitable_lower_fid = True
                    optimal_X = X_prev_fid
                    optimal_X_cost = prev_cost
                    break

            prev_fids = curr_fids.clone()
            prev_cost = curr_cost.clone()
            prev_zeta = curr_zeta.clone()
            X_prev_fid = X_curr_fid.clone()
            prev_posterior_uncertainty = curr_posterior_uncertainty.clone()

        if not found_suitable_lower_fid: 
            optimal_X = X_curr_fid
            optimal_X_cost = curr_cost
        
        return optimal_X, optimal_X_cost

# class MultiFidelityBOCA(AnalyticAcquisitionFunction):
#     r"""Two-stage Multi Fidelity Upper Confidence Bound (UCB), based on Kandasamy (2016).

#     Analytic upper confidence bound that comprises of the posterior mean plus two
#     additional terms: the posterior standard deviation weighted by a trade-off
#     parameter, `beta`; and a fidelity-based tolerance parameter (Jordan MHS: BLAH). 
#     Only supports the case of `q=1` (i.e. greedy, non-batch
#     selection of design points). The model must be single-outcome.

#     `UCB(x, m) = mu(x, m) + sqrt(beta) * sigma(x, m) + zeta(m)`, where `mu` and `sigma` are the
#     posterior mean and standard deviation, respectively, and `zeta(m)` is the maximum absolute discrepancy between
#     fidelity `m` and the highest fidelity `M`.

#     `MFUCB(x) = softmin_m(UCB(x, m))` where `softmin_m(v_1, ..., v_m) = (sum_{i=1}^m v_i exp(v_i/T))/(sum_{i=1}^m exp(v_i)`.

#     # Example (Jordan MHS ---update later---):
#     #     >>> model = SingleTaskGP(train_X, train_Y)
#     #     >>> UCB = UpperConfidenceBound(model, beta=0.2)
#     #     >>> ucb = UCB(test_X)
#     """

#     def __init__(
#         self,
#         model: Model,
#         beta: float | Tensor,
#         fidelities: dict[int, tuple[float, ...]],
#         costs: dict[int, tuple[float, ...]],
#         zetas: dict[int, tuple[float, ...]] | None = None,
#         softmin_temperature: float = 1e-2,
#         posterior_transform: PosteriorTransform | None = None,
#         maximize: bool = True,
#         p: int | None = None
#     ) -> None:
#         r"""Bayesian Optimization with Continuous Outcomes. To be used with an RBF kernel (TODO Jordan MHS: check this)
        
#         Args:
#             model: A fitted single-outcome GP model (must be in batch mode if
#                 candidate sets X will be)
#             beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
#                 representing the trade-off parameter between mean and covariance
#             costs: Cost of querying each . Has structure {fidelity_col_idx, costs}.           # Jordan MHS: Finish annotations
#             zetas:                                                                            # Jordan MHS: Explain the BOCA interpretation
#             posterior_transform: A PosteriorTransform. If using a multi-output model,
#                 a PosteriorTransform that transforms the multi-output posterior into a
#                 single-output posterior is required.
#             maximize: If True, consider the problem a maximization problem.
#             p: ... Default set up for a radial basis kernel in fidelity param,,, .            # Jordan MHS: Explain this too
#         """
#         super().__init__(model=model, posterior_transform=posterior_transform)
#         self.register_buffer("beta", torch.as_tensor(beta))
#         self.register_buffer("softmin_temperature", torch.as_tensor(softmin_temperature))
        
#         fidelity_indices = torch.tensor(list(fidelities.keys()), dtype=torch.long)
        
#         # Cartesian product of fidelity values over the indices
        
#         # Possible TODO Jordan MHS: include logical constraints on different fidelity combinations.
#         # Maybe do this by having an optional second format of fidelities
#         fidelity_combos_product = list(iter_product(*fidelities.values()))
#         fidelity_combos_tensor = torch.tensor(fidelity_combos_product, dtype=torch.double)
        
#         self.register_buffer("fidelity_columns", fidelity_indices)
#         self.register_buffer("fidelities_comb", fidelity_combos_tensor)
        
#         # Jordan MHS: use a fidelity parameter-based heuristic for this. 
#         if zetas is None: 
#             zetas = {fid_col: torch.tensor((0.0) * len(fid_vals)) for fid_col, fid_vals in fidelities.items()}
        
#         zetas_product = list(iter_product(*zetas.values()))
#         zetas_tensor = torch.tensor(zetas_product, dtype=torch.double)
        
#         self.register_buffer("zetas_comb", torch.as_tensor(zetas_tensor))
        
#         costs_product = list(iter_product(*costs.values()))
#         costs_tensor = torch.tensor(costs_product, dtype=torch.double)
        
#         self.register_buffer("costs_comb", torch.as_tensor(costs_tensor))
        
#         self.maximize = maximize

#         # if p is None:
#         #     self.p = 
#         # else: 
#         #     self.p = p

#     @t_batch_mode_transform(expected_q=1)
#     def forward(self, X: Tensor) -> Tensor:
#         r"""Evaluate the softmin over Upper Confidence Bounds on the candidate set X.

#         Args:
#             X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design/fidelity points.

#         Returns:
#             A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
#             given design and fidelity points `X`.
#         """

#         batch_size, q, d = X.shape
#         # Jordan MHS: only works for one fidelity col so far
#         n_comb, k = self.fidelities.shape

#         X_extended = X.clone().unsqueeze(1).repeat(1, n_comb, 1, 1)
#         X_extended[..., :, self.fidelity_columns] = self.fidelities.view(1, n_comb, 1, k)
        
#         # If there is more than one fidelity column, treat the the zeta of the fidelity choices as a sum of the contributions per column.
#         # Motivated by a setting where we have different costs and different biases for different parts/stages of an experiment. 
#         zetas_comb_sum = self.zetas_comb.sum(dim=-1)
#         zetas_comb_sum = zetas_comb_sum.view(1, n_comb, 1, 1)
#         zetas_extended = zetas_comb_sum.expand(batch_size, n_comb, q, 1)
        
#         X_eval = X_extended.reshape(batch_size * n_comb, q, d)
#         means, sigmas = self._mean_and_sigma(X_eval)
        
#         means = means.view(batch_size, n_comb, q, 1)
#         sigmas = sigmas.view(batch_size, n_comb, q, 1)
        
#         sign = 1 if self.maximize else -1
#         indiv_ucbs = sign * means + (self.beta ** 0.5) * sigmas

#         min_indiv_ucb = torch.min(indiv_ucbs)
#         ucb_mins, _ = indiv_ucbs.min(dim=1, keepdim=True)

#         T = self.softmin_temperature
        
#         acq_values = (-T * torch.log(torch.sum(torch.exp(-(indiv_ucbs - ucb_mins)/T), dim=1)) + ucb_mins.squeeze(-1)).squeeze(-1).squeeze(-1)
        
#         return acq_values

#     def optimize_stage_two(self, X: Tensor) -> Tensor:
#         r"""Jordan MHS: describe here"""
        
#         # Jordan MHS: only use if kernel supports heteroskedastic noise?
#         aleatoric_uncertainty = torch.sqrt(self.model.likelihood.noise)
        
#         found_suitable_lower_fid = False
#         best_fid_idx = None
#         optimal_X_cost = None

#         prev_fids = None 
#         prev_cost = None
#         prev_zeta = None

#         total_costs_comb = self.costs_comb.sum(dim =-1)
#         increasing_cost_order = torch.argsort(total_costs_comb)

#         for i in increasing_cost_order: 
#             curr_fids = self.fidelities_comb[i].clone()
#             curr_cost = self.costs_comb.sum(dim =-1)[i]
#             curr_zeta = self.zetas_comb.sum(dim =-1)[i]

#             X_curr_fid = X.clone()
#             X_curr_fid[:, self.fidelity_columns] = curr_fids

#             _, curr_posterior_uncertainty = self._mean_and_sigma(X_curr_fid)

#             if prev_cost is not None:

#                 if (self.beta ** 0.5) * prev_posterior_uncertainty >= (aleatoric_uncertainty + prev_zeta) * torch.sqrt(prev_cost / curr_cost):
#                     found_suitable_lower_fid = True
#                     optimal_X = X_prev_fid
#                     optimal_X_cost = prev_cost
#                     break

#             prev_fids = curr_fids.clone()
#             prev_cost = curr_cost.clone()
#             prev_zeta = curr_zeta.clone()
#             X_prev_fid = X_curr_fid.clone()
#             prev_posterior_uncertainty = curr_posterior_uncertainty.clone()

#         if not found_suitable_lower_fid: 
#             optimal_X = X_curr_fid
#             optimal_X_cost = curr_cost
        
#         return optimal_X, optimal_X_cost