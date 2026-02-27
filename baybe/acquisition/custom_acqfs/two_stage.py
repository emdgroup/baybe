"""BayBE two-stage acquisition functions."""

from __future__ import annotations

from itertools import product as iter_product

import torch
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import (
    t_batch_mode_transform,
)
from torch import Tensor

_neg_inv_sqrt2 = -0.7071067811865476
_log_sqrt_pi_div_2 = 0.2257913526447274


class MultiFidelityUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Two-stage Multi Fidelity Upper Confidence Bound (UCB).

    First stage selects the design parameter choice through a discrepancy-parameter
    adjusted upper confidence bound. Selection is done by gradient-based optimization
    of a softmin over each fidelity-adjusted UCB.
    Second stage makes a cost-aware decision of the fidelity parameter to be queried, by
    searching through each fidelity at the chosen design parameter, which balances cost
    of querying with fidelity-specific UCB.

    Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.
    """

    # Jordan MHS TODO: Initialize via attrs and not __init__.
    def __init__(
        self,
        model: Model,
        beta: float | Tensor,
        fidelities: dict[int, tuple[float, ...]],
        costs: dict[int, tuple[float, ...]],
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
            softmin_temperature: smoothing parameter for gradient based optimization
                of design.
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer(
            "softmin_temperature", torch.as_tensor(softmin_temperature)
        )

        fidelity_indices = torch.tensor(list(fidelities.keys()), dtype=torch.long)

        fidelity_combos_product = list(iter_product(*fidelities.values()))
        fidelity_combos_tensor = torch.tensor(
            fidelity_combos_product, dtype=torch.double
        )

        self.register_buffer("fidelity_columns", fidelity_indices)
        self.register_buffer("fidelities_comb", fidelity_combos_tensor)

        zetas_product = list(iter_product(*zetas.values()))
        zetas_tensor = torch.tensor(zetas_product, dtype=torch.double)

        self.register_buffer("zetas_comb", torch.as_tensor(zetas_tensor))

        costs_product = list(iter_product(*costs.values()))
        costs_tensor = torch.tensor(costs_product, dtype=torch.double)

        self.register_buffer("costs_comb", torch.as_tensor(costs_tensor))

        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""First optimization stage: choose optimal design design to query.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim tensor of `d`-dim design/fidelity points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design and fidelity points `X`.
        """
        batch_size, q, d = X.shape

        n_comb, k = self.fidelities_comb.shape

        X_extended = X.clone().unsqueeze(1).repeat(1, n_comb, 1, 1)
        X_extended[..., :, self.fidelity_columns] = self.fidelities_comb.view(
            1, n_comb, 1, k
        )

        zetas_comb_sum = self.zetas_comb.sum(dim=-1)
        zetas_comb_sum = zetas_comb_sum.view(1, n_comb, 1, 1)
        zetas_extended = zetas_comb_sum.expand(batch_size, n_comb, q, 1)

        X_eval = X_extended.reshape(batch_size * n_comb, q, d)
        means, sigmas = self._mean_and_sigma(X_eval)

        means = means.view(batch_size, n_comb, q, 1)
        sigmas = sigmas.view(batch_size, n_comb, q, 1)

        sign = 1 if self.maximize else -1
        indiv_ucbs = sign * means + (self.beta**0.5) * sigmas + zetas_extended

        ucb_mins, _ = indiv_ucbs.min(dim=1, keepdim=True)

        T = self.softmin_temperature

        acq_values = (
            (
                -T
                * torch.log(torch.sum(torch.exp(-(indiv_ucbs - ucb_mins) / T), dim=1))
                + ucb_mins.squeeze(-1)
            )
            .squeeze(-1)
            .squeeze(-1)
        )

        return acq_values

    def optimize_stage_two(self, X: Tensor) -> Tensor:
        r"""Second optimisation stage: choose optimal fidelity to query."""
        # Jordan MHS possible TODO: consider heteroskedastic noise between fidelities.
        aleatoric_uncertainty = torch.sqrt(self.model.likelihood.noise)

        found_suitable_lower_fid = False
        optimal_X_cost = None

        prev_fid = None
        prev_cost = None
        prev_zeta = None

        total_costs_comb = self.costs_comb.sum(dim=-1)
        increasing_cost_order = torch.argsort(total_costs_comb)

        for i in increasing_cost_order:
            curr_fid = self.fidelities_comb[i].clone()
            curr_cost = self.costs_comb.sum(dim=-1)[i]
            curr_zeta = self.zetas_comb.sum(dim=-1)[i]

            if prev_cost is not None:
                X_prev_fid = X.clone()
                X_prev_fid[:, self.fidelity_columns] = prev_fid

                _, curr_posterior_uncertainty = self._mean_and_sigma(X_prev_fid)

                if (self.beta**0.5) * curr_posterior_uncertainty >= (
                    aleatoric_uncertainty + prev_zeta
                ) * torch.sqrt(prev_cost / curr_cost):
                    found_suitable_lower_fid = True
                    optimal_X = X_prev_fid
                    optimal_X_cost = prev_cost
                    break

            prev_fid = curr_fid.clone()
            prev_cost = curr_cost.clone()
            prev_zeta = curr_zeta.clone()

        if not found_suitable_lower_fid:
            optimal_X = X.clone()
            optimal_X[:, self.fidelity_columns] = curr_fid
            optimal_X_cost = curr_cost

        return optimal_X, optimal_X_cost
