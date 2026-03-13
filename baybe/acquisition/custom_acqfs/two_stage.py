"""BayBE two-stage acquisition functions."""

from __future__ import annotations

from itertools import pairwise as iter_pairwise
from itertools import product as iter_product
from typing import cast

import torch
from attrs import define, field
from attrs.validators import deep_iterable, deep_mapping, ge, instance_of
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.acquisition.objective import PosteriorTransform
from botorch.models.model import Model
from botorch.utils.transforms import (
    average_over_ensemble_models,
    t_batch_mode_transform,
)
from gpytorch.likelihoods import GaussianLikelihood
from torch import Tensor
from typing_extensions import override

from baybe.parameters.validation import validate_contains_exactly_one
from baybe.utils.validation import finite_float, validate_dict_shape

_neg_inv_sqrt2 = -0.7071067811865476
_log_sqrt_pi_div_2 = 0.2257913526447274


@define
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

    # Jordan MHS NOTE: typing awkward here since register_buffer does not declare attr.
    fidelity_columns: Tensor
    fidelities_comb: Tensor
    zetas_comb: Tensor
    costs_comb: Tensor

    model: Model = field(validator=instance_of(Model))
    """A fitted single-outcome GP model.
    """

    beta: float | Tensor = field(validator=[instance_of(float), ge(0.0)])
    """Trade-off parameter between mean and covariance.
    """

    fidelities: dict[int, tuple[float, ...]] = field(
        validator=deep_mapping(
            key_validator=instance_of(int),
            value_validator=deep_iterable(
                member_validator=instance_of(float),
                iterable_validator=instance_of(tuple),
            ),
            mapping_validator=instance_of(dict),
        )
    )
    """Computational representation of fidelity values.
    """

    costs: dict[int, tuple[float, ...]] = field(
        validator=deep_mapping(
            key_validator=instance_of(int),
            value_validator=deep_iterable(
                member_validator=(instance_of(float), ge(0.0)),
                iterable_validator=(
                    instance_of(tuple),
                    validate_contains_exactly_one(0.0),
                ),
            ),
            mapping_validator=(instance_of(dict), validate_dict_shape("fidelities")),
        )
    )
    """Cost of querying each fidelity parameter at each fidelity. Costs between
    fidelity parameters are summed.
    """

    zetas: dict[int, tuple[float, ...]] = field(
        validator=deep_mapping(
            key_validator=instance_of(int),
            value_validator=deep_iterable(
                member_validator=(instance_of(float), ge(0.0)),
                iterable_validator=(
                    instance_of(tuple),
                    validate_contains_exactly_one(0.0),
                ),
            ),
            mapping_validator=(instance_of(dict), validate_dict_shape("fidelities")),
        )
    )
    """Maximum absolute discrepancy between each fidelity and the
    highest fidelity output.
    """

    softmin_temperature: float = field(
        converter=float, validator=[finite_float, ge(0.0)], default=1e-2
    )
    """Smoothing parameter for gradient-based optimization of the design.
    """

    posterior_transform: PosteriorTransform | None = field(default=None)
    """PosteriorTransform used to convert multi-output posteriors to
    single-output posteriors if necessary.
    """

    maximize: bool = field(default=True)
    """If True, treat the problem as a maximization problem.
    """

    def __post_attrs_init__(self) -> None:
        super().__init__(model=self.model, posterior_transform=self.posterior_transform)

        self.register_buffer("beta", torch.as_tensor(self.beta))

        self.register_buffer(
            "softmin_temperature", torch.as_tensor(self.softmin_temperature)
        )

        self.register_buffer(
            "fidelity_columns",
            torch.tensor(list(self.fidelities.keys()), dtype=torch.long),
        )

        self.register_buffer(
            "fidelities_comb",
            torch.tensor(
                list(iter_product(*self.fidelities.values())), dtype=torch.double
            ),
        )

        self.register_buffer(
            "zetas_comb",
            torch.tensor(list(iter_product(*self.zetas.values())), dtype=torch.double),
        )

        self.register_buffer(
            "costs_comb",
            torch.tensor(list(iter_product(*self.costs.values())), dtype=torch.double),
        )

    # Jordan MHS NOTE: mypy typing errors for these decorators with on
    # subclasses of AcquistionFunction appear in Botorch as well as here.
    @override
    @t_batch_mode_transform(expected_q=1)  # type: ignore
    @average_over_ensemble_models  # type: ignore
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
        # Jordan MHS NOTE: typing workaround to ignore possibility for botorch
        # AnalyticAcquisitionFunction _mean_and_sigma to have compute_sigma=False.
        sigmas = sigmas.view(batch_size, n_comb, q, 1)  # type: ignore

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
        # Jordan MHS NOTE: casting here because botorch model likelihood is too
        # broadly typed. Check best practice in case likelihood does not have noise.
        likelihood = cast(GaussianLikelihood, self.model.likelihood)

        # Possible TODO: consider heteroskedastic noise between fidelities.
        aleatoric_uncertainty = torch.sqrt(likelihood.noise)

        found_suitable_lower_fid = False

        total_costs_comb = self.costs_comb.sum(dim=-1)
        increasing_cost_order = torch.argsort(total_costs_comb)

        for prev_i, curr_i in iter_pairwise(increasing_cost_order):
            prev_fid = self.fidelities_comb[prev_i].clone()
            prev_cost = self.costs_comb.sum(dim=-1)[prev_i]
            curr_cost = self.costs_comb.sum(dim=-1)[curr_i]
            prev_zeta = self.zetas_comb.sum(dim=-1)[prev_i]

            X_prev_fid = X.clone()
            X_prev_fid[:, self.fidelity_columns] = prev_fid

            _, prev_posterior_uncertainty = self._mean_and_sigma(X_prev_fid)

            # Jordan MHS NOTE: workaround poor typing in Botorch.
            # _mean_and_sigma always returns two values unless the argument
            # compute_sigma is set to False.
            assert prev_posterior_uncertainty is not None, "This shouldn't be accesible"

            if (self.beta**0.5) * prev_posterior_uncertainty >= (
                aleatoric_uncertainty + prev_zeta
            ) * torch.sqrt(prev_cost / curr_cost):
                found_suitable_lower_fid = True
                optimal_X = X_prev_fid
                break

        if not found_suitable_lower_fid:
            optimal_X = X.clone()
            last_fid = self.fidelities_comb[curr_i].clone()
            optimal_X[:, self.fidelity_columns] = last_fid

        return optimal_X
