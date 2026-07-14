"""Rank-weighted GP ensemble (RGPE) surrogate for transfer learning.

Implements the rank-weighted Gaussian process ensemble (RGPE) of Feurer, Letham and
Bakshy (ICML 2018 AutoML Workshop, https://arxiv.org/abs/1802.02219), adapted to BayBE.
One single-task GP is fitted per source task and one on the target task; the ensemble
posterior is a rank-weighted sum of the individual posteriors, where the weights are
derived from a ranking loss (using leave-one-out cross-validation for the target model).
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from attrs import define, field
from attrs.validators import ge, instance_of
from typing_extensions import override

from baybe.surrogates.transfer_learning.base import _SourceTargetTransferSurrogate
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    import pandas as pd
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.posteriors import GPyTorchPosterior, Posterior
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.searchspace.core import SearchSpace


def _roll_col(x: Tensor, shift: int) -> Tensor:
    """Rotate the columns of a tensor to the right by ``shift``.

    Args:
        x: The tensor whose last dimension is rotated.
        shift: The number of positions to rotate by.

    Returns:
        The column-rotated tensor.
    """
    import torch

    return torch.cat((x[..., -shift:], x[..., :-shift]), dim=-1)


def _compute_ranking_loss(f_samps: Tensor, target_y: Tensor) -> Tensor:
    """Compute the ranking loss for each posterior sample over the target points.

    The ranking loss counts the number of discordant pairwise orderings between the
    model's sampled predictions and the observed targets (an exclusive-or of the two
    ``<`` relations). A 3-dimensional ``f_samps`` is interpreted as leave-one-out
    cross-validation samples of the target model (its diagonal holds the out-of-sample
    predictions); a 2-dimensional ``f_samps`` as ordinary samples of a source model.

    Args:
        f_samps: Either an ``n_samples x n x n`` tensor (target LOOCV samples) or an
            ``n_samples x n`` tensor (source model samples).
        target_y: An ``n x 1`` tensor of observed target values.

    Returns:
        An ``n_samples`` tensor with the ranking loss of each sample.
    """
    import torch

    n = target_y.shape[0]
    if f_samps.ndim == 3:
        # Target model: compare each LOO out-of-sample prediction (the diagonal) to
        # every in-sample prediction.
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1),
            target_y.squeeze(-1),
        ).view(n, n, 2)
        rank_loss = (
            (
                (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps)
                ^ (cartesian_y[..., 0] < cartesian_y[..., 1])
            )
            .sum(dim=-1)
            .sum(dim=-1)
        )
    else:
        rank_loss = torch.zeros(
            f_samps.shape[0], dtype=torch.long, device=target_y.device
        )
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1, n):
            rank_loss += (
                (_roll_col(f_samps, i) < f_samps) ^ (_roll_col(y_stack, i) < y_stack)
            ).sum(dim=-1)
    return rank_loss


def _loocv_sample_preds(
    train_x: Tensor, train_y: Tensor, target_model: GPyTorchModel, num_samples: int
) -> Tensor:
    """Draw leave-one-out cross-validation samples of the target model.

    A batch-mode Gaussian process is built in the target model's *transformed* input and
    output space (so no input/output transforms are needed), sharing the target model's
    fitted hyperparameters. Each batch element is trained on all but one target point,
    and a joint sample is drawn across all target points. Working in the transformed
    space is valid here because the ranking loss depends only on the *ordering* of the
    predictions, which the (monotonic) output standardization preserves.

    Args:
        train_x: The target training inputs in the model's transformed input space
            (``n x d``).
        train_y: The target training targets in the model's transformed output space
            (``n x 1``).
        target_model: The fitted target Gaussian process model.
        num_samples: The number of Monte Carlo samples to draw.

    Returns:
        A ``num_samples x n x n`` tensor of samples, where dimension 1 indexes the ``n``
        leave-one-out models and dimension 2 the ``n`` target points.
    """
    from copy import deepcopy

    import torch
    from botorch.models import SingleTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler

    n = len(train_x)
    masks = torch.eye(n, dtype=torch.bool, device=train_x.device)
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])

    # A transform-free batch GP reusing the target model's fitted kernel, mean and
    # likelihood. The (non-batched) modules broadcast across the LOO batch, so every
    # leave-one-out model shares the target model's fitted hyperparameters.
    model = SingleTaskGP(
        train_x_cv,
        train_y_cv,
        covar_module=deepcopy(target_model.covar_module),
        mean_module=deepcopy(target_model.mean_module),
        likelihood=deepcopy(target_model.likelihood),
    )
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        return sampler(posterior).squeeze(-1)


def _compute_rank_weights(
    train_x: Tensor,
    train_y: Tensor,
    source_models: tuple[GPyTorchModel, ...],
    target_model: GPyTorchModel,
    num_samples: int,
) -> Tensor:
    """Compute the RGPE rank weights for the source models and the target model.

    Args:
        train_x: The target training inputs (raw comp-rep, ``n x d``).
        train_y: The target training targets (``n x 1``).
        source_models: The fitted source Gaussian process models.
        target_model: The fitted target Gaussian process model.
        num_samples: The number of Monte Carlo samples used to estimate the weights.

    Returns:
        A tensor of length ``len(source_models) + 1`` with the weight of each model,
        ordered as ``(sources..., target)``.
    """
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler

    ranking_losses = []
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
    for model in source_models:
        posterior = model.posterior(train_x)
        f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        ranking_losses.append(_compute_ranking_loss(f_samps, train_y))

    # The target model's LOOCV samples are drawn in its transformed space. The output
    # transform is only order-preserving, which is all the ranking loss depends on.
    target_model.eval()
    transformed_x = target_model.transform_inputs(train_x)
    transformed_y, _ = target_model.outcome_transform(train_y)  # type: ignore[operator]
    target_f_samps = _loocv_sample_preds(
        transformed_x, transformed_y, target_model, num_samples
    )
    ranking_losses.append(_compute_ranking_loss(target_f_samps, train_y))

    ranking_loss_tensor = torch.stack(ranking_losses)
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    rank_weights = (
        best_models.bincount(minlength=len(ranking_losses)).type_as(train_x)
        / num_samples
    )
    return rank_weights


@define
class RGPETransferSurrogate(_SourceTargetTransferSurrogate):
    """A rank-weighted Gaussian process ensemble (RGPE) transfer learning surrogate.

    Fits one single-task Gaussian process per source task and, once enough target data
    is available, one on the target task. The ensemble posterior is a rank-weighted sum
    of the individual posteriors: ``mean = Σ wᵢ μᵢ`` and ``cov = Σ wᵢ² Σᵢ``.

    The weights are estimated from a ranking loss that measures how well each model
    ranks the observed target points (using leave-one-out cross-validation for the
    target model), following Feurer, Letham and Bakshy (ICML 2018 AutoML Workshop).

    Cold start: at least two target points are required to compute the ranking weights.
    With fewer target points the weights fall back to a uniform average over the
    available models (the source GPs, plus the target GP once it has at least one
    point).

    Note:
        Only a single target task is currently supported. This implementation does not
        address weight dilution across a large number of source tasks.
    """

    num_mc_samples: int = field(default=256, validator=[instance_of(int), ge(1)])
    """The number of Monte Carlo samples used to estimate the ranking weights."""

    _target_gp = field(init=False, default=None, eq=False, repr=False)
    """The single-task GP trained on the target data. ``None`` before fitting or when
    the target task has no measurements yet (cold start)."""

    _weights = field(init=False, default=None, eq=False, repr=False)
    """The ensemble weights, ordered as ``(sources..., target)``. Available after
    fitting. The target entry is present only when a target GP was fitted."""

    @override
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Fit the target GP (if enough data) and compute the ensemble weights.

        Args:
            reduced_searchspace: The task-free search space for the target GP.
            objective: The objective (a single modeled quantity after replication).
            target_measurements: The measurements belonging to the target task (may be
                empty).
        """
        import torch
        from attrs import evolve

        n_target = len(target_measurements)
        source_models = tuple(gp._model for gp in self._source_gps)

        if n_target == 0:
            # Cold start: uniform average over the source models only.
            self._target_gp = None
            n_models = len(source_models)
            self._weights = torch.full((n_models,), 1.0 / n_models)
            return

        self._target_gp = evolve(self.base_surrogate)
        self._target_gp.fit(reduced_searchspace, objective, target_measurements)

        n_models = len(source_models) + 1
        if n_target < 2:
            # Not enough points to rank: uniform average over all available models.
            self._weights = torch.full((n_models,), 1.0 / n_models)
            return

        train_x = to_tensor(
            reduced_searchspace.transform(target_measurements, allow_extra=True)
        )
        train_y = to_tensor(
            objective._pre_transform(target_measurements, allow_extra=True)
        )
        if train_y.ndim == 1:
            train_y = train_y.unsqueeze(-1)
        self._weights = _compute_rank_weights(
            train_x, train_y, source_models, self._target_gp._model, self.num_mc_samples
        )

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the rank-weighted ensemble posterior on task-stripped candidates.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            A posterior whose mean and covariance are the rank-weighted sum of the
            individual model posteriors.
        """
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        assert self._weights is not None  # set during fitting

        reduced_candidates = self._strip_task(candidates_comp_scaled)

        models = [gp for gp in self._source_gps]
        if self._target_gp is not None:
            models.append(self._target_gp)

        weighted_means = []
        weighted_covars = []
        nonzero = self._weights > 0
        weights = self._weights[nonzero] / self._weights[nonzero].sum()
        selected = [m for m, keep in zip(models, nonzero.tolist()) if keep]
        for weight, model in zip(weights, selected):
            posterior = cast("GPyTorchPosterior", model._posterior(reduced_candidates))
            weighted_means.append(weight * posterior.mean.squeeze(-1))
            weighted_covars.append(
                posterior.distribution.lazy_covariance_matrix * weight**2
            )

        import torch

        mean = torch.stack(weighted_means).sum(dim=0)
        covariance = weighted_covars[0]
        for covar in weighted_covars[1:]:
            covariance = covariance + covar
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
