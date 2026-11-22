"""Rank-weighted Gaussian process ensemble (RGPE) model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any, NoReturn, cast

import torch
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import Self, override

from baybe.exceptions import NoMeasurementsError

if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform
    from botorch.models import SingleTaskGP
    from botorch.posteriors import Posterior
    from gpytorch.likelihoods import GaussianLikelihood
    from linear_operator.operators import LinearOperator

    from baybe.searchspace import SearchSpace

_N_BOOTSTRAP_SAMPLES = 1000
"""The number of bootstrap resamples of the target data used to estimate the rank
weights."""


def _compute_loocv_means(model: SingleTaskGP, /) -> Tensor:
    """Compute the leave-one-out predictive means of a fitted GP at its training points.

    The ``j``-th mean is the prediction at training point ``j`` of the GP conditioned on
    all other training points, keeping the fitted hyperparameters. All means follow in
    closed form from a single Cholesky decomposition (Rasmussen & Williams, Eq. 5.12).

    Args:
        model: The fitted Gaussian process model.

    Returns:
        A tensor with the ``n`` leave-one-out means, in the model's standardized
        output space.
    """
    from linear_operator.utils.cholesky import psd_safe_cholesky

    # After `eval()`, the training inputs are input-transformed and the training
    # targets are standardized
    model.eval()
    train_x = model.train_inputs[0]
    train_y = model.train_targets
    with torch.no_grad():
        prior = model.forward(train_x)
        noisy_prior = model.likelihood(prior)
        assert isinstance(noisy_prior, MultivariateNormal)  # Gaussian likelihood
        precision = torch.cholesky_inverse(
            psd_safe_cholesky(noisy_prior.covariance_matrix)
        )
        residuals = precision @ (train_y - prior.mean)
        return train_y - residuals / precision.diagonal()


def _make_misranking_matrix(
    predictions: Tensor, references: Tensor, targets: Tensor, /
) -> Tensor:
    """Make the matrix of misranked pairs of target points.

    Entry ``[k, l]`` is ``True`` if the order of ``predictions[k]`` and
    ``references[l]`` disagrees with the order of ``targets[k]`` and ``targets[l]``.
    Self-pairs ``k = l`` are never misranked.

    Args:
        predictions: The ``n`` predictions at the target points.
        references: The ``n`` values the predictions are compared against.
        targets: The ``n`` observed target values.

    Returns:
        A Boolean ``n x n`` tensor.
    """
    # NOTE: The paper's Eq. 4 counts the target's self-pairs, which adds about n/2 to
    #   its loss only, as a source's self-pairs are never misranked. Excluding them
    #   keeps the comparison fair and also treats bootstrap duplicates as ties.
    predictions_less = predictions.unsqueeze(-1) < references.unsqueeze(-2)
    targets_less = targets.unsqueeze(-1) < targets.unsqueeze(-2)
    is_self_pair = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    return (predictions_less ^ targets_less) & ~is_self_pair


def _compute_rank_weights(
    train_x: Tensor,
    source_models: tuple[SingleTaskGP, ...],
    target_model: SingleTaskGP,
    /,
    *,
    n_samples: int,
) -> Tensor:
    """Compute the rank weights for the source and target models.

    Each model is weighted by the probability that it has the lowest ranking loss on
    the target points, i.e. that it orders them best. The probability is estimated from
    ``n_samples`` bootstrap resamples of the target data, shared by all models: in each
    resample, the models with the lowest ranking loss share one vote evenly, and a
    model's weight is its average vote.

    The ranking losses are computed on predictive means. Source models predict the
    target points directly (Eq. 3). The target model, which was trained on these points,
    is instead evaluated via its leave-one-out means against the observed values
    (Eq. 4), so that its ranking loss measures generalization rather than in-sample fit.

    To prevent weight dilution by many poor source models, each source model is kept
    only with the probability that it has a lower ranking loss than the target model on
    a resample (Eq. 9 with an infinite horizon), and dropped otherwise.

    Args:
        train_x: The inputs of the ``n`` target points (comp-rep, ``n x d``).
        source_models: The fitted source Gaussian process models.
        target_model: The fitted target Gaussian process model, trained on
            ``train_x``.
        n_samples: The number of bootstrap resamples used to estimate the weights.

    Returns:
        A tensor of length ``len(source_models) + 1`` with the model weights, ordered as
        ``(sources..., target)``.
    """
    # Rankings are invariant under the affine outcome transforms, so source means in
    # original units can be compared with standardized target values
    targets = target_model.train_targets
    n = len(targets)
    with torch.no_grad():
        matrices = []
        for model in source_models:
            mean = model.posterior(train_x).mean.squeeze(-1)
            matrices.append(_make_misranking_matrix(mean, mean, targets))
        loo_means = _compute_loocv_means(target_model)
        matrices.append(_make_misranking_matrix(loo_means, targets, targets))

        # A resample is given by how often it contains each target point, and its
        # ranking loss counts the misranked pairs over all drawn copies of the points
        counts = (
            torch.distributions.Multinomial(n, targets.new_ones(n))
            .sample(torch.Size([n_samples]))
            .to(targets)
        )
        losses = torch.einsum(
            "sk,mkl,sl->ms", counts, torch.stack(matrices).to(targets), counts
        )

        # Keep each source with the probability that it beats the target on a resample
        # NOTE: The comparison is strict as in the paper, so ties count against the
        #   source. Without a known optimization horizon, the prior factor is omitted.
        p_keep = (losses[:-1] < losses[-1]).to(targets).mean(dim=-1)
        is_dropped = torch.rand_like(p_keep) >= p_keep
        losses[:-1][is_dropped] = torch.inf

    # In each resample, the models with the lowest loss share one vote evenly
    is_best = losses == losses.min(dim=0, keepdim=True).values
    rank_weights = (is_best / is_best.sum(dim=0, keepdim=True)).mean(dim=1)
    return rank_weights.type_as(train_x)


class RGPEModel(Model):
    """A rank-weighted ensemble of per-task Gaussian processes.

    The posterior is the weighted sum of the members' posteriors in their original
    output units. Observation noise is the target task's, added once at ensemble level.

    Note:
        Conditioning on observations is not supported, and hence neither are
        acquisition functions that fantasize observations, such as
        :class:`~baybe.acquisition.acqfs.qKnowledgeGradient`.
    """

    def __init__(
        self,
        models: Sequence[SingleTaskGP],
        weights: Tensor,
        /,
        *,
        target_index: int | None = None,
    ):
        super().__init__()
        self.models = ModuleList(models)
        # The normalized ensemble weights, aligned with the members
        self.weights: Tensor
        self.register_buffer("weights", weights)
        # Position of the target member; `None` at cold start (no target data)
        self.target_index = target_index

    @property
    def _members(self) -> tuple[SingleTaskGP, ...]:
        """The members with their concrete type."""
        return cast("tuple[SingleTaskGP, ...]", tuple(self.models))

    @override
    @property
    def batch_shape(self) -> torch.Size:
        return torch.broadcast_shapes(*(m.batch_shape for m in self._members))

    @override
    @property
    def num_outputs(self) -> int:
        return 1

    def fantasize(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Raise, since fantasizing is not supported."""
        raise NotImplementedError(
            f"'{self.__class__.__name__}' does not support fantasizing observations."
        )

    @classmethod
    def from_searchspace(
        cls,
        searchspace: SearchSpace,
        train_x: Tensor,
        train_y: Tensor,
        fit_gp: Callable[[Tensor, Tensor], SingleTaskGP],
        /,
    ) -> Self:
        """Fit one GP per task and combine them into a rank-weighted ensemble.

        Args:
            searchspace: The search space whose task parameter identifies the target
                and source tasks.
            train_x: The training inputs of all tasks (comp-rep, ``n x d``).
            train_y: The training targets of all tasks (``n x 1``).
            fit_gp: A callable fitting a Gaussian process to a subset of the data.

        Returns:
            The fitted ensemble, weighted by how well each member ranks the target
            data. Without two distinct target values, only the source members are
            combined, with uniform weights. With fewer than three target points, all
            members are weighted uniformly.
        """
        target_mask, source_masks = _split_by_task(searchspace, train_x)

        # Sources are fitted before the target, as the weights depend on the RNG state
        source_models = tuple(fit_gp(train_x[m], train_y[m]) for m in source_masks)
        target_y = train_y[target_mask]
        if target_y.unique().numel() < 2:
            # Without two distinct target values, the target GP has no meaningful
            # output scale, so the sources are combined on their own
            weights = train_x.new_full((len(source_models),), 1 / len(source_models))
            return cls.from_ranked_models(source_models, weights)

        target_model = fit_gp(train_x[target_mask], target_y)
        models = (*source_models, target_model)
        if len(target_y) < 3:
            # Ranking starts at three target points, as the target's leave-one-out
            # models need at least two points each
            weights = train_x.new_full((len(models),), 1 / len(models))
        else:
            weights = _compute_rank_weights(
                train_x[target_mask],
                source_models,
                target_model,
                n_samples=_N_BOOTSTRAP_SAMPLES,
            )

        return cls.from_ranked_models(models, weights, target_index=len(models) - 1)

    @classmethod
    def from_ranked_models(
        cls,
        models: Sequence[SingleTaskGP],
        weights: Tensor,
        /,
        *,
        target_index: int | None = None,
    ) -> Self:
        """Drop zero-weight models and renormalize the remaining weights."""
        keep = weights > 0
        if target_index is not None:
            # The target is kept even at zero weight since its noise is needed
            keep[target_index] = True
            target_index = int(keep[:target_index].sum())
        kept = weights[keep]
        return cls(
            [m for m, k in zip(models, keep.tolist()) if k],
            kept / kept.sum(),
            target_index=target_index,
        )

    @override
    def posterior(
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> Posterior:
        if output_indices not in (None, [0]):
            raise ValueError(f"'{self.__class__.__name__}' has a single output.")

        # Mean Σ wᵢμᵢ and covariance Σ wᵢ²Σᵢ in original output units, without
        # observation noise
        distributions = []
        for m in self._members:
            member_posterior = m.posterior(X)
            # Gaussian, since the members' outcome transforms are affine
            assert isinstance(member_posterior, GPyTorchPosterior)
            distributions.append(member_posterior.distribution)
        mean = reduce(add, (w * d.mean for w, d in zip(self.weights, distributions)))
        covariance = reduce(
            add,
            (
                cast("LinearOperator", d.lazy_covariance_matrix) * w**2
                for w, d in zip(self.weights, distributions)
            ),
        )

        if observation_noise is not False:
            if isinstance(observation_noise, Tensor):
                noise = observation_noise
            elif self.target_index is None:
                # No target noise exists, so it is estimated as the weighted average of
                # the source noises. Squared weights would shrink it with more sources.
                noise = reduce(
                    add,
                    (
                        w * _compute_outcome_noise(m)
                        for w, m in zip(self.weights, self._members)
                    ),
                )
            else:
                noise = _compute_outcome_noise(self._members[self.target_index])
            covariance = covariance + noise * torch.eye(
                covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
            )

        posterior = GPyTorchPosterior(MultivariateNormal(mean, covariance))
        if posterior_transform is None:
            return posterior
        return posterior_transform(posterior)


def _compute_outcome_noise(model: SingleTaskGP, /) -> Tensor:
    """Compute a member's observation noise in original outcome units."""
    # The likelihood noise lives in the member's standardized output space
    likelihood = cast("GaussianLikelihood", model.likelihood)
    outcome_transform = model.outcome_transform
    assert isinstance(outcome_transform, Standardize)  # ensured by the GP fit
    return likelihood.noise * outcome_transform.stdvs.squeeze(-1) ** 2


def _split_by_task(
    searchspace: SearchSpace, train_x: Tensor, /
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Split the training rows into a target mask and masks for sources with data."""
    task_param = searchspace._task_parameter
    assert task_param is not None
    assert len(task_param.active_values) == 1
    (target_value,) = task_param.active_values

    codes = task_param.comp_df.iloc[:, 0]
    task_column = train_x[:, searchspace.task_idx]
    target_mask = task_column == float(codes[target_value])
    source_masks = tuple(
        mask
        for value in task_param.values
        if value != target_value
        if (mask := task_column == float(codes[value])).any()
    )
    if not source_masks:
        raise NoMeasurementsError(
            "RGPE requires measurements for at least one source task, but none were "
            "provided."
        )
    return target_mask, source_masks
