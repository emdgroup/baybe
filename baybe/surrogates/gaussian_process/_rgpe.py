"""Rank-weighted Gaussian process ensemble (RGPE) model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any, NoReturn, cast

import torch
from botorch.models.model import Model
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils import gpt_posterior_settings
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

    from baybe.surrogates.gaussian_process.core import _ModelContext

_N_MC_SAMPLES = 256
"""The number of posterior samples used to estimate the rank weights."""


def _compute_ranking_loss(samples: Tensor, targets: Tensor, /) -> Tensor:
    """Compute the ranking loss, i.e. the number of misranked pairs, per sample.

    Row ``j`` of ``samples`` holds the predictions of the model used to rank the target
    point ``j``. A pair ``(j, k)`` is misranked if the order of ``samples[s, j, j]`` and
    ``samples[s, j, k]`` disagrees with the order of ``targets[j]`` and ``targets[k]``.

    For a source model, every row holds the same predictions. For the target model, row
    ``j`` holds the predictions of its leave-one-out model without point ``j``.

    Args:
        samples: A ``n_samples x n x n`` tensor of predictions at the ``n`` target
            points, whose entry ``[s, j, k]`` is the ``s``-th posterior sample of the
            model for row ``j``, evaluated at point ``k``.
        targets: A tensor of the ``n`` observed target values.

    Returns:
        An ``n_samples`` tensor with the number of misranked pairs per sample.
    """
    predictions = samples.diagonal(dim1=-2, dim2=-1)
    predictions_less = predictions.unsqueeze(-1) < samples
    targets_less = targets.unsqueeze(-1) < targets.unsqueeze(-2)
    return (predictions_less ^ targets_less).sum(dim=(-2, -1))


def _sample_loocv_predictions(model: SingleTaskGP, n_samples: int, /) -> Tensor:
    """Draw posterior samples from the leave-one-out versions of a fitted GP.

    The ``j``-th leave-one-out GP is the given GP conditioned on all its training points
    except the ``j``-th one, keeping the fitted hyperparameters. Each of these GPs is
    sampled jointly at all ``n`` training points, including the one it left out.

    The samples are drawn from the latent function, i.e. without observation noise, and
    in the model's standardized output space. This suffices for the ranking loss, which
    only depends on the order of the predictions.

    Args:
        model: The fitted Gaussian process model.
        n_samples: The number of posterior samples to draw.

    Returns:
        A tensor of shape ``n_samples x n x n``, whose entry ``[s, j, k]`` is the
        ``s``-th posterior sample of the GP without training point ``j``, evaluated at
        training point ``k``.
    """
    from copy import deepcopy

    from botorch.sampling.normal import SobolQMCNormalSampler

    # In eval mode, the training data is in transformed form
    model.eval()
    train_x = model.train_inputs[0]
    train_y = model.train_targets

    # Fold j contains all training points except point j
    masks = torch.eye(len(train_x), dtype=torch.bool, device=train_x.device)

    # Copy with the folds as batched training data, which `set_train_data` stores as is
    loo_model = deepcopy(model)
    # `set_train_data` comes from GPyTorch, which mypy can't see, so mypy
    # falls back to `nn.Module.__getattr__` and types it as a tensor
    loo_model.set_train_data(  # type: ignore[operator]
        torch.stack([train_x[~m] for m in masks]),
        torch.stack([train_y[~m] for m in masks]),
        strict=False,
    )

    with torch.no_grad(), gpt_posterior_settings():
        # Calling the model directly skips the transforms
        posterior = GPyTorchPosterior(loo_model(train_x))
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
        return sampler(posterior).squeeze(-1)


def _compute_rank_weights(
    train_x: Tensor,
    targets: Tensor,
    source_models: tuple[SingleTaskGP, ...],
    target_model: SingleTaskGP,
    /,
    *,
    n_samples: int,
) -> Tensor:
    """Compute the rank weights for the source and target models.

    Each model is weighted by the probability that it has the lowest ranking loss on
    the target points, i.e. that it orders them best. The probability is estimated from
    ``n_samples`` joint posterior samples per model: in each sample, the models with the
    lowest ranking loss share one vote evenly, and a model's weight is its average vote.

    Source models are evaluated at the target points directly. The target model, which
    was trained on these points, is instead evaluated via its leave-one-out models, so
    that its ranking loss measures generalization rather than in-sample fit.

    Args:
        train_x: The inputs of the ``n`` target points (comp-rep, ``n x d``).
        targets: The ``n`` observed target values.
        source_models: The fitted source Gaussian process models.
        target_model: The fitted target Gaussian process model, trained on
            ``train_x`` and ``targets``.
        n_samples: The number of posterior samples used to estimate the weights.

    Returns:
        A tensor of length ``len(source_models) + 1`` with the model weights, ordered as
        ``(sources..., target)``.
    """
    from botorch.sampling.normal import SobolQMCNormalSampler

    n = len(targets)
    ranking_losses = []
    with torch.no_grad():
        for model in source_models:
            # Each source gets its own sampler, so their samples are independent
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
            f = sampler(model.posterior(train_x)).squeeze(-1)
            # A source model ranks every point itself, so all rows are the same
            samples = f.unsqueeze(-2).expand(-1, n, -1)
            ranking_losses.append(_compute_ranking_loss(samples, targets))

        # The target model ranks point j with its leave-one-out model without point j
        samples = _sample_loocv_predictions(target_model, n_samples)
        ranking_losses.append(_compute_ranking_loss(samples, targets))

    ranking_loss_tensor = torch.stack(ranking_losses)
    # In each sample, the models with the lowest loss share one vote evenly
    is_best = ranking_loss_tensor == ranking_loss_tensor.min(dim=0, keepdim=True).values
    rank_weights = (is_best / is_best.sum(dim=0, keepdim=True)).mean(dim=1)
    return rank_weights.type_as(train_x)


class RGPEModel(Model):
    """A rank-weighted ensemble of per-task Gaussian processes.

    The posterior is the weighted sum of the members' posteriors in their standardized
    output spaces, mapped to the target's output units via the target member's outcome
    transform. Observation noise is the target task's, added once at ensemble level.

    Note:
        Tools that require a single exact GP (e.g. pathwise sampling) are not supported.
        Neither is conditioning on observations, and hence fantasizing.
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
        if target_index is not None:
            # Only set with a target, as BoTorch checks for the attribute's existence.
            # The transform maps the combined standardized predictions to target units,
            # which requires it to be affine.
            outcome_transform = models[target_index].outcome_transform
            assert isinstance(outcome_transform, Standardize)  # ensured by the GP fit
            self.outcome_transform = outcome_transform

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
    def from_context(
        cls,
        context: _ModelContext,
        train_x: Tensor,
        train_y: Tensor,
        fit_gp: Callable[[Tensor, Tensor], SingleTaskGP],
        /,
    ) -> Self:
        """Fit one GP per task and combine them into a rank-weighted ensemble.

        Args:
            context: The model context providing the search space with the task
                parameter identifying the target and source tasks.
            train_x: The training inputs of all tasks (comp-rep, ``n x d``).
            train_y: The training targets of all tasks (``n x 1``).
            fit_gp: A callable fitting a Gaussian process to a subset of the data.

        Returns:
            The fitted ensemble, weighted by how well each member ranks the target
            data. Falls back to uniform weights with fewer than two target points.
        """
        target_mask, source_masks = _split_by_task(context, train_x)

        # Sources are fitted before the target, as the weights depend on the RNG state
        source_models = tuple(fit_gp(train_x[m], train_y[m]) for m in source_masks)
        n_target = int(target_mask.sum())
        if n_target == 0:
            weights = train_x.new_full((len(source_models),), 1 / len(source_models))
            return cls.from_ranked_models(source_models, weights)

        target_model = fit_gp(train_x[target_mask], train_y[target_mask])
        models = (*source_models, target_model)
        if n_target < 2:
            # Ranking needs at least two target points
            weights = train_x.new_full((len(models),), 1 / len(models))
        else:
            weights = _compute_rank_weights(
                train_x[target_mask],
                train_y[target_mask].squeeze(-1),
                source_models,
                target_model,
                n_samples=_N_MC_SAMPLES,
            )

        # The target's standardization defines the ensemble's output units. For a
        # single or constant target point, BoTorch falls back to unit scale around
        # the mean.
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

        # Mean Σ wᵢμᵢ and covariance Σ wᵢ²Σᵢ, without observation noise
        self.eval()
        if self.target_index is None:
            # Without a target scale, the members are combined in their own units
            distributions = []
            for m in self._members:
                member_posterior = m.posterior(X)
                # Gaussian, since the members' outcome transforms are affine
                assert isinstance(member_posterior, GPyTorchPosterior)
                distributions.append(member_posterior.distribution)
        else:
            # Standardized member distributions, i.e. before their outcome transforms
            with gpt_posterior_settings():
                distributions = [m(m.transform_inputs(X)) for m in self._members]
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
                # No target noise exists, use the weighted source noises
                noise = reduce(
                    add,
                    (
                        w * _compute_outcome_noise(m)
                        for w, m in zip(self.weights, self._members)
                    ),
                )
            else:
                # The likelihood noise lives in the standardized target space
                target = self._members[self.target_index]
                noise = cast("GaussianLikelihood", target.likelihood).noise
            covariance = covariance + noise * torch.eye(
                covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
            )

        posterior: Posterior = GPyTorchPosterior(MultivariateNormal(mean, covariance))
        if self.target_index is not None:
            posterior = self.outcome_transform.untransform_posterior(posterior)
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
    context: _ModelContext, train_x: Tensor, /
) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Split the training rows into a target mask and one mask per source task."""
    task_param = context.searchspace._task_parameter
    assert task_param is not None
    assert len(task_param.active_values) == 1
    (target_value,) = task_param.active_values

    codes = task_param.comp_df.iloc[:, 0]
    task_column = train_x[:, context.task_idx]
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
