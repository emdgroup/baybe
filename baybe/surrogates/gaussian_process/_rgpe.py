"""Rank-weighted Gaussian process ensemble (RGPE) model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any, NoReturn, cast

import torch
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import LikelihoodList
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import Self, override

from baybe.exceptions import IncompatibleSearchSpaceError

if TYPE_CHECKING:
    from botorch.acquisition.objective import PosteriorTransform
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import Mean
    from gpytorch.module import Module
    from linear_operator.operators import LinearOperator

    from baybe.surrogates.gaussian_process.core import _ModelContext

_N_MC_SAMPLES = 256
"""Monte Carlo samples for estimating the ranking weights."""

_CONDITIONING_ERROR_MESSAGE = (
    "'{}' does not support conditioning on observations or fantasizing."
)


def _compute_ranking_loss(f_samps: Tensor, target_y: Tensor) -> Tensor:
    """Compute the ranking loss of each posterior sample against the target points.

    The loss counts discordant pairwise orderings between the sampled predictions and
    the observed targets. A 3D ``f_samps`` holds leave-one-out samples of the target
    model (its diagonal are the out-of-sample predictions); a 2D ``f_samps`` holds
    source-model samples.

    Args:
        f_samps: An ``n_samples x n x n`` tensor (target LOOCV samples) or an
            ``n_samples x n`` tensor (source-model samples).
        target_y: An ``n x 1`` tensor of observed target values.

    Returns:
        An ``n_samples`` tensor with the ranking loss per sample.
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
                (torch.roll(f_samps, i, dims=-1) < f_samps)
                ^ (torch.roll(y_stack, i, dims=-1) < y_stack)
            ).sum(dim=-1)
    return rank_loss


def _loocv_sample_preds(
    train_x: Tensor, train_y: Tensor, target_model: GPyTorchModel, n_samples: int
) -> Tensor:
    """Draw leave-one-out cross-validation samples of the target model.

    Builds a batch GP in the target model's transformed input/output space, reusing its
    fitted hyperparameters, with each batch element trained on all but one target point.
    Working in the transformed space is fine because the ranking loss only depends on
    the ordering of predictions, which output standardization preserves.

    Args:
        train_x: The target inputs in the model's transformed input space (``n x d``).
        train_y: The target values in the model's transformed output space (``n x 1``).
        target_model: The fitted target Gaussian process model.
        n_samples: The number of Monte Carlo samples to draw.

    Returns:
        An ``n_samples x n x n`` tensor, indexing the ``n`` leave-one-out models along
        dimension 1 and the ``n`` target points along dimension 2.
    """
    from copy import deepcopy

    import torch
    from botorch.models import SingleTaskGP
    from botorch.sampling.normal import SobolQMCNormalSampler

    n = len(train_x)
    masks = torch.eye(n, dtype=torch.bool, device=train_x.device)
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])

    # Transform-free batch GP reusing the target model's fitted modules, which broadcast
    # across the LOO batch. Transforms are off since the data is already in the model's
    # transformed space (this also avoids re-standardizing the standardized targets).
    model = SingleTaskGP(
        train_x_cv,
        train_y_cv,
        covar_module=cast("Module", deepcopy(target_model.covar_module)),
        mean_module=cast("Mean", deepcopy(target_model.mean_module)),
        likelihood=deepcopy(target_model.likelihood),
        input_transform=None,
        outcome_transform=None,
    )
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
        return sampler(posterior).squeeze(-1)


def _compute_rank_weights(
    train_x: Tensor,
    train_y: Tensor,
    source_models: tuple[GPyTorchModel, ...],
    target_model: GPyTorchModel,
    n_samples: int,
) -> Tensor:
    """Compute the RGPE rank weights for the source and target models.

    Each model is weighted by the Monte Carlo probability that it has the lowest ranking
    loss on the target points, following Feurer, Letham and Bakshy (ICML 2018 AutoML
    Workshop). Mirrors the BoTorch RGPE tutorial without weight-dilution regularization.

    Args:
        train_x: The target training inputs (raw comp-rep, ``n x d``).
        train_y: The target training targets (``n x 1``).
        source_models: The fitted source Gaussian process models.
        target_model: The fitted target Gaussian process model.
        n_samples: The number of Monte Carlo samples used to estimate the weights.

    Returns:
        A tensor of length ``len(source_models) + 1`` with the model weights, ordered as
        ``(sources..., target)``.
    """
    import torch
    from botorch.sampling.normal import SobolQMCNormalSampler

    ranking_losses = []
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([n_samples]))
    for model in source_models:
        posterior = model.posterior(train_x)
        f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        ranking_losses.append(_compute_ranking_loss(f_samps, train_y))

    # The target model's LOOCV samples are drawn in its transformed space. The output
    # transform is only order-preserving, which is all the ranking loss depends on.
    target_model.eval()
    transformed_x = target_model.transform_inputs(train_x)
    # `outcome_transform` is always present on a fitted SingleTaskGP, but mypy sees it
    # as an optional, non-callable attribute.
    transformed_y, _ = target_model.outcome_transform(train_y)  # type: ignore[operator]
    target_f_samps = _loocv_sample_preds(
        transformed_x, transformed_y, target_model, n_samples
    )
    ranking_losses.append(_compute_ranking_loss(target_f_samps, train_y))

    ranking_loss_tensor = torch.stack(ranking_losses)
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    rank_weights = (
        best_models.bincount(minlength=len(ranking_losses)).type_as(train_x) / n_samples
    )
    return rank_weights


class RGPEModel(GPyTorchModel):
    """A rank-weighted ensemble of per-task Gaussian processes.

    The posterior is the weighted sum of the members' posteriors. Observation noise is
    the target task's, added once at ensemble level.

    Note:
        Tools that require a single exact GP (e.g. pathwise sampling) are not supported.
        Neither is conditioning on observations, and hence fantasizing.
    """

    # Required by `ModelListGP`, which sums its members' outputs
    _num_outputs = 1

    weights: Tensor
    """The normalized ensemble weights, aligned with the members."""

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
        self.register_buffer("weights", weights)
        # Position of the target member; `None` at cold start (no target data)
        self.target_index = target_index

        # Required by `ModelListGP`. Never used for noise computation, which
        # `posterior` handles explicitly.
        self.likelihood = LikelihoodList(*(m.likelihood for m in models))

    @property
    def _members(self) -> tuple[SingleTaskGP, ...]:
        """The members with their concrete type."""
        return cast("tuple[SingleTaskGP, ...]", tuple(self.models))

    @override
    @property
    def batch_shape(self) -> torch.Size:
        # The inherited version reads `train_inputs`, which this model doesn't have
        return torch.broadcast_shapes(*(m.batch_shape for m in self._members))

    @override
    def condition_on_observations(
        self, X: Tensor, Y: Tensor, noise: Tensor | None = None, **kwargs: Any
    ) -> NoReturn:
        raise NotImplementedError(
            _CONDITIONING_ERROR_MESSAGE.format(self.__class__.__name__)
        )

    def fantasize(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Raise, since conditioning on (fantasized) observations is not supported.

        Defined explicitly so that fantasy-based tools (e.g. knowledge gradient)
        fail with an informative error.

        Args:
            *args: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            _CONDITIONING_ERROR_MESSAGE.format(self.__class__.__name__)
        )

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
    def load_state_dict(  # type: ignore[override]  # BoTorch adds `keep_transforms`
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        keep_transforms: bool = True,
        assign: bool = False,
    ) -> None:
        # The inherited version reads `train_targets`, which this model doesn't have.
        # Each member keeps its own transforms, so `keep_transforms` has no effect.
        torch.nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

    @override
    def posterior(  # type: ignore[override]  # Follows `Model`; GPyTorchModel differs
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool | Tensor = False,
        posterior_transform: PosteriorTransform | None = None,
        **kwargs: Any,
    ) -> GPyTorchPosterior:
        if output_indices not in (None, [0]):
            raise ValueError(f"'{self.__class__.__name__}' has a single output.")

        # Members are combined latent and in original output units:
        # mean Σ wᵢμᵢ, covariance Σ wᵢ²Σᵢ
        # Members have linear outcome transforms, so their posteriors are Gaussian
        posteriors = [cast(GPyTorchPosterior, m.posterior(X)) for m in self._members]
        mean = reduce(
            add, (w * p.mean.squeeze(-1) for w, p in zip(self.weights, posteriors))
        )
        covariance = reduce(
            add,
            (
                cast("LinearOperator", p.distribution.lazy_covariance_matrix) * w**2
                for w, p in zip(self.weights, posteriors)
            ),
        )

        if observation_noise is not False:
            if isinstance(observation_noise, Tensor):
                noise = observation_noise
            elif self.target_index is not None:
                noise = _compute_outcome_noise(self._members[self.target_index])
            else:
                # Cold start: no target noise exists, use the weighted source noises
                noise = reduce(
                    add,
                    (
                        w * _compute_outcome_noise(m)
                        for w, m in zip(self.weights, self._members)
                    ),
                )
            covariance = covariance + noise * torch.eye(
                covariance.shape[-1], dtype=covariance.dtype, device=covariance.device
            )

        posterior = GPyTorchPosterior(MultivariateNormal(mean, covariance))
        if posterior_transform is None:
            return posterior
        # BoTorch types the result as `Posterior`, but scalarization keeps it Gaussian
        return cast(GPyTorchPosterior, posterior_transform(posterior))


def _compute_outcome_noise(model: SingleTaskGP, /) -> Tensor:
    """Compute a member's observation noise in original outcome units."""
    # The likelihood noise lives in the member's standardized output space
    likelihood = cast("GaussianLikelihood", model.likelihood)
    outcome_transform = cast("Standardize", model.outcome_transform)
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
        raise IncompatibleSearchSpaceError(
            "RGPE requires measurements for at least one source task, but none were "
            "provided."
        )
    return target_mask, source_masks


def make_rgpe_model(
    context: _ModelContext,
    train_x: Tensor,
    train_y: Tensor,
    fit_gp: Callable[[Tensor, Tensor], SingleTaskGP],
    /,
) -> RGPEModel:
    """Fit one GP per task and combine them into a rank-weighted ensemble."""
    target_mask, source_masks = _split_by_task(context, train_x)

    # Sources are fitted before the target, as the weights depend on the RNG state
    source_models = tuple(fit_gp(train_x[m], train_y[m]) for m in source_masks)
    n_target = int(target_mask.sum())
    if n_target == 0:
        weights = torch.full(
            (len(source_models),), 1 / len(source_models), dtype=train_x.dtype
        )
        return RGPEModel.from_ranked_models(source_models, weights)

    target_model = fit_gp(train_x[target_mask], train_y[target_mask])
    models = (*source_models, target_model)
    if n_target < 2:
        # Ranking needs at least two target points
        weights = torch.full((len(models),), 1 / len(models), dtype=train_x.dtype)
    else:
        weights = _compute_rank_weights(
            train_x[target_mask],
            train_y[target_mask],
            source_models,
            target_model,
            _N_MC_SAMPLES,
        )
    return RGPEModel.from_ranked_models(models, weights, target_index=len(models) - 1)
