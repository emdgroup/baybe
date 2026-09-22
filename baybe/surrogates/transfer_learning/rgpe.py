"""Rank-weighted Gaussian process ensemble (RGPE) surrogate for transfer learning.

The implementation follows the BoTorch RGPE tutorial:
https://archive.botorch.org/v/latest/tutorials/meta_learning_with_rgpe
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, ClassVar, cast

from attrs import define, evolve, field
from attrs.validators import ge, instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    import pandas as pd
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import GPyTorchPosterior, Posterior
    from gpytorch.means import Mean
    from gpytorch.module import Module
    from torch import Tensor

    from baybe.parameters.base import Parameter
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
                (_roll_col(f_samps, i) < f_samps) ^ (_roll_col(y_stack, i) < y_stack)
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


@define
class RGPESurrogate(Surrogate):
    """A rank-weighted Gaussian process ensemble (RGPE) for transfer learning.

    Implements the ensemble of Feurer, Letham and Bakshy (ICML 2018 AutoML Workshop,
    https://arxiv.org/abs/1802.02219): one Gaussian process is fitted per source task
    and, once enough target data exists, one on the target task. The posterior is a
    rank-weighted sum of the individual posteriors (``mean = Σ wᵢ μᵢ``,
    ``cov = Σ wᵢ² Σᵢ``), with weights measuring how well each model ranks the target
    points (via leave-one-out cross-validation for the target model).

    Select it directly or by setting
    :attr:`~baybe.parameters.categorical.TaskParameter.override_transfer_learning_mode`
    to :attr:`~baybe.parameters.enum.TransferLearningMode.RGPE`.

    Computing the weights needs at least two target points; with fewer, they fall back
    to a uniform average over the available models.

    Note:
        Only single-output objectives and a single target task are supported, and no
        weight-dilution regularization is applied. The ``to_botorch`` representation
        does not support ``fantasize`` (and hence acquisition functions that require it,
        such as the knowledge gradient).
    """

    supports_kernel_overrides: ClassVar[bool] = True
    # See base class. The inner GPs are fitted on the identity-mode space, which still
    # carries the parameter overrides (only the task dimension is made inert).

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    base_surrogate: GaussianProcessSurrogate = field(
        factory=GaussianProcessSurrogate,
        validator=instance_of(GaussianProcessSurrogate),
    )
    """The Gaussian process configuration used for the source and target models."""

    n_mc_samples: int = field(default=256, validator=[instance_of(int), ge(1)])
    """The number of Monte Carlo samples used to estimate the ranking weights."""

    _source_gps: tuple[GaussianProcessSurrogate, ...] = field(
        init=False, factory=tuple, eq=False, repr=False
    )
    """The GPs trained on the source data, one per source task that has measurements.

    Available after fitting.
    """

    _target_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The GP trained on the target data.

    ``None`` before fitting or when the target task has no measurements yet.
    """

    # TODO: type should be `Tensor | None` but is currently omitted due to:
    #   https://github.com/python-attrs/cattrs/issues/531
    _weights = field(init=False, default=None, eq=False, repr=False)
    """The ensemble weights, ordered as ``(sources..., target)``. Available after
    fitting. The target entry is present only when a target GP was fitted."""

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        # The base-class inputs are ignored: the inner GPs are refitted from the stored
        # measurements over the identity-mode space (task dimension inert).
        import torch

        assert self._searchspace is not None  # ensured by base class
        assert self._objective is not None  # ensured by base class
        assert self._measurements is not None  # ensured by base class

        identity_searchspace, sources, target_measurements = self._split_measurements()

        # Fit one source GP per source task that has data.
        source_gps = []
        for _, source_measurements in sources:
            source_gp = evolve(self.base_surrogate)
            source_gp.fit(identity_searchspace, self._objective, source_measurements)
            source_gps.append(source_gp)
        self._source_gps = tuple(source_gps)

        source_models = tuple(gp._model for gp in self._source_gps)
        n_target = len(target_measurements)

        if n_target == 0:
            # Cold start: no target data yet, average over the source models only.
            self._target_gp = None
            n_models = len(source_models)
            self._weights = torch.full((n_models,), 1.0 / n_models)
            return

        self._target_gp = evolve(self.base_surrogate)
        self._target_gp.fit(identity_searchspace, self._objective, target_measurements)

        n_models = len(source_models) + 1
        if n_target < 2:
            # Not enough target points to rank, average over all available models.
            self._weights = torch.full((n_models,), 1.0 / n_models)
            return

        train_x_target = to_tensor(
            identity_searchspace.transform(target_measurements, allow_extra=True)
        )
        train_y_target = to_tensor(
            self._objective._pre_transform(target_measurements, allow_extra=True)
        )
        if train_y_target.ndim == 1:
            train_y_target = train_y_target.unsqueeze(-1)
        self._weights = _compute_rank_weights(
            train_x_target,
            train_y_target,
            source_models,
            self._target_gp._model,
            self.n_mc_samples,
        )

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter, /
    ) -> type[InputTransform] | None:
        # The inner Gaussian processes handle input scaling themselves.
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # The inner Gaussian processes handle output scaling themselves.
        return None

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        import torch
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal

        assert self._weights is not None  # set during fitting

        models: list[GaussianProcessSurrogate] = list(self._source_gps)
        if self._target_gp is not None:
            models.append(self._target_gp)

        # Restrict the combination to models with non-zero weight and renormalize.
        nonzero = self._weights > 0
        weights = self._weights[nonzero] / self._weights[nonzero].sum()
        selected = [m for m, keep in zip(models, nonzero.tolist()) if keep]

        weighted_means = []
        weighted_covars = []
        for weight, model in zip(weights, selected):
            # Inner GPs share the full comp-rep layout, so candidates pass through
            # unchanged.
            posterior = cast(
                "GPyTorchPosterior", model._posterior(candidates_comp_scaled)
            )
            weighted_means.append(weight * posterior.mean.squeeze(-1))
            weighted_covars.append(
                posterior.distribution.lazy_covariance_matrix * weight**2
            )

        mean = torch.stack(weighted_means).sum(dim=0)
        covariance = weighted_covars[0]
        for covar in weighted_covars[1:]:
            covariance = covariance + covar
        return GPyTorchPosterior(MultivariateNormal(mean, covariance))

    def _split_measurements(
        self,
    ) -> tuple[SearchSpace, list[tuple[Any, pd.DataFrame]], pd.DataFrame]:
        """Validate the task configuration and split the measurements by task.

        Returns:
            The search space switched to identity mode (task dimension inert but still
            present in the layout), an ordered list of ``(task_value, measurements)``
            pairs for the source tasks that have data, and the target-task measurements
            (which may be empty).

        Raises:
            IncompatibleSearchSpaceError: If the space has no task parameter, does not
                describe exactly one active (target) task, or if no source task has
                measurements.
        """
        from baybe.parameters.enum import TransferLearningMode

        assert self._searchspace is not None
        assert self._measurements is not None

        searchspace = self._searchspace
        measurements = self._measurements

        task_param = searchspace._task_parameter
        if task_param is None:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires a search space that contains a "
                f"task parameter."
            )

        active_values = set(task_param.active_values)
        source_values = set(task_param.values) - active_values
        if len(active_values) != 1:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires exactly one active (target) "
                f"task value, but the task parameter describes {len(active_values)}."
            )
        # A task parameter has at least two values, so one active value leaves at least
        # one source value.
        (target_value,) = active_values

        task_name = task_param.name
        sources = [
            (value, subset)
            for value in task_param.values
            if value in source_values
            if not (subset := measurements[measurements[task_name] == value]).empty
        ]
        if not sources:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires measurements for at least one "
                f"source task, but none were provided."
            )
        target_measurements = measurements[measurements[task_name] == target_value]

        identity_searchspace = searchspace._with_task_mode(
            TransferLearningMode.IDENTITY
        )
        return identity_searchspace, sources, target_measurements


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
