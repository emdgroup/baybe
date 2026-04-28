"""RGPE (Rank-Weighted GP Ensemble) surrogate for transfer learning.

Implements the Rank-weighted GP Ensemble from:
Feurer, Letham, Bakshy. "Scalable Meta-Learning for Bayesian Optimization using
Ranking-Weighted Gaussian Process Ensembles." ICML AutoML Workshop, 2018.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import torch
from attrs import define, field
from attrs.validators import ge, instance_of
from torch import Tensor
from torch.nn import ModuleList
from typing_extensions import override

from baybe.parameters.base import Parameter
from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior

    from baybe.surrogates.gaussian_process.components.kernel import (
        KernelFactoryProtocol,
    )


# --- Helper functions ---


def _roll_col(X: Tensor, shift: int) -> Tensor:
    """Rotate columns to right by shift."""
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def _compute_ranking_loss(f_samps: Tensor, target_y: Tensor) -> Tensor:
    """Compute ranking loss for each sample from the posterior over target points.

    Args:
        f_samps: ``n_samples x (n) x n``-dim tensor of samples.
            If 3D, the diagonal represents LOO predictions for the target model.
        target_y: ``n x 1``-dim tensor of targets.

    Returns:
        ``n_samples``-dim tensor containing the ranking loss for each sample.
    """
    n = target_y.shape[0]
    if f_samps.ndim == 3:
        # Target model LOOCV case
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
        # Base model case
        rank_loss = torch.zeros(
            f_samps.shape[0], dtype=torch.long, device=target_y.device
        )
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1, target_y.shape[0]):
            rank_loss += (
                (_roll_col(f_samps, i) < f_samps) ^ (_roll_col(y_stack, i) < y_stack)
            ).sum(dim=-1)
    return rank_loss


def _get_target_model_loocv_sample_preds(
    train_x: Tensor,
    train_y: Tensor,
    target_model: SingleTaskGP,
    num_samples: int,
) -> Tensor:
    """Create a batch-mode LOOCV GP and draw joint samples.

    Uses the hyperparameters from target_model to create n LOO models
    (one for each training point left out) and draws joint samples.

    Args:
        train_x: ``n x d`` tensor of training points.
        train_y: ``n x 1`` tensor of training targets.
        target_model: Fitted target model whose hyperparameters are reused.
        num_samples: Number of MC samples to draw.

    Returns:
        ``num_samples x n x n``-dim tensor of samples.
    """
    from copy import deepcopy

    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import (
        ChainedInputTransform,
        FilterFeatures,
        Normalize,
    )
    from botorch.models.transforms.outcome import Standardize
    from botorch.sampling.normal import SobolQMCNormalSampler

    batch_size = len(train_x)
    masks = torch.eye(batch_size, dtype=torch.bool, device=train_x.device)
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])

    # Create fresh transforms for batch-mode model
    # Copy the FilterFeatures config from the target model's input_transform
    orig_input_tf = target_model.input_transform
    if isinstance(orig_input_tf, ChainedInputTransform):
        filter_tf = orig_input_tf["filter"]
        feature_indices = filter_tf.feature_indices
        n_features = len(feature_indices)
        input_transform = ChainedInputTransform(
            filter=FilterFeatures(feature_indices=feature_indices.clone()),
            normalize=Normalize(d=n_features, batch_shape=torch.Size([batch_size])),
        )
    else:
        input_transform = deepcopy(orig_input_tf)

    # Create batch-mode model
    model = SingleTaskGP(
        train_X=train_x_cv,
        train_Y=train_y_cv,
        input_transform=input_transform,
        outcome_transform=Standardize(1, batch_shape=torch.Size([batch_size])),
    )

    # Load hyperparameters from target model (only covar and likelihood params)
    target_state = target_model.state_dict()
    model_state = model.state_dict()
    for name, t in target_state.items():
        if name in model_state:
            target_shape = model_state[name].shape
            if t.shape == target_shape:
                model_state[name] = t
            elif t.dtype.is_floating_point and t.ndim > 0:
                # Try to expand to batch shape
                try:
                    model_state[name] = t.expand(target_shape)
                except RuntimeError:
                    pass
    model.load_state_dict(model_state)
    model.eval()

    with torch.no_grad():
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
        return sampler(posterior).squeeze(-1)


def _compute_rank_weights(
    train_x: Tensor,
    train_y: Tensor,
    base_models: list[SingleTaskGP],
    target_model: SingleTaskGP,
    num_samples: int,
) -> Tensor:
    """Compute ranking weights for base models and target model.

    Args:
        train_x: ``n x d`` tensor of target training points.
        train_y: ``n x 1`` tensor of target training targets.
        base_models: List of fitted base (source) models.
        target_model: Fitted target model.
        num_samples: Number of MC samples for weight estimation.

    Returns:
        ``n_models``-dim tensor with ranking weight for each model
        (base models first, target model last). Weights sum to 1.
    """
    from botorch.sampling.normal import SobolQMCNormalSampler

    ranking_losses = []

    # Compute ranking loss for each base model
    for model in base_models:
        with torch.no_grad():
            posterior = model.posterior(train_x)
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([num_samples]))
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        ranking_losses.append(_compute_ranking_loss(base_f_samps, train_y))

    # Compute ranking loss for target model using LOOCV
    target_f_samps = _get_target_model_loocv_sample_preds(
        train_x, train_y, target_model, num_samples
    )
    ranking_losses.append(_compute_ranking_loss(target_f_samps, train_y))

    ranking_loss_tensor = torch.stack(ranking_losses)

    # Compute best model (minimum ranking loss) for each sample
    best_models = torch.argmin(ranking_loss_tensor, dim=0)

    # Compute proportion of samples for which each model is best
    rank_weights = (
        best_models.bincount(minlength=len(ranking_losses)).type_as(train_x)
        / num_samples
    )
    return rank_weights


# --- BoTorch Model ---


class RGPEModel(torch.nn.Module):
    """Rank-weighted GP ensemble as a BoTorch-compatible model.

    Combines multiple GP models using rank-based weights. The ensemble prediction
    is a weighted sum of individual model posteriors:
        mean = sum(w_i * mu_i)
        var  = sum(w_i^2 * sigma_i^2)

    Inherits from botorch Model (via virtual subclass registration) to satisfy
    type checks in acquisition function builders.
    """

    _num_outputs = 1

    def __init__(self, models: list[SingleTaskGP], weights: Tensor):
        super().__init__()
        self.models = ModuleList(models)
        self.weights = weights

    def posterior(
        self, X: Tensor, observation_noise: bool = False, **kwargs
    ) -> Posterior:
        """Compute the weighted ensemble posterior.

        Args:
            X: ``batch_shape x q x d``-dim tensor of candidate points.
            observation_noise: Ignored (kept for interface compatibility).
            **kwargs: Ignored.

        Returns:
            A GPyTorchPosterior representing the ensemble prediction.
        """
        from botorch.posteriors import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal
        from linear_operator.operators import PsdSumLinearOperator

        weighted_means = []
        weighted_covars = []

        # Filter models with zero weights
        non_zero_mask = self.weights**2 > 0
        non_zero_indices = non_zero_mask.nonzero(as_tuple=True)[0]
        non_zero_weights = self.weights[non_zero_indices]
        # Re-normalize
        non_zero_weights = non_zero_weights / non_zero_weights.sum()

        for i, idx in enumerate(non_zero_indices):
            model = self.models[idx]
            with torch.no_grad():
                post = model.posterior(X)
            weight = non_zero_weights[i]
            weighted_means.append(weight * post.mean.squeeze(-1))
            weighted_covars.append(post.mvn.lazy_covariance_matrix * weight**2)

        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLinearOperator(*weighted_covars)

        mvn = MultivariateNormal(mean_x, covar_x)
        return GPyTorchPosterior(mvn)

    @property
    def num_outputs(self) -> int:
        """Number of outputs."""
        return self._num_outputs


# Register RGPEModel as a virtual subclass of botorch Model
# so that isinstance(rgpe_model, Model) returns True
def _register_rgpe_model():
    from botorch.models.model import Model

    Model.register(RGPEModel)


_register_rgpe_model()


# --- BayBE Surrogate ---


@define
class RGPESurrogate(Surrogate):
    """RGPE surrogate for transfer learning via ranked model ensemble.

    Trains independent GPs on each source task and on the target task,
    then combines them using ranking-based weights. Each sub-GP uses
    FilterFeatures to strip the task column from input, making the
    ensemble's input interface identical to ICM-kernel based models.
    """

    supports_transfer_learning: ClassVar[bool] = True
    supports_multi_output: ClassVar[bool] = False

    num_posterior_samples: int = field(default=256, validator=[instance_of(int), ge(1)])
    """Number of MC samples for computing rank weights."""

    source_gp_kernel_factory: KernelFactoryProtocol | None = field(default=None)
    """Optional kernel factory for source/target GPs. None uses BoTorch defaults."""

    _rgpe_model: RGPEModel | None = field(init=False, default=None, eq=False)

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        # RGPE handles its own scaling inside each sub-GP
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # RGPE handles its own scaling inside each sub-GP
        return None

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the RGPE ensemble: independent source GPs + target GP + rank weights."""
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import (
            ChainedInputTransform,
            FilterFeatures,
            Normalize,
        )
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        assert self._searchspace is not None

        task_col_idx = self._searchspace.task_idx
        target_task_idxs = self._searchspace.target_task_idxs
        n_features = train_x.shape[-1]

        # Feature indices: all columns except the task column
        feature_indices = torch.tensor(
            [i for i in range(n_features) if i != task_col_idx], dtype=torch.long
        )

        # Split data by task
        target_mask = torch.zeros(len(train_x), dtype=torch.bool)
        for tidx in target_task_idxs:
            target_mask |= train_x[:, task_col_idx] == tidx

        target_x = train_x[target_mask]
        target_y = train_y[target_mask]

        # Get unique source task values
        all_task_vals = train_x[:, task_col_idx].unique()
        target_task_tensor = torch.tensor(
            target_task_idxs, dtype=all_task_vals.dtype, device=all_task_vals.device
        )
        source_task_vals = all_task_vals[~torch.isin(all_task_vals, target_task_tensor)]

        # Fit source GPs
        base_models: list[SingleTaskGP] = []
        for task_val in source_task_vals:
            mask = train_x[:, task_col_idx] == task_val
            source_x = train_x[mask]
            source_y = train_y[mask]
            model = self._fit_single_gp(
                source_x,
                source_y,
                feature_indices,
                fit_gpytorch_mll,
                SingleTaskGP,
                ChainedInputTransform,
                FilterFeatures,
                Normalize,
                Standardize,
                ExactMarginalLogLikelihood,
            )
            base_models.append(model)

        # Fit target GP (only if we have target data)
        if len(target_x) >= 2:
            target_model = self._fit_single_gp(
                target_x,
                target_y,
                feature_indices,
                fit_gpytorch_mll,
                SingleTaskGP,
                ChainedInputTransform,
                FilterFeatures,
                Normalize,
                Standardize,
                ExactMarginalLogLikelihood,
            )
        else:
            target_model = None

        # Compute rank weights
        # LOOCV requires at least 3 target points to be meaningful
        if target_model is not None and len(base_models) > 0 and len(target_x) >= 3:
            try:
                weights = _compute_rank_weights(
                    target_x,
                    target_y,
                    base_models,
                    target_model,
                    self.num_posterior_samples,
                )
            except (RuntimeError, ValueError):
                # Fall back to target-only if weight computation fails
                n_models = len(base_models) + 1
                weights = torch.zeros(n_models, dtype=train_x.dtype)
                weights[-1] = 1.0
        elif target_model is not None:
            # Have target model but insufficient data for LOOCV: use target only
            n_models = len(base_models) + 1
            weights = torch.zeros(n_models, dtype=train_x.dtype)
            weights[-1] = 1.0
        else:
            # No target model: equal weights on source models
            n_models = len(base_models)
            if n_models > 0:
                weights = torch.ones(n_models, dtype=train_x.dtype) / n_models
            else:
                raise ValueError(
                    "RGPE requires at least some training data "
                    "(either source or target)."
                )

        # Assemble RGPE
        all_models = base_models + ([target_model] if target_model is not None else [])
        self._rgpe_model = RGPEModel(all_models, weights)

    def _fit_single_gp(
        self,
        train_x: Tensor,
        train_y: Tensor,
        feature_indices: Tensor,
        fit_gpytorch_mll,
        SingleTaskGP,
        ChainedInputTransform,
        FilterFeatures,
        Normalize,
        Standardize,
        ExactMarginalLogLikelihood,
    ) -> SingleTaskGP:
        """Fit a single GP with FilterFeatures to strip the task column.

        The GP accepts full-dimensional input (including task column) but
        only operates on the feature columns via FilterFeatures.
        """
        n_features = len(feature_indices)

        input_transform = ChainedInputTransform(
            filter=FilterFeatures(feature_indices=feature_indices),
            normalize=Normalize(d=n_features),
        )

        kwargs: dict = {
            "train_X": train_x,
            "train_Y": train_y,
            "input_transform": input_transform,
            "outcome_transform": Standardize(1),
        }

        # Optionally use custom kernel
        if self.source_gp_kernel_factory is not None:
            kernel = self.source_gp_kernel_factory(self._searchspace, train_x, train_y)
            if hasattr(kernel, "to_gpytorch"):
                kernel = kernel.to_gpytorch(searchspace=self._searchspace)
            kwargs["covar_module"] = kernel

        model = SingleTaskGP(**kwargs)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    @override
    def to_botorch(self):
        """Return the RGPE model for use with BoTorch acquisition functions.

        The returned model accepts full-dimensional input (including task column)
        because each sub-GP uses FilterFeatures internally.
        """
        return self._rgpe_model

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute the posterior via the RGPE ensemble."""
        return self._rgpe_model.posterior(candidates_comp_scaled)
