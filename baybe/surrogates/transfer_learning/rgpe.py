"""Rank-weighted GP ensemble (RGPE) surrogate for transfer learning.

Implements the rank-weighted Gaussian process ensemble (RGPE) of Feurer, Letham and
Bakshy (ICML 2018 AutoML Workshop, https://arxiv.org/abs/1802.02219), adapted to BayBE:
one single-task GP is fitted per task on the task-stripped data and the ensemble
posterior is a rank-weighted sum of the individual posteriors.

NOTE: This module is a skeleton. Bodies raising :class:`NotImplementedError` mark the
numerical core that still needs to be filled in.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from attrs.validators import ge, instance_of
from typing_extensions import override

from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    import pandas as pd
    from botorch.models.gpytorch import GPyTorchModel
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.parameters.base import Parameter
    from baybe.searchspace.core import SearchSpace
    from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate


def _roll_col(x: Tensor, shift: int) -> Tensor:
    """Rotate the columns of a tensor to the right by ``shift``.

    Used by the source-model branch of :func:`_ranking_loss` to enumerate all pairwise
    orderings of the target points without materializing the full Cartesian product.

    Args:
        x: The tensor whose last dimension is rotated.
        shift: The number of positions to rotate by.

    Returns:
        The column-rotated tensor.
    """
    raise NotImplementedError


def _draw_posterior_samples(model: GPyTorchModel, x: Tensor, n_samples: int) -> Tensor:
    """Draw Monte Carlo posterior samples of a model at the given points.

    Wraps the model's posterior in a quasi-Monte Carlo (Sobol) sampler. Used to sample
    each source model at the target points before scoring it with :func:`_ranking_loss`.

    Args:
        model: The fitted Gaussian process model to sample from.
        x: The points at which to evaluate the posterior (``n x d``).
        n_samples: The number of Monte Carlo samples to draw.

    Returns:
        An ``n_samples x n`` tensor of posterior samples.
    """
    raise NotImplementedError


def _ranking_loss(f_samps: Tensor, target_y: Tensor) -> Tensor:
    """Count discordant pairwise orderings between sampled predictions and targets.

    A 3-dimensional ``f_samps`` is interpreted as the target model's leave-one-out
    cross-validation samples (its diagonal holds the out-of-sample predictions); a
    2-dimensional ``f_samps`` as ordinary source-model samples.

    Args:
        f_samps: Either an ``n_samples x n x n`` (target LOOCV) or ``n_samples x n``
            (source) tensor of posterior samples.
        target_y: An ``n x 1`` tensor of observed target values.

    Returns:
        An ``n_samples`` tensor holding the ranking loss of each sample.
    """
    raise NotImplementedError


def _loocv_samples(
    train_x: Tensor, train_y: Tensor, target_model: GPyTorchModel, n_samples: int
) -> Tensor:
    """Draw leave-one-out cross-validation samples of the target model.

    Builds a single batched GP whose batch dimension indexes the left-out point (valid
    because every fold has ``n - 1`` points), reusing the target model's fitted
    hyperparameters, and draws a joint sample across all target points.

    Args:
        train_x: The target training inputs (``n x d``).
        train_y: The target training targets (``n x 1``).
        target_model: The fitted target Gaussian process model.
        n_samples: The number of Monte Carlo samples to draw.

    Returns:
        A ``n_samples x n x n`` tensor of samples (dim 1 = LOO models, dim 2 = points).
    """
    raise NotImplementedError


def _weights_from_losses(ranking_losses: Tensor) -> Tensor:
    """Turn per-model ranking losses into ensemble weights.

    For each Monte Carlo sample the model with the lowest ranking loss is credited; the
    weight of a model is the fraction of samples for which it is the best.

    Args:
        ranking_losses: A ``n_models x n_samples`` tensor of ranking losses, ordered as
            ``(sources..., target)`` along the first dimension.

    Returns:
        A ``n_models`` tensor of weights summing to one.
    """
    # argmin over models per sample, then bincount / n_samples.
    raise NotImplementedError


def _rank_weights(
    train_x: Tensor,
    train_y: Tensor,
    source_models: tuple[GPyTorchModel, ...],
    target_model: GPyTorchModel,
    n_samples: int,
) -> Tensor:
    """Compute the RGPE rank weights for the source models and the target model.

    Orchestrates the weight computation: scores each source model via
    :func:`_draw_posterior_samples` + :func:`_ranking_loss`, scores the target model
    via :func:`_loocv_samples` + :func:`_ranking_loss`, then combines the stacked
    losses with :func:`_weights_from_losses`.

    Args:
        train_x: The target training inputs (``n x d``).
        train_y: The target training targets (``n x 1``).
        source_models: The fitted source Gaussian process models.
        target_model: The fitted target Gaussian process model.
        n_samples: The number of Monte Carlo samples used to estimate the weights.

    Returns:
        A tensor of length ``len(source_models) + 1`` with the weight of each model,
        ordered as ``(sources..., target)``.
    """
    raise NotImplementedError


def _make_ensemble_model(
    models: list[GPyTorchModel], weights: Tensor, task_idx: int
) -> GPyTorchModel:
    """Wrap the fitted per-task GPs into a single botorch model that blends them.

    Holds the per-task GPs in a :class:`~botorch.models.ModelListGP` (as independent
    outputs) and collapses them into the RGPE posterior with a
    :class:`~botorch.acquisition.objective.ScalarizedPosteriorTransform`. Because the
    outputs are independent, scalarizing with ``weights`` yields ``mean = Σ wᵢ·μᵢ`` and
    ``cov = Σ wᵢ²·Σᵢ`` exactly. Subclassing ``ModelListGP`` keeps ``fantasize`` /
    ``condition_on_observations`` working (they are delegated to the sub-models).

    Defined as a factory so the botorch imports stay lazy.

    Args:
        models: The fitted per-task botorch models, ordered ``(sources..., target)``.
        weights: The ensemble weights aligned with ``models``.
        task_idx: The computational-representation column of the task parameter, which
            is stripped from candidates before hitting the task-free sub-models.

    Returns:
        A botorch model whose posterior is the rank-weighted sum of the per-task GPs.
    """
    import torch
    from botorch.acquisition.objective import ScalarizedPosteriorTransform
    from botorch.models import ModelListGP

    class _RGPEnsembleModel(ModelListGP):
        """The RGPE ensemble: a ``ModelListGP`` collapsed to the rank-weighted blend."""

        # NOTE: The underlying list is multi-output, but the scalarized posterior this
        # model exposes is single-output.

        def __init__(self) -> None:
            super().__init__(*models)
            self._transform = ScalarizedPosteriorTransform(weights=weights)
            self._task_idx = task_idx

        def _strip_task(self, x: Tensor) -> Tensor:
            """Drop the task column so the task-free sub-models can be evaluated.

            Args:
                x: Candidates in computational representation (task column included).

            Returns:
                The candidates without the task column.
            """
            return torch.cat(
                [x[..., : self._task_idx], x[..., self._task_idx + 1 :]], dim=-1
            )

        @override
        def posterior(self, X: Tensor, *args, **kwargs) -> Posterior:
            """Return the rank-weighted blend of the per-task posteriors.

            Args:
                X: Candidates in computational representation (task column included).
                *args: Forwarded to :meth:`ModelListGP.posterior`.
                **kwargs: Forwarded to :meth:`ModelListGP.posterior`.

            Returns:
                The scalarized (rank-weighted) ensemble posterior.
            """
            posterior = super().posterior(self._strip_task(X), *args, **kwargs)
            return self._transform(posterior)

        @override
        def fantasize(self, *args, **kwargs):
            """Fantasize each sub-model, then re-apply the scalarization.

            ``ModelListGP.fantasize`` returns a plain ``ModelListGP``, so the result
            must be re-wrapped to preserve the rank-weighted blend.

            Args:
                *args: Forwarded to :meth:`ModelListGP.fantasize`.
                **kwargs: Forwarded to :meth:`ModelListGP.fantasize`.
            """
            raise NotImplementedError

    return _RGPEnsembleModel()


@define
class RGPETransferSurrogate(Surrogate):
    """A rank-weighted Gaussian process ensemble (RGPE) transfer learning surrogate.

    Fits one single-task Gaussian process per task on the task-stripped data and blends
    their posteriors as ``mean = Σ wᵢ·μᵢ`` and ``cov = Σ wᵢ²·Σᵢ``, where the weights are
    derived from a ranking loss (leave-one-out cross-validation for the target task),
    following Feurer, Letham and Bakshy (ICML 2018 AutoML Workshop).
    """

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    base_surrogate: GaussianProcessSurrogate = field()
    """The Gaussian process template cloned and fitted once per task."""

    num_mc_samples: int = field(default=256, validator=[instance_of(int), ge(1)])
    """The number of Monte Carlo samples used to estimate the ranking weights."""

    _models = field(init=False, default=None, eq=False, repr=False)
    """The fitted per-task GPs, ordered ``(sources..., target)``. Set during fitting."""

    _weights = field(init=False, default=None, eq=False, repr=False)
    """The ensemble weights, aligned with ``_models``. Set during fitting."""

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter, /
    ) -> type[InputTransform] | None:
        # Input scaling is delegated to the per-task GPs.
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        # Output scaling is delegated to the per-task GPs.
        return None

    def _task_free_searchspace(self) -> SearchSpace:
        """Return a task-free search space for fitting the per-task GPs.

        The reduced space must still expose ``transform``/``scaling_bounds`` (see
        :class:`~baybe.searchspace.core._ReducedSearchSpace`), which is why those
        attributes are allow-listed there.

        Returns:
            The search space without the task parameter.
        """
        assert self._searchspace is not None  # set during fitting
        task_parameter = self._searchspace._task_parameter
        assert task_parameter is not None  # ensured by the dispatch
        return self._searchspace._drop_parameters({task_parameter.name})

    def _split_by_task(self) -> tuple[tuple[pd.DataFrame, ...], pd.DataFrame]:
        """Split the training data into source and target measurements.

        Returns:
            A tuple ``(source_frames, target_frame)`` where the target frame holds the
            rows whose task label is in the task parameter's ``active_values``.
        """
        assert self._searchspace is not None  # set during fitting
        assert self._measurements is not None  # set during fitting
        task_parameter = self._searchspace._task_parameter
        assert task_parameter is not None  # ensured by the dispatch

        column = task_parameter.name
        target_labels = set(task_parameter.active_values)
        is_target = self._measurements[column].isin(target_labels)
        target_frame = self._measurements[is_target]
        source_frames = tuple(
            frame for _, frame in self._measurements[~is_target].groupby(column)
        )
        return source_frames, target_frame

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit one GP per task and compute the ensemble weights.

        Args:
            train_x: Unused; the per-task data is re-derived from the stored
                measurements. Present to satisfy the base class interface.
            train_y: Unused; see ``train_x``.
        """
        # Loop, not batch: tasks have different numbers of points, and a batched GP
        # needs an equal count per batch element. Cold start (<2 target points):
        # fall back to uniform weights.
        raise NotImplementedError

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the rank-weighted ensemble posterior on task-stripped candidates.

        Args:
            candidates_comp_scaled: Candidates in the computational representation of
                the full search space (task column included).

        Returns:
            The rank-weighted ensemble posterior.
        """
        # Strip the task column, evaluate the per-task GPs, and scalarize with the
        # weights to get mean = Σ wᵢ·μᵢ and cov = Σ wᵢ²·Σᵢ (see `_make_ensemble_model`).
        raise NotImplementedError

    @override
    def to_botorch(self) -> GPyTorchModel:
        """Expose the ensemble as one botorch model.

        Builds a :class:`~botorch.models.ModelListGP` of the per-task GPs whose
        posterior is scalarized into the rank-weighted RGPE blend. Going through
        ``ModelListGP`` (instead of a hand-written GP) keeps ``fantasize`` /
        ``condition_on_observations`` working, which the fantasy-based acquisition
        functions (qKG, qNIPV) rely on.

        Note:
            Extending ``AdapterModel`` to apply a ``posterior_transform`` would also
            work, but only unlocks the analytic acquisition functions; the fantasy-based
            ones still require a real GP as returned here.

        Returns:
            The ensemble botorch model.
        """
        assert self._models is not None  # set during fitting
        assert self._searchspace is not None  # set during fitting
        assert self._weights is not None  # set during fitting
        return _make_ensemble_model(
            models=[gp.to_botorch() for gp in self._models],
            weights=self._weights,
            task_idx=self._searchspace.task_idx,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
