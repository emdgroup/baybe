"""Generality model wrapper for context-aggregated Bayesian optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

if TYPE_CHECKING:
    from baybe.aggregation.base import AggregationFunction


class _GeneralityPosterior(Posterior):
    """Posterior that aggregates base GP posterior across context values."""

    def __init__(
        self,
        base_posterior: Posterior,
        q: int,
        r: int,
        m: int,
        aggregation: AggregationFunction,
        target_transform: torch.nn.Module,
        n_variance_samples: int = 64,
    ) -> None:
        super().__init__()
        self._base = base_posterior
        self._q = q
        self._r = r
        self._m = m
        self._aggregation = aggregation
        self._target_transform = target_transform
        self._n_variance_samples = n_variance_samples

    @override
    @property
    def device(self) -> torch.device:
        """The device of the posterior."""
        return self._base.device

    @override
    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the posterior."""
        return self._base.dtype

    @property
    def batch_shape(self) -> torch.Size:
        """The batch shape of the posterior."""
        return self._base.batch_shape  # type: ignore[attr-defined]

    @override
    @property
    def base_sample_shape(self) -> torch.Size:
        """The base sample shape."""
        return self._base.base_sample_shape

    @override
    @property
    def batch_range(self) -> tuple[int, int]:
        """The t-batch range."""
        return self._base.batch_range

    @override
    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        """Draw samples and aggregate over contexts."""
        if sample_shape is None:
            sample_shape = torch.Size()
        samples = self._base.rsample(sample_shape)
        return self._aggregate(samples)

    @override
    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
    ) -> Tensor:
        """Draw samples from base samples and aggregate over contexts."""
        samples = self._base.rsample_from_base_samples(sample_shape, base_samples)
        return self._aggregate(samples)

    @property
    def mean(self) -> Tensor:
        """Aggregated posterior mean."""
        mean = self._base.mean  # type: ignore[attr-defined]
        return self._aggregate_mean(mean)

    @property
    def variance(self) -> Tensor:
        """Aggregated posterior variance via MC samples."""
        samples = self.rsample(torch.Size([self._n_variance_samples]))
        return samples.var(dim=0)

    def _aggregate(self, samples: Tensor) -> Tensor:
        """Aggregate samples over context dimension."""
        samples = samples.view(*samples.shape[:-2], self._q, self._r, self._m)
        samples = self._target_transform(samples, None)
        samples = samples.transpose(-1, -2)  # (..., q, m, r)
        return self._aggregation.forward(samples)  # (..., q, m)

    def _aggregate_mean(self, mean: Tensor) -> Tensor:
        """Aggregate mean tensor."""
        mean = mean.view(*mean.shape[:-2], self._q, self._r, self._m)
        mean = self._target_transform(mean, None)
        mean = mean.transpose(-1, -2)  # (..., q, m, r)
        return self._aggregation.forward(mean)  # (..., q, m)


def _register_generality_sampler() -> None:
    """Register a sampler for _GeneralityPosterior with BoTorch's dispatcher."""
    from botorch.sampling.get_sampler import GetSampler
    from botorch.sampling.normal import SobolQMCNormalSampler

    @GetSampler.register(_GeneralityPosterior)
    def _get_sampler_generality(
        posterior: _GeneralityPosterior,
        sample_shape: torch.Size,
        seed: int | None = None,
    ) -> SobolQMCNormalSampler:
        # Delegate to base posterior's sampler, but wrap in our aggregation.
        # Since _GeneralityPosterior.rsample() handles aggregation internally,
        # we just need the sampler to call rsample() correctly.
        # Use the base posterior's event shape for base_samples.
        return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


_register_generality_sampler()


class _GeneralityModel(Model):
    """BoTorch Model wrapper for generality-oriented optimization."""

    w_values_comp: Tensor

    def __init__(
        self,
        base_model: Model,
        w_values_comp: Tensor,
        x_col_indices: list[int],
        w_col_indices: list[int],
        aggregation: AggregationFunction,
        target_transform: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.register_buffer("w_values_comp", w_values_comp)
        self._x_col_indices = x_col_indices
        self._w_col_indices = w_col_indices
        self._aggregation = aggregation
        self._target_transform = target_transform
        self._r = w_values_comp.shape[0]
        self._m = base_model.num_outputs
        self._d_full = len(x_col_indices) + len(w_col_indices)

    @override
    @property
    def num_outputs(self) -> int:
        """Number of outputs after aggregation (one per target)."""
        return self._m

    @override
    def posterior(  # type: ignore[override]
        self,
        X: Tensor,
        output_indices: list[int] | None = None,
        observation_noise: bool = False,
        posterior_transform: Any = None,
    ) -> _GeneralityPosterior:
        """Compute the aggregated posterior.

        Args:
            X: Input tensor in x-only space.
            output_indices: Ignored.
            observation_noise: Ignored.
            posterior_transform: Ignored.

        Returns:
            _GeneralityPosterior: Aggregated posterior over context values.
        """
        q = X.shape[-2]
        X_full = self._expand(X)
        base_posterior = self.base_model.posterior(X_full)
        return _GeneralityPosterior(
            base_posterior,
            q,
            self._r,
            self._m,
            self._aggregation,
            self._target_transform,
        )

    def _expand(self, X: Tensor) -> Tensor:
        """Pair each candidate with all context values for the base GP.

        Args:
            X: Tensor in x-subspace.

        Returns:
            Tensor in (x,w)-space, expanded over all contexts.
        """
        batch_shape = X.shape[:-2]
        q = X.shape[-2]

        x_expanded = X.unsqueeze(-2).expand(*batch_shape, q, self._r, -1)
        w_expanded = self.w_values_comp.expand(*batch_shape, q, self._r, -1)

        parts: list[Tensor] = []
        x_idx = 0
        w_idx = 0
        x_set = set(self._x_col_indices)
        for i in range(self._d_full):
            if i in x_set:
                parts.append(x_expanded[..., x_idx : x_idx + 1])
                x_idx += 1
            else:
                parts.append(w_expanded[..., w_idx : w_idx + 1])
                w_idx += 1

        X_full = torch.cat(parts, dim=-1)  # (..., q, r, d_full)
        return X_full.reshape(*batch_shape, q * self._r, self._d_full)
