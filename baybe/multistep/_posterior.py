"""Lazy, reparameterized posteriors over complete function-network paths."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from copy import copy
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from attrs import define, field
from botorch.posteriors.posterior import Posterior
from botorch.sampling.get_sampler import GetSampler
from typing_extensions import override

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    import torch
    from botorch.sampling.normal import NormalMCSampler
    from torch import Tensor

    from .model import MultiStepModel


def _freeze_outputs(value: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Copy fixed-output bindings without copying their differentiable tensors."""
    return MappingProxyType(dict(value))


def _snapshot_model(model: MultiStepModel) -> MultiStepModel:
    """Retain the published fit context without copying or evaluating node models."""
    snapshot = copy(model)
    snapshot._modules = model._modules.copy()
    snapshot._node_models = model._node_models.copy()
    snapshot._node_surrogates = model._node_surrogates.copy()
    snapshot._feasibility_models = model._feasibility_models.copy()
    return snapshot


@define(frozen=True, slots=False, eq=False)
class _NetworkPosterior(Posterior):
    """Propagate sampler-provided randomness through the DAG on demand.

    Each base sample contains one independent normal coordinate per node and
    q-point. Fixed nodes retain their coordinate slots so fixing an output does
    not change the randomness supplied to the other nodes. Candidate batch axes
    are collapsed by native BoTorch normal samplers, not by the network model.
    """

    model: MultiStepModel = field(converter=_snapshot_model)
    """The fitted network supplying local models and input bookkeeping."""

    X: Tensor = field()
    """Candidate configurations in computational representation."""

    fixed_outputs: Mapping[str, Tensor] = field(factory=dict, converter=_freeze_outputs)
    """Observed outputs replacing draws at the corresponding nodes."""

    def __attrs_post_init__(self) -> None:
        """Validate the network context without evaluating any node posterior.

        Raises:
            IncompatibilityError: If candidate or fixed-output values are non-finite,
                or a fixed output refers to an unknown node.
            ModelNotTrainedError: If any node model is missing.
        """
        import torch

        self.model._validate_fitted()
        self.model.unfixed_nodes(self.fixed_outputs)
        if not torch.isfinite(self.X).all():
            raise IncompatibilityError("Network candidates must be finite.")
        if any(
            not torch.isfinite(value).all() for value in self.fixed_outputs.values()
        ):
            raise IncompatibilityError("Fixed node outputs must be finite.")

    @property
    @override
    def base_sample_shape(self) -> torch.Size:
        """The candidate batch/q axes followed by one coordinate per DAG node."""
        import torch

        return self.X.shape[:-1] + torch.Size([self.model.num_nodes])

    @property
    @override
    def batch_range(self) -> tuple[int, int]:
        """The comparison axes, excluding q and node coordinates."""
        return (0, -2)

    @property
    def batch_shape(self) -> torch.Size:
        """The candidate batch shape, including any supplied fantasy axes."""
        return self.X.shape[:-2]

    @property
    @override
    def device(self) -> torch.device:
        """The device of the candidate configurations."""
        return self.X.device

    @property
    @override
    def dtype(self) -> torch.dtype:
        """The dtype of the candidate configurations."""
        return self.X.dtype

    @property
    def mean(self) -> Tensor:
        """The deterministic Monte-Carlo estimate of the final-output means."""
        return self._moment_samples.mean(dim=0)

    @cached_property
    def _moment_samples(self) -> Tensor:
        """The shared network draws used to estimate posterior moments."""
        import torch

        return self.draw_outcomes(
            torch.Size([self.model.num_network_samples]),
            sampling_seed=self.model.sampling_seed,
        )[0]

    @property
    def variance(self) -> Tensor:
        """The sample variance of the final outputs, or zero for a single draw."""
        return self._moment_samples.var(
            dim=0, unbiased=self._moment_samples.shape[0] > 1
        )

    def _draw_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: Tensor,
        *,
        include_feasibility: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Propagate normal base samples and optionally compute success weights.

        Args:
            sample_shape: The independent network-sample axes.
            base_samples: Normal coordinates matching the sample and q/node axes;
                singleton comparison-batch axes may be broadcast.
            include_feasibility: Whether to accumulate local success probabilities.

        Returns:
            Complete network outputs and the corresponding success weights.

        Raises:
            IncompatibilityError: If the base samples have incompatible shapes or
                non-finite values, or local node sampling is incompatible.
        """
        import torch

        from ._sampling import _propagate_network_samples

        expected_shape = sample_shape + self.base_sample_shape
        if base_samples.shape[: len(sample_shape)] != sample_shape:
            raise IncompatibilityError(
                f"Sample shape {tuple(sample_shape)} does not match base samples "
                f"with shape {tuple(base_samples.shape)}."
            )
        if (
            base_samples.ndim != len(expected_shape)
            or base_samples.shape[-2:] != expected_shape[-2:]
        ):
            raise IncompatibilityError(
                f"Base samples must match the q/node dimensions of "
                f"{tuple(expected_shape)}, got {tuple(base_samples.shape)}. "
                "Only comparison batch axes may be broadcast."
            )
        try:
            expanded = base_samples.to(self.X).expand(expected_shape)
        except RuntimeError as ex:
            raise IncompatibilityError(
                f"Base samples with shape {tuple(base_samples.shape)} cannot be "
                f"broadcast to {tuple(expected_shape)}."
            ) from ex
        if not torch.isfinite(expanded).all():
            raise IncompatibilityError("Network base samples must be finite.")
        n_paths = sample_shape.numel()
        samples, weights = _propagate_network_samples(
            self.model,
            self.X,
            expanded.reshape(n_paths, *self.base_sample_shape),
            self.fixed_outputs,
            include_feasibility=include_feasibility,
        )
        return (
            samples.reshape(self._extended_shape(sample_shape)),
            weights.reshape(sample_shape + self.X.shape[:-1]),
        )

    @override
    def _extended_shape(self, sample_shape: torch.Size | None = None) -> torch.Size:
        import torch

        shape = torch.Size() if sample_shape is None else sample_shape
        return shape + self.X.shape[:-1] + torch.Size([self.model.num_outputs])

    def draw_outcomes(
        self,
        sample_shape: torch.Size,
        *,
        sampling_seed: int | None = None,
        include_feasibility: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Draw complete network paths and optional conditional success weights.

        Args:
            sample_shape: Shape of the independent network draws.
            sampling_seed: Optional seed for the native normal sampler.
            include_feasibility: Whether to accumulate local success probabilities.

        Returns:
            Final-output draws and success weights with matching sample/batch/q axes.
        """
        sampler = _make_network_sampler(self, sample_shape, seed=sampling_seed)
        sampler._construct_base_samples(self)
        return self._draw_from_base_samples(
            sample_shape,
            cast("Tensor", sampler.base_samples),
            include_feasibility=include_feasibility,
        )

    @override
    def rsample(self, sample_shape: torch.Size | None = None) -> Tensor:
        import torch

        shape = torch.Size([1]) if sample_shape is None else sample_shape
        return self.draw_outcomes(shape)[0]

    @override
    def rsample_from_base_samples(
        self, sample_shape: torch.Size, base_samples: Tensor
    ) -> Tensor:
        return self._draw_from_base_samples(sample_shape, base_samples)[0]


@GetSampler.register(_NetworkPosterior)
def _make_network_sampler(
    posterior: _NetworkPosterior,
    sample_shape: torch.Size,
    *,
    seed: int | None = None,
) -> NormalMCSampler:
    """Create a native Sobol sampler over node coordinates, not over path axes.

    Args:
        posterior: The network posterior for sampler dispatch.
        sample_shape: The independent network-sample axes.
        seed: Optional Sobol scrambling seed.

    Returns:
        A native normal sampler over complete network paths.
    """
    from botorch.sampling.normal import SobolQMCNormalSampler

    return SobolQMCNormalSampler(sample_shape=sample_shape, seed=seed)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
