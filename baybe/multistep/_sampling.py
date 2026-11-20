"""Native sampling helpers for multi-step function networks.

The helpers configure BoTorch samplers, align their base samples across
comparison batches, and propagate the draws through the DAG. They contain no
fitting, policy, or posterior mathematics.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    from botorch.models.model import Model
    from torch import Tensor

    from .model import MultiStepModel


def _derive_seed(base_seed: int, node_name: str, purpose: str) -> int:
    """Derive a stable node- and purpose-specific seed.

    Args:
        base_seed: The model- or policy-level base seed.
        node_name: The name of the node whose random draw is seeded.
        purpose: The domain of the random draw.

    Returns:
        An unsigned 32-bit integer seed.
    """
    payload = f"{base_seed}:{node_name}:{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _prepare_fixed_outputs(
    X: Tensor,
    fixed_outputs: Mapping[str, Tensor] | None,
    n_paths: int,
    batch_shape: tuple[int, ...],
) -> dict[str, Tensor]:
    """Broadcast observed node outputs to the network sample shape.

    Args:
        X: Candidate configurations supplying device and dtype.
        fixed_outputs: Observed node outputs keyed by node name.
        n_paths: The number of network paths.
        batch_shape: The candidate batch shape.

    Returns:
        The fixed outputs expanded to ``(n_paths, *batch_shape, q, 1)``.

    Raises:
        IncompatibilityError: If an output cannot be broadcast to that shape.
    """
    target_shape = (n_paths, *batch_shape, X.shape[-2], 1)
    prepared: dict[str, Tensor] = {}
    for name, value in (fixed_outputs or {}).items():
        try:
            prepared[name] = value.to(X).expand(target_shape)
        except RuntimeError as ex:
            raise IncompatibilityError(
                f"Fixed output for node '{name}' with shape {tuple(value.shape)} "
                f"cannot be broadcast to {target_shape}."
            ) from ex
    return prepared


def _sample_node_posterior(model: Model, inputs: Tensor, seed: int) -> Tensor:
    """Draw one posterior sample per network path with native BoTorch samplers.

    Args:
        model: The fitted single-output node model.
        inputs: Node inputs with shape ``(n_paths, *batch_shape, q, d)``.
        seed: Seed for the native sampler.

    Returns:
        Node samples with shape ``(n_paths, *batch_shape, q, num_outputs)``.

    Raises:
        IncompatibilityError: If the posterior type or the resulting shape is
            unsupported.
    """
    import torch
    from botorch.posteriors.ensemble import EnsemblePosterior
    from botorch.posteriors.torch import TorchPosterior
    from botorch.sampling import get_sampler

    n_paths = inputs.shape[0]
    batch_shape = tuple(inputs.shape[1:-2])
    q = inputs.shape[-2]
    posterior = model.posterior(inputs, observation_noise=False)
    # Concrete posteriors expose this; the base posterior protocol does not.
    posterior_batch = tuple(posterior.batch_shape)  # type: ignore[attr-defined]
    if posterior_batch != (n_paths, *batch_shape):
        raise IncompatibilityError(
            f"The posterior of node model '{type(model).__name__}' has batch shape "
            f"{posterior_batch}, expected {(n_paths, *batch_shape)}."
        )

    sample_shape = torch.Size([1])
    if isinstance(posterior, EnsemblePosterior):
        # The native index sampler ignores batch-range overrides. Draw one
        # member per path on a path-only anchor, then broadcast that choice
        # over the comparison batches.
        n_comparison_batches = posterior.values.ndim - 4
        values = posterior.values[(slice(None),) + (0,) * n_comparison_batches]
        weights = posterior.weights
        if weights.ndim > 1:
            weights = weights[(slice(None),) + (0,) * n_comparison_batches]
        anchor = EnsemblePosterior(values, weights=weights)
        sampler = get_sampler(
            cast(TorchPosterior, anchor), torch.Size([n_paths]), seed=seed
        )
        path_samples = sampler(anchor)
        samples = path_samples.reshape(
            1, n_paths, *([1] * len(batch_shape)), *path_samples.shape[1:]
        ).expand(1, n_paths, *batch_shape, *path_samples.shape[1:])
    else:
        sampler = get_sampler(cast(TorchPosterior, posterior), sample_shape, seed=seed)
        try:
            _, batch_end = posterior.batch_range
        except (AttributeError, NotImplementedError):
            pass
        else:
            # Keep the path axis independent and collapse all comparison
            # batch axes, including for model-provided custom samplers.
            setattr(sampler, "batch_range_override", (1, batch_end))
        samples = sampler(posterior)

    expected_shape = (1, *posterior_batch, q, model.num_outputs)
    if tuple(samples.shape) != expected_shape:
        raise IncompatibilityError(
            f"Sampling the fitted model of type '{type(model).__name__}' produced "
            f"shape {tuple(samples.shape)}, expected {expected_shape}."
        )
    return samples.squeeze(0)


def _draw_network_samples(
    model: MultiStepModel,
    X: Tensor,
    n_paths: int,
    fixed_outputs: Mapping[str, Tensor] | None,
    *,
    sampling_seed: int,
    include_feasibility: bool,
) -> tuple[Tensor, Tensor]:
    """Forward-sample the DAG with native local samplers.

    Args:
        model: The fitted network model.
        X: Validated candidates with shape ``(batch..., q, d)``.
        n_paths: The number of network paths.
        fixed_outputs: Optional observed node outputs.
        sampling_seed: The base seed for all node draws.
        include_feasibility: Whether to accumulate network success weights.

    Returns:
        Output samples with shape ``(n_paths, *batch_shape, q, num_outputs)`` and
        success weights with shape ``(n_paths, *batch_shape, q)``.

    Raises:
        ValueError: If ``n_paths`` or ``sampling_seed`` is invalid.
        ModelNotTrainedError: If a node model is missing.
        IncompatibilityError: If node features, fixed outputs, or fitted model
            batch shapes are incompatible.
    """
    import torch

    if not isinstance(n_paths, int) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError(f"'n_paths' must be a positive integer, got {n_paths!r}.")
    if (
        not isinstance(sampling_seed, int)
        or isinstance(sampling_seed, bool)
        or sampling_seed < 0
    ):
        raise ValueError(
            f"'sampling_seed' must be a nonnegative integer, got {sampling_seed!r}."
        )
    model._validate_fitted()

    batch_shape = tuple(X.shape[:-2])
    q = X.shape[-2]
    samples: dict[str, Tensor] = _prepare_fixed_outputs(
        X, fixed_outputs, n_paths, batch_shape
    )
    success_weights = X.new_ones((n_paths, *batch_shape, q))
    full_index = model._full_comp_rep_index

    for node_name in model.unfixed_nodes(fixed_outputs):
        node = model.dag.node_by_name[node_name]
        features: dict[str, Tensor] = {}
        for column in node._own_comp_rep_columns:
            try:
                value = X[..., full_index[column]]
            except KeyError as ex:
                raise IncompatibilityError(
                    f"The computational-representation column '{column}' of node "
                    f"'{node_name}' is missing from the DAG-wide search space."
                ) from ex
            features[column] = value.expand(n_paths, *batch_shape, q).unsqueeze(-1)
        for parent in node.parents:
            features[parent.output] = samples[parent.name].to(X)

        columns = model.get_node_model_columns(node_name)
        missing = [column for column in columns if column not in features]
        if missing:
            raise IncompatibilityError(
                f"The node-model features of node '{node_name}' are incomplete: "
                f"{missing}."
            )
        node_x = torch.cat([features[column] for column in columns], dim=-1)

        if include_feasibility and node.feasibility_conditions:
            probabilities = model._feasibility_models[
                node_name
            ].posterior_probabilities(node_x)
            success_weights = success_weights * probabilities.reshape(
                success_weights.shape
            )

        samples[node_name] = _sample_node_posterior(
            model._node_models[node_name],
            node_x,
            _derive_seed(sampling_seed, node_name, "network-output"),
        )

    return (
        torch.cat([samples[name] for name in model.dag.final_nodes], dim=-1),
        success_weights,
    )
