"""Propagate sampler-provided base samples through multi-step function networks."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    from botorch.models.model import Model
    from torch import Tensor

    from .model import MultiStepModel


def _derive_seed(base_seed: int, node_name: str, purpose: str) -> int:
    """Separate local observations from complete network-path randomness.

    Args:
        base_seed: The model- or policy-level seed.
        node_name: The node whose observations are sampled.
        purpose: The observation sampling domain.

    Returns:
        A stable unsigned 32-bit seed.
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


def _sample_node_posterior(
    model: Model, inputs: Tensor, base_samples: Tensor
) -> Tensor:
    """Draw one local output per path using the supplied normal coordinates.

    Args:
        model: The fitted single-output node model.
        inputs: Node inputs with shape ``(n_paths, *batch_shape, q, d)``.
        base_samples: Normal coordinates with shape ``(n_paths, batch..., q)``.

    Returns:
        Node samples with shape ``(n_paths, *batch_shape, q, num_outputs)``.

    Raises:
        IncompatibilityError: If the posterior type or the resulting shape is
            unsupported.
    """
    import torch
    from botorch.posteriors.ensemble import EnsemblePosterior
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from botorch.posteriors.transformed import TransformedPosterior

    n_paths = inputs.shape[0]
    batch_shape = tuple(inputs.shape[1:-2])
    q = inputs.shape[-2]
    posterior = model.posterior(inputs, observation_noise=False)
    original = posterior
    while isinstance(original, TransformedPosterior):
        original = original._posterior
    if not isinstance(original, (GPyTorchPosterior, EnsemblePosterior)):
        raise IncompatibilityError(
            f"Network node model '{type(model).__name__}' must provide a Gaussian "
            f"or ensemble posterior, got '{type(original).__name__}'."
        )
    posterior_batch = tuple(original.batch_shape)
    if posterior_batch != (n_paths, *batch_shape):
        raise IncompatibilityError(
            f"The posterior of node model '{type(model).__name__}' has batch shape "
            f"{posterior_batch}, expected {(n_paths, *batch_shape)}."
        )

    if isinstance(original, EnsemblePosterior):
        # Select one member jointly for all q-points. Only the member index,
        # not the member prediction, is shared across comparison batches.
        weights = original._weights
        weights = original.weights if weights is None else weights
        mass = weights.sum(dim=-1, keepdim=True)
        if (
            not torch.isfinite(weights).all()
            or (weights < 0).any()
            or not torch.isfinite(mass).all()
            or (mass <= 0).any()
        ):
            raise IncompatibilityError(
                "Ensemble weights must be finite, nonnegative, and have positive mass."
            )
        probabilities = torch.special.ndtr(base_samples[..., 0]).unsqueeze(-1)
        upper = torch.nextafter(
            torch.ones_like(probabilities), torch.zeros_like(probabilities)
        )
        probabilities = torch.minimum(probabilities, upper)
        cumulative = weights.cumsum(dim=-1)
        cumulative = cumulative / cumulative[..., -1:]
        coordinates = (probabilities >= cumulative).sum(dim=-1)
    else:
        if original.base_sample_shape != inputs.shape[:-1]:
            raise IncompatibilityError(
                f"The Gaussian posterior of node model '{type(model).__name__}' "
                "must use one normal coordinate per q-point."
            )
        coordinates = base_samples
    samples = posterior.rsample_from_base_samples(torch.Size(), coordinates)

    expected_shape = (*posterior_batch, q, model.num_outputs)
    if tuple(samples.shape) != expected_shape:
        raise IncompatibilityError(
            f"Sampling the fitted model of type '{type(model).__name__}' produced "
            f"shape {tuple(samples.shape)}, expected {expected_shape}."
        )
    return samples


def _draw_network_samples(
    model: MultiStepModel,
    X: Tensor,
    n_paths: int,
    fixed_outputs: Mapping[str, Tensor] | None,
    *,
    sampling_seed: int,
    include_feasibility: bool,
) -> tuple[Tensor, Tensor]:
    """Forward-sample the DAG with one native sampler over complete network paths.

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
    from ._posterior import _NetworkPosterior

    posterior = _NetworkPosterior(model, X, fixed_outputs or {})
    return posterior.draw_outcomes(
        torch.Size([n_paths]),
        sampling_seed=sampling_seed,
        include_feasibility=include_feasibility,
    )


def _propagate_network_samples(
    model: MultiStepModel,
    X: Tensor,
    base_samples: Tensor,
    fixed_outputs: Mapping[str, Tensor] | None,
    *,
    include_feasibility: bool,
) -> tuple[Tensor, Tensor]:
    """Forward-propagate one complete DAG path per supplied base sample.

    Args:
        model: The fitted network context.
        X: Global candidate configurations in computational representation.
        base_samples: Normal coordinates with shape ``(n_paths, batch..., q, n_nodes)``.
        fixed_outputs: Observed outputs replacing the respective local draws.
        include_feasibility: Whether to accumulate conditional success probabilities.

    Returns:
        Final-output draws and success weights with aligned path/batch/q axes.

    Raises:
        IncompatibilityError: If fixed outputs, local features, or node posteriors
            are incompatible with the requested network paths.
    """
    import torch

    n_paths = base_samples.shape[0]
    batch_shape = tuple(X.shape[:-2])
    q = X.shape[-2]
    samples: dict[str, Tensor] = _prepare_fixed_outputs(
        X, fixed_outputs, n_paths, batch_shape
    )
    success_weights = X.new_ones((n_paths, *batch_shape, q))
    full_index = model._full_comp_rep_index

    for node_index, node_name in enumerate(model._node_names):
        if node_name in samples:
            continue
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
            base_samples[..., node_index],
        )

    return (
        torch.cat([samples[name] for name in model.dag.final_nodes], dim=-1),
        success_weights,
    )
