"""The pKGFN gain: one finite reference set, one fantasy reduction.

The gain is ``max(0, mean_f(max_r E[U] - baseline))``:

- ``r`` ranges over the finite global reference configurations,
- ``f`` ranges over the fantasy observations of the candidate action,
- every branch baseline is the best utility over the same reference set,
- the clamp happens once, after the fantasy mean, and is never repeated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

from attrs import define

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    from torch import Tensor

    from ..dag import MultiStepDAG
    from ..data import _PendingAction
    from ..model import MultiStepModel
    from ..node import MultiStepNode


GainScale = Literal["linear", "log"]
"""The representation used by an acquisition function's raw gain."""


@define(frozen=True, slots=False)
class _GainContext:
    """Prepared finite pKGFN context for one recommendation."""

    evaluate: Callable[[MultiStepNode, Tensor], Tensor]
    """Evaluate a node and local inputs with shape '(batch..., 1, d_node)'."""

    num_branches: int
    """The number of pending branches (exactly 1 without pending actions)."""

    num_references: int
    """The number of global reference points in use."""


def _validate_gain_models(model: MultiStepModel) -> None:
    """Validate the fitted-model capabilities required by p-KGFN.

    Args:
        model: The fitted network model.

    Raises:
        IncompatibilityError: If a node model is batched or cannot be conditioned.
    """
    import torch

    for name in model.dag.topological_order:
        node_model = model.get_node_model(name)
        if node_model.batch_shape != torch.Size() or not callable(
            getattr(node_model, "condition_on_observations", None)
        ):
            raise IncompatibilityError(
                f"p-KGFN requires an unbatched model supporting fantasy "
                f"conditioning for node '{name}', but found "
                f"'{type(node_model).__name__}'."
            )


def compute_terminal_means_from_network_samples(
    dag: MultiStepDAG,
    samples: Tensor,
) -> Tensor:
    """Compute terminal utility means from raw network samples.

    Args:
        dag: The DAG defining the terminal objective.
        samples: Raw samples with shape
            ``(n_paths, batch..., q, num_outputs)``.

    Returns:
        Mean transformed utility for each candidate configuration.

    Raises:
        IncompatibilityError: If the sample shape is unsupported, a sample or
            utility is non-finite, or the objective has incompatible outputs.
    """
    import torch

    if (
        samples.shape[-2:] != (1, len(dag.graph_output_names))
        or not torch.isfinite(samples).all()
    ):
        raise IncompatibilityError(
            "Terminal samples must be finite, have q=1, and match the graph "
            "outputs."
        )

    objective_samples = samples.index_select(
        -1,
        torch.tensor(
            dag.objective_output_indices,
            dtype=torch.long,
            device=samples.device,
        ),
    )
    # Network outputs are raw node measurements. In particular, a
    # ``DesirabilityObjective`` configured with ``as_pre_transformation``
    # exposes a one-output surrogate objective from ``to_botorch()``; using
    # that here would select only one of the raw target columns. Apply the
    # objective's end-to-end target transformations and scalarization instead.
    objective_function = dag.objective._full_transformation
    objective_function.to(device=samples.device, dtype=samples.dtype)
    for component in getattr(objective_function, "objectives", ()):
        component.to(device=samples.device, dtype=samples.dtype)
    utilities = objective_function(objective_samples, None)
    if utilities.shape != samples.shape[:-1] or not torch.isfinite(utilities).all():
        raise IncompatibilityError(
            "The objective must produce one finite utility per terminal sample."
        )

    return utilities.squeeze(-1).mean(dim=0)


def compute_pkgfn_gain(
    fantasy_terminal_means: Tensor,
    baseline: Tensor,
) -> Tensor:
    """Reduce fantasy terminal means to an unweighted p-KGFN gain estimate.

    Args:
        fantasy_terminal_means: Network-averaged utilities with shape
            ``(num_fantasies, num_terminal_candidates, batch...)``.
        baseline: The baseline utility of the current data. A scalar applies
            to every fantasy; a tensor provides one baseline per fantasy and
            must broadcast to the best-terminal means.

    Returns:
        One finite nonnegative gain estimate per candidate batch.

    Raises:
        IncompatibilityError: If the input shape, baseline, or gain is invalid
            or non-finite.
    """
    import torch

    if (
        fantasy_terminal_means.dim() < 2
        or min(fantasy_terminal_means.shape[:2]) < 1
    ):
        raise IncompatibilityError(
            "Expected nonempty (fantasies, terminal candidates, batch...) "
            "means, got "
            f"{tuple(fantasy_terminal_means.shape)}."
        )
    if (
        not torch.isfinite(fantasy_terminal_means).all()
        or not torch.isfinite(baseline).all()
    ):
        raise IncompatibilityError(
            "p-KGFN requires finite means and finite fantasy baselines."
        )

    best = fantasy_terminal_means.amax(dim=1)
    try:
        if torch.broadcast_shapes(best.shape, baseline.shape) != best.shape:
            raise IncompatibilityError(
                f"The fantasy baseline with shape {tuple(baseline.shape)} does "
                f"not broadcast to the best-terminal means with shape "
                f"{tuple(best.shape)}."
            )
    except RuntimeError as ex:
        raise IncompatibilityError(
            f"The fantasy baseline with shape {tuple(baseline.shape)} does not "
            f"broadcast to the best-terminal means with shape {tuple(best.shape)}."
        ) from ex

    expected = (best - baseline).mean(dim=0)
    if not torch.isfinite(expected).all():
        raise IncompatibilityError("The p-KGFN gain estimate is not finite.")
    # Average the fantasy gains first, then clamp the finished estimate.
    return expected.clamp_min(0.0)


def _terminal_means_from_model(
    model: MultiStepModel,
    points: Tensor,
    *,
    num_network_samples: int,
    sampling_seed: int,
) -> Tensor:
    """Draw network paths and return the mean terminal utility."""
    samples = model.draw_network_samples(
        points,
        num_network_samples,
        sampling_seed=sampling_seed,
    )
    return compute_terminal_means_from_network_samples(model.dag, samples)


def _reference_max(
    model: MultiStepModel,
    reference_points: Tensor,
    *,
    num_network_samples: int,
    sampling_seed: int,
) -> Tensor:
    """Return the maximum terminal mean over all references."""
    return _terminal_means_from_model(
        model,
        reference_points,
        num_network_samples=num_network_samples,
        sampling_seed=sampling_seed,
    ).max()


def _make_gain_context(
    model: MultiStepModel,
    pending: tuple[_PendingAction, ...],
    reference_points: Tensor,
    *,
    num_fantasies: int,
    num_network_samples: int,
    sampling_seed: int,
) -> _GainContext:
    """Prepare baselines and pending branches once.

    Expects a model already validated by `_validate_gain_models`; the callers
    (`make_gain_function`, `recommend_direct`) perform that check once.
    """
    import torch
    from botorch.settings import propagate_grads
    from gpytorch.settings import fast_pred_var

    if (
        reference_points.ndim != 3
        or reference_points.shape[-2] != 1
        or reference_points.shape[0] < 1
    ):
        raise IncompatibilityError(
            "Global reference points must have shape '(n, 1, d)', got "
            f"{tuple(reference_points.shape)}."
        )
    if reference_points.shape[-1] != len(model.dag.comp_rep_columns):
        raise IncompatibilityError(
            "Global reference points must match the model width "
            f"{len(model.dag.comp_rep_columns)}, got "
            f"{reference_points.shape[-1]}."
        )
    if not torch.isfinite(reference_points).all():
        raise IncompatibilityError("Global reference points must be finite.")

    conditioned = (
        model.condition_on_pending(
            pending,
            num_fantasies=num_fantasies,
            sampling_seed=sampling_seed,
        )
        if pending
        else (model,)
    )
    with torch.no_grad():
        baselines = tuple(
            _reference_max(
                branch,
                reference_points,
                num_network_samples=num_network_samples,
                sampling_seed=sampling_seed,
            )
            for branch in conditioned
        )

    def evaluate(node: MultiStepNode, inputs: Tensor) -> Tensor:
        if inputs.dim() < 2 or inputs.shape[-2] != 1:
            raise IncompatibilityError(
                "Local gain inputs must have shape '(batch..., 1, d_node)', got "
                f"{tuple(inputs.shape)}."
            )
        candidate_batch = tuple(inputs.shape[:-2])
        fantasy_means: list[Tensor] = []
        fantasy_baselines: list[Tensor] = []
        for branch_model, branch_baseline in zip(conditioned, baselines):
            with propagate_grads(True), fast_pred_var(False):
                fantasy_model = branch_model.make_node_fantasies(
                    node, inputs, num_fantasies, sampling_seed
                )
            fantasy_batch = tuple(
                fantasy_model.get_node_model(node.name).batch_shape
            )
            expected_batch = (num_fantasies, *candidate_batch)
            if fantasy_batch == expected_batch:
                output_batch = fantasy_batch
            elif fantasy_batch == (*expected_batch, 1):
                output_batch = expected_batch
            else:
                raise IncompatibilityError(
                    f"Fantasizing node '{node.name}' produced batch shape "
                    f"{fantasy_batch}, expected {expected_batch} optionally "
                    "followed by a singleton output axis."
                )
            points = reference_points.reshape(
                len(reference_points),
                *([1] * len(fantasy_batch)),
                *reference_points.shape[-2:],
            ).expand(
                len(reference_points),
                *fantasy_batch,
                *reference_points.shape[-2:],
            )
            with propagate_grads(True), fast_pred_var(False):
                means = _terminal_means_from_model(
                    fantasy_model,
                    points,
                    num_network_samples=num_network_samples,
                    sampling_seed=sampling_seed,
                )
            if output_batch != fantasy_batch:
                means = means.squeeze(-1)
            fantasy_means.append(means.movedim(1, 0))
            fantasy_baselines.append(
                branch_baseline.reshape(()).expand(num_fantasies, *candidate_batch)
            )

        return compute_pkgfn_gain(
            torch.cat(fantasy_means, dim=0),
            torch.cat(fantasy_baselines, dim=0),
        )

    return _GainContext(
        evaluate=evaluate,
        num_branches=len(conditioned),
        num_references=len(reference_points),
    )
