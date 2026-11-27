"""Direct finite-reference partial-knowledge-gradient policy for networks.

The policy scores each executable local action against a fixed, finite set of
global reference configurations. Only the chosen action's own free parameters
are optimized, using BoTorch's local mixed/continuous search; no global or local
parameter product is ever constructed.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, NoReturn, cast

import numpy as np
import pandas as pd
from attrs import Attribute, define, field
from attrs.validators import and_, ge, gt, instance_of, not_
from typing_extensions import override

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.categorical import CategoricalParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.searchspace.core import SearchSpace
from baybe.serialization import SerialMixin
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import classproperty
from baybe.utils.validation import finite_float

from ..acquisition import (
    CostMode,
    GainScale,
    _get_action_cost,
    _make_gain_context,
    _validate_gain_models,
)
from ..data import _ExecutablePlan, _PendingAction, contains_action
from .gain import _GainContext
from .references import _make_reference_points

if TYPE_CHECKING:
    from torch import Tensor

    from ..dag import MultiStepDAG
    from ..data import MultiStepData
    from ..model import MultiStepModel
    from ..node import MultiStepNode


_INTEGER_VALIDATOR: Callable[[Any, Attribute, int], None] = and_(
    instance_of(int),
    not_(instance_of(bool), msg="Expected an integer, not a Boolean."),
)
"""The attrs validator shared by the integer profile fields."""


@define(frozen=True, slots=False)
class TerminalSearchProfile(SerialMixin):
    """Immutable numerical settings of the local direct action search.

    The defaults are the released search profile. They are numerical optimizer
    settings, not an additional action or success threshold.
    """

    num_network_samples: int = field(
        default=64, validator=[_INTEGER_VALIDATOR, gt(0)]
    )
    """The number of complete network-path samples per terminal evaluation."""

    sampling_seed: int = field(
        default=0, validator=[_INTEGER_VALIDATOR, ge(0)]
    )
    """The base seed for references, network paths, and fantasy observations."""

    raw_samples: int = field(default=64, validator=[_INTEGER_VALIDATOR, gt(0)])
    """The number of parameter-wise raw starts per local action search."""

    num_restarts: int = field(default=10, validator=[_INTEGER_VALIDATOR, gt(0)])
    """The number of admissible raw starts that are optimized."""

    tolerance: float = field(
        default=1e-9, converter=float, validator=[finite_float, gt(0.0)]
    )
    """The convergence tolerance of the local action search."""

    def __attrs_post_init__(self) -> None:
        """Reject profiles whose raw sample count cannot seed the restarts."""
        if self.raw_samples < self.num_restarts:
            raise ValueError(
                "raw_samples must be greater than or equal to num_restarts."
            )


@define(frozen=True)
class DirectPartialKnowledgeGradientFN(AcquisitionFunction):
    """Direct finite-reference p-KGFN over an explicit product-free multi-step DAG.

    The policy scores each executable local action against a fixed, finite set of
    global reference configurations and optimizes only that action's own free
    parameters with BoTorch's local search. It is an explicit opt-in handled by
    :meth:`MultiStepCampaign.recommend` and cannot be used as a stock BoTorch
    acquisition function.
    """

    abbreviation: ClassVar[str] = "qDpKGFN"
    """An alternative name for type resolution."""

    supports_multi_output: ClassVar[bool] = False
    """Whether this acquisition function can handle models with multiple outputs."""

    gain_scale: ClassVar[GainScale] = "linear"
    """The direct policy uses a linear, nonnegative pKGFN gain."""

    supports_signed_gains: ClassVar[bool] = False
    """The pKGFN gain is clamped to be nonnegative before cost weighting."""

    num_fantasies: int = field(default=8, validator=[_INTEGER_VALIDATOR, gt(0)])
    """The number of fantasy observations per action and pending branch."""

    search_profile: TerminalSearchProfile = field(
        factory=TerminalSearchProfile,
        validator=instance_of(TerminalSearchProfile),
    )
    """The immutable deterministic search profile."""

    @override
    @classproperty
    def supports_batching(cls) -> bool:
        """Whether batch recommendation is supported."""
        return False

    @override
    @classproperty
    def supports_pending_experiments(cls) -> bool:
        """Whether pending experiments are supported."""
        return True

    @override
    def to_botorch(
        self,
        surrogate: SurrogateProtocol,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
        pending_experiments: pd.DataFrame | None = None,
    ) -> NoReturn:
        """Reject the stock BoTorch acquisition construction.

        Args:
            surrogate: The surrogate model.
            searchspace: The search space.
            objective: The objective.
            measurements: The available measurements.
            pending_experiments: Optional pending experiments.

        Raises:
            IncompatibilityError: Always, because the policy requires the
                multi-step campaign orchestration.
        """
        raise IncompatibilityError(
            f"'{self.__class__.__name__}' is a multi-step pKGFN policy. "
            "Use 'MultiStepCampaign.recommend()' instead."
        )

    def validate_context(
        self, dag: MultiStepDAG, data: MultiStepData, batch_size: int
    ) -> None:
        """Validate the policy's static prerequisites before fitting models.

        Args:
            dag: The DAG topology, including the objective.
            data: The measurements determining executable actions.
            batch_size: The requested number of recommendations.

        Raises:
            IncompatibilityError: If the BoTorch version, batch size, DAG opt-in,
                objective, parameter types, or layout are unsupported.
        """
        _validate_botorch_version()
        del data

        if batch_size != 1:
            raise IncompatibilityError(
                "The direct pKGFN policy supports only a single recommendation, "
                f"got {batch_size=}."
            )
        if not dag.product_free:
            raise IncompatibilityError(
                "The direct pKGFN policy requires an explicit product-free DAG "
                "(product_free=True)."
            )

        objective = dag.objective
        if objective.is_multi_output or not all(
            isinstance(target, NumericalTarget) for target in objective.targets
        ):
            raise IncompatibilityError(
                "The direct pKGFN policy requires a scalar objective over "
                "numerical targets, but cannot handle "
                f"'{objective.__class__.__name__}' with targets "
                f"{[target.name for target in objective.targets]}."
            )
        if not callable(getattr(objective, "_full_transformation", None)):
            raise IncompatibilityError(
                "The direct pKGFN policy requires the objective's end-to-end "
                "target transformation."
            )

        try:
            layout = dag._configuration_space
        except IncompatibilityError:
            raise
        unsupported = sorted(
            parameter.name
            for parameter in layout.parameters
            if not isinstance(
                parameter,
                (
                    NumericalContinuousParameter,
                    NumericalDiscreteParameter,
                ),
            )
            and not isinstance(parameter, CategoricalParameter)
        )
        if unsupported:
            raise IncompatibilityError(
                "The direct pKGFN policy supports numerical continuous, "
                "numerical discrete, and categorical OHE/INT parameters, got: "
                f"{unsupported}."
            )

    def validate_model(self, model: MultiStepModel) -> None:
        """Validate the fitted-model capabilities required by the policy.

        Args:
            model: The fitted network model.

        Raises:
            IncompatibilityError: If the DAG is not product-free or a node model
                is batched or cannot be conditioned on fantasy observations.
        """
        if not model.dag.product_free:
            raise IncompatibilityError(
                "The direct pKGFN policy requires a product-free DAG."
            )
        _validate_gain_models(model)

    def make_gain_function(
        self,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...] = (),
    ) -> Callable[[Any, MultiStepNode, Tensor], Tensor]:
        """Reject nested per-action gain construction.

        Args:
            model: The fitted network model.
            pending: Already reserved actions.

        Raises:
            IncompatibilityError: Always, because the direct policy prepares its
                own finite reference context inside ``recommend_direct``.
        """
        del model, pending
        raise IncompatibilityError(
            "The direct pKGFN policy does not provide a local gain function. It "
            "prepares its finite reference context inside 'recommend_direct'."
        )


def _validate_botorch_version() -> None:
    """Reject BoTorch versions without reliable categorical rounding."""
    import botorch

    try:
        from packaging.version import Version
    except ImportError as ex:
        raise IncompatibilityError(
            "Validating the BoTorch version for the direct pKGFN policy requires "
            "the 'packaging' package. Install it in this environment."
        ) from ex

    if Version(botorch.__version__) < Version("0.16"):
        raise IncompatibilityError(
            "The direct pKGFN policy requires BoTorch >= 0.16 for reliable "
            "discrete and categorical rounding, but found "
            f"{botorch.__version__}. Upgrade BoTorch in this environment."
        )


@define(frozen=True, slots=False)
class _ActionSearchSpace:
    """Encode one node's free own parameters as BoTorch coordinates."""

    parameters: tuple[Any, ...] = field(converter=tuple)
    """All own parameters of the node in layout order."""

    continuous: tuple[NumericalContinuousParameter, ...] = field(converter=tuple)
    """The continuous own parameters, which occupy the leading coordinates."""

    free_discrete: tuple[DiscreteParameter, ...] = field(converter=tuple)
    """The discrete own parameters with more than one active level."""

    @classmethod
    def from_layout(cls, layout: Any) -> _ActionSearchSpace:
        """Build the encoding from a factorized own-parameter layout."""
        return cls(
            parameters=layout.parameters,
            continuous=layout._continuous_parameters,
            free_discrete=tuple(
                parameter
                for parameter in layout._discrete_parameters
                if len(parameter.active_values) > 1
            ),
        )

    @property
    def continuous_dimension(self) -> int:
        """The number of leading continuous coordinates."""
        return len(self.continuous)

    @property
    def dimension(self) -> int:
        """The total number of action coordinates."""
        return self.continuous_dimension + len(self.free_discrete)

    @property
    def bounds(self) -> np.ndarray:
        """The action bounds of shape '(d, 2)' in coordinate order."""
        rows = [
            [parameter.bounds.lower, parameter.bounds.upper]
            for parameter in self.continuous
        ]
        rows.extend(
            [0.0, len(parameter.active_values) - 1.0]
            for parameter in self.free_discrete
        )
        return np.asarray(rows, dtype=float).reshape(-1, 2)

    def _code_dims(self, parameter_type: type) -> dict[int, list[float]]:
        """Return all admissible codes for free discrete parameters of one type."""
        return {
            self.continuous_dimension + position: [
                float(level) for level in range(len(parameter.active_values))
            ]
            for position, parameter in enumerate(self.free_discrete)
            if isinstance(parameter, parameter_type)
        }

    @property
    def discrete_dims(self) -> dict[int, list[float]]:
        """BoTorch discrete dimensions and their admissible code values."""
        return self._code_dims(NumericalDiscreteParameter)

    @property
    def cat_dims(self) -> dict[int, list[float]]:
        """BoTorch categorical dimensions and their admissible code values."""
        return self._code_dims(CategoricalParameter)

    def decode_continuous(self, points: Tensor) -> Tensor:
        """Return differentiable continuous values, shape '(batch, n_continuous)'."""
        return points[:, : self.continuous_dimension]

    def decode_levels(self, points: Tensor) -> tuple[tuple[int, ...], ...]:
        """Return one active-level index per declared discrete parameter."""
        import torch

        levels: list[Tensor] = []
        free_position = 0
        for parameter in self.parameters:
            if isinstance(parameter, NumericalContinuousParameter):
                continue
            count = len(parameter.active_values)
            if count == 1:
                levels.append(
                    torch.zeros(len(points), dtype=torch.long, device=points.device)
                )
                continue
            coordinate = points[:, self.continuous_dimension + free_position]
            levels.append(coordinate.round().long().clamp(0, count - 1))
            free_position += 1
        if not levels:
            return tuple(() for _ in range(len(points)))
        stacked = torch.stack(levels, dim=1)
        return tuple(tuple(int(value) for value in row) for row in stacked)

    def decode_inputs(
        self, points: Tensor, plan_inputs: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Decode action coordinates into complete local experimental inputs."""
        import torch

        continuous = self.decode_continuous(points).detach()
        if self.continuous:
            lower = torch.tensor(
                [parameter.bounds.lower for parameter in self.continuous],
                dtype=continuous.dtype,
                device=continuous.device,
            )
            upper = torch.tensor(
                [parameter.bounds.upper for parameter in self.continuous],
                dtype=continuous.dtype,
                device=continuous.device,
            )
            continuous = torch.minimum(torch.maximum(continuous, lower), upper)
        values = continuous.cpu().tolist()
        levels = self.decode_levels(points)
        rows: list[dict[str, Any]] = []
        for index in range(len(points)):
            row = dict(plan_inputs)
            for position, parameter in enumerate(self.continuous):
                row[parameter.name] = float(values[index][position])
            level_position = 0
            for parameter in self.parameters:
                if isinstance(parameter, NumericalContinuousParameter):
                    continue
                row[parameter.name] = parameter.active_values[
                    levels[index][level_position]
                ]
                level_position += 1
            rows.append(row)
        return rows


def _action_comp_rep_batch(
    model: MultiStepModel,
    node: MultiStepNode,
    plan: _ExecutablePlan,
    space: _ActionSearchSpace,
    points: Tensor,
) -> Tensor:
    """Build fitted model inputs of shape '(batch, d_node)'."""
    import torch

    levels = space.decode_levels(points)
    rows: list[dict[str, Any]] = []
    for index in range(len(points)):
        row = dict(plan.inputs)
        level_position = 0
        for parameter in space.parameters:
            if isinstance(parameter, NumericalContinuousParameter):
                row[parameter.name] = parameter.bounds.lower
            else:
                row[parameter.name] = cast(
                    DiscreteParameter, parameter
                ).active_values[levels[index][level_position]]
                level_position += 1
        rows.append(row)

    base = model.transform_node_inputs(node.name, pd.DataFrame(rows))
    columns = model.get_node_model_columns(node.name)
    if not columns:
        return base.new_empty((len(points), 0))
    live = {
        parameter.comp_rep_columns[0]: points[:, index]
        for index, parameter in enumerate(space.continuous)
    }
    pieces = [
        live[column].to(dtype=base.dtype, device=base.device).reshape(-1, 1)
        if column in live
        else base[:, index].reshape(-1, 1)
        for index, column in enumerate(columns)
    ]
    return torch.cat(pieces, dim=1)


_CODE_TOLERANCE: float = 1e-6
"""Maximum numerical deviation of a returned code value."""

_SEARCH_RETRY_MESSAGES: tuple[str, ...] = (
    "Optimization failed in `gen_candidates_scipy`",
    "Optimization failed on the second try",
)


def _action_score(
    gain: Tensor, probability: Tensor, cost: float
) -> tuple[Tensor, Tensor]:
    """Return '(KG, score)' with 'KG = probability * gain' and 'score = KG / cost'."""
    import torch

    if gain.shape != probability.shape:
        raise IncompatibilityError(
            "The pKGFN gain and local success probability must have identical "
            "shapes."
        )
    if not torch.isfinite(gain).all():
        raise IncompatibilityError("The pKGFN gain must be finite.")
    if (gain < 0).any():
        raise IncompatibilityError(
            "The pKGFN gain must be nonnegative after clamping."
        )
    if (
        not torch.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        raise IncompatibilityError(
            "The local success probability must be finite and in [0, 1]."
        )
    if not math.isfinite(cost) or cost <= 0:
        raise IncompatibilityError(
            "The action cost must be finite and strictly positive."
        )
    kg = probability * gain
    return kg, kg / cost


def _action_scores(
    points: Tensor,
    *,
    model: MultiStepModel,
    node: MultiStepNode,
    plan: _ExecutablePlan,
    space: _ActionSearchSpace,
    gain_context: _GainContext,
    feasibility_threshold: float,
    cost: float,
    pending: tuple[_PendingAction, ...],
) -> Tensor:
    """Return final scores with the '-1.0' sentinel for excluded points."""
    import torch

    action_comp = _action_comp_rep_batch(model, node, plan, space, points)
    probability = model._predict_feasibility_tensor(
        node.name, action_comp.unsqueeze(-2)
    ).reshape(-1)
    if not torch.isfinite(probability).all():
        raise IncompatibilityError(
            f"The local success probability of '{node.name}' is not finite."
        )
    gain = gain_context.evaluate(node, action_comp.unsqueeze(-2)).reshape(-1)
    if not torch.isfinite(gain).all():
        raise IncompatibilityError(f"The pKGFN gain of '{node.name}' is not finite.")
    _, score = _action_score(gain, probability, cost)

    admissible = probability > 0
    if feasibility_threshold > 0:
        admissible = admissible & (probability >= feasibility_threshold)
    if pending:
        frames = space.decode_inputs(points, plan.inputs)
        allowed_inputs = torch.tensor(
            [
                not contains_action(
                    pending, node=node.name, inputs=frame, dag=model.dag
                )
                for frame in frames
            ],
            dtype=torch.bool,
            device=score.device,
        )
        admissible = admissible & allowed_inputs
    return torch.where(admissible, score, torch.full_like(score, -1.0))


def _make_action_acquisition(
    model: MultiStepModel, evaluate_points: Callable[[Tensor], Tensor]
) -> Any:
    """Create a lazy BoTorch adapter around the local action scores."""
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from botorch.utils.transforms import t_batch_mode_transform

    class _ActionAcquisition(BoAcquisitionFunction):
        @override
        @t_batch_mode_transform(expected_q=1)
        def forward(self, X):
            return evaluate_points(X[..., 0, :])

    return _ActionAcquisition(model=model)


def _raise_for_search_warnings(
    caught: Sequence[warnings.WarningMessage], *, description: str
) -> None:
    """Turn BoTorch optimization warnings in the search window into errors."""
    from botorch.exceptions.warnings import OptimizationWarning

    for warning in caught:
        if issubclass(warning.category, OptimizationWarning):
            raise IncompatibilityError(
                f"{description} raised an optimization warning: {warning.message}"
            )
        if issubclass(warning.category, RuntimeWarning) and any(
            message in str(warning.message) for message in _SEARCH_RETRY_MESSAGES
        ):
            raise IncompatibilityError(
                f"{description} failed with an optimization failure: "
                f"{warning.message}"
            )


def _validate_returned_candidate(
    candidate: Tensor,
    space: _ActionSearchSpace,
    *,
    description: str,
) -> None:
    """Validate the shape, bounds, and integer codes of an optimizer result."""
    import torch

    if candidate.dim() != 2 or candidate.shape[-1] != space.dimension:
        raise IncompatibilityError(
            f"{description} returned an invalid candidate shape "
            f"{tuple(candidate.shape)}."
        )
    if not torch.isfinite(candidate).all():
        raise IncompatibilityError(f"{description} returned non-finite values.")
    bounds = torch.as_tensor(
        space.bounds.T, dtype=candidate.dtype, device=candidate.device
    )
    if (candidate < bounds[0] - _CODE_TOLERANCE).any() or (
        candidate > bounds[1] + _CODE_TOLERANCE
    ).any():
        raise IncompatibilityError(f"{description} returned values outside bounds.")
    code_dims = set(space.discrete_dims) | set(space.cat_dims)
    for index in code_dims:
        column = candidate[:, index]
        if ((column - column.round()).abs() > _CODE_TOLERANCE).any():
            raise IncompatibilityError(
                f"{description} returned a non-integer code for dimension {index}."
            )


def _optimize_action(
    *,
    model: MultiStepModel,
    plan: _ExecutablePlan,
    gain_context: _GainContext,
    profile: TerminalSearchProfile,
    feasibility_threshold: float,
    cost: float,
    pending: tuple[_PendingAction, ...],
    context_index: int,
) -> tuple[_PendingAction, float, dict[str, Any]]:
    """Optimize only the free own parameters of one executable action."""
    import torch

    node = model.dag.node_by_name[plan.node]
    space = _ActionSearchSpace.from_layout(node._own_parameter_layout)
    reference = next(model.parameters())
    description = (
        f"The local action search for node '{node.name}' in experiment "
        f"'{plan.experiment_id}'"
    )

    def evaluate_points(points: Tensor) -> Tensor:
        return _action_scores(
            points,
            model=model,
            node=node,
            plan=plan,
            space=space,
            gain_context=gain_context,
            feasibility_threshold=feasibility_threshold,
            cost=cost,
            pending=pending,
        )

    if space.dimension == 0:
        candidate = reference.new_empty((1, 0))
        score = float(evaluate_points(candidate)[0].detach())
        if score < 0:
            raise IncompatibilityError(
                f"The only action of node '{node.name}' in experiment "
                f"'{plan.experiment_id}' is excluded by pending state or "
                "feasibility; this is not proof that the context is infeasible."
            )
        best_value = score
    else:
        from botorch.exceptions.errors import (
            CandidateGenerationError,
            UnsupportedError,
        )

        acquisition = _make_action_acquisition(model, evaluate_points)
        bounds = torch.as_tensor(space.bounds.T).to(reference)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if space.discrete_dims or space.cat_dims:
                from botorch.optim import optimize_acqf_mixed_alternating

                try:
                    candidate, best = optimize_acqf_mixed_alternating(
                        acquisition,
                        bounds,
                        discrete_dims=space.discrete_dims or None,
                        cat_dims=space.cat_dims or None,
                        options={
                            "initialization_strategy": "random",
                            "num_spray_points": 0,
                            "max_discrete_values": 10**6,
                            "maxiter_alternating": 64,
                            "maxiter_discrete": 4,
                            "maxiter_continuous": 8,
                            "tol": profile.tolerance,
                        },
                        q=1,
                        raw_samples=profile.raw_samples,
                        num_restarts=profile.num_restarts,
                    )
                except (CandidateGenerationError, UnsupportedError) as ex:
                    raise IncompatibilityError(f"{description} failed.") from ex
            else:
                from botorch.optim import optimize_acqf

                try:
                    candidate, best = optimize_acqf(
                        acquisition,
                        bounds,
                        q=1,
                        num_restarts=profile.num_restarts,
                        raw_samples=profile.raw_samples,
                        options={"maxiter": 200, "ftol": profile.tolerance},
                    )
                except (CandidateGenerationError, UnsupportedError) as ex:
                    raise IncompatibilityError(f"{description} failed.") from ex
        _raise_for_search_warnings(caught, description=description)
        candidate = candidate.detach().reshape(-1, space.dimension)
        _validate_returned_candidate(candidate, space, description=description)
        score = float(evaluate_points(candidate)[0].detach())
        if score < 0:
            raise IncompatibilityError(
                f"{description} returned an excluded action. This is not proof "
                "that the context is infeasible."
            )
        best_value = float(best.reshape(-1)[0].detach()) if best is not None else score

    inputs = space.decode_inputs(candidate, plan.inputs)[0]
    action = _PendingAction(
        node=plan.node,
        inputs=inputs,
        experiment_id=plan.experiment_id,
        parent_refs=plan.parent_refs,
    )
    diagnostics = {
        "context_index": context_index,
        "node": plan.node,
        "experiment_id": plan.experiment_id,
        "score": score,
        "references": gain_context.num_references,
        "branches": gain_context.num_branches,
        "best_value": best_value,
    }
    return action, score, diagnostics


def recommend_direct(
    model: MultiStepModel,
    plans: tuple[_ExecutablePlan, ...],
    pending: tuple[_PendingAction, ...] = (),
    *,
    policy: DirectPartialKnowledgeGradientFN,
    feasibility_threshold: float = 1e-6,
    cost_mode: CostMode = "remaining",
) -> tuple[_PendingAction, dict[str, Any]]:
    """Score all executable contexts with local BoTorch actions and select one.

    Args:
        model: The fitted network model.
        plans: The concrete executable action contexts.
        pending: Already reserved actions whose outcomes are fantasized and whose
            identical actions are excluded.
        policy: The direct policy.
        feasibility_threshold: The minimum positive local success probability.
        cost_mode: Whether to score with the next node's cost or the remaining
            process cost. Defaults to ``"remaining"``.

    Returns:
        The selected action and deterministic per-context diagnostics.

    Raises:
        IncompatibilityError: If no context exists, the policy or model is
            unsupported, or a local action search fails.
    """
    policy.validate_model(model)
    if not plans:
        raise IncompatibilityError(
            "The direct pKGFN policy found no executable local context."
        )

    profile = policy.search_profile
    references = _make_reference_points(model, profile)
    gain_context = _make_gain_context(
        model,
        pending,
        references,
        num_fantasies=policy.num_fantasies,
        num_network_samples=profile.num_network_samples,
        sampling_seed=profile.sampling_seed,
    )

    scored: list[tuple[float, int, _PendingAction, dict[str, Any]]] = []
    for context_index, plan in enumerate(plans):
        action, score, diagnostics = _optimize_action(
            model=model,
            plan=plan,
            gain_context=gain_context,
            profile=profile,
            feasibility_threshold=feasibility_threshold,
            cost=_get_action_cost(model.dag, plan, cost_mode),
            pending=pending,
            context_index=context_index,
        )
        scored.append((score, context_index, action, diagnostics))

    best_score = max(item[0] for item in scored)
    # Stable tie rule: the earliest existing context wins among equal scores.
    selected = min(
        (item for item in scored if item[0] == best_score),
        key=lambda item: item[1],
    )
    diagnostics = {
        "selected_context_index": selected[1],
        "selected_score": selected[0],
        "contexts": tuple(item[3] for item in scored),
        "tie_count": sum(item[0] == best_score for item in scored),
        "branches": gain_context.num_branches,
        "references": gain_context.num_references,
    }
    return selected[2], diagnostics
