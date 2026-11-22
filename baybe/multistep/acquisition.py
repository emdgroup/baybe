"""Acquisition gains and shared recommendation logic for multi-step networks.

The module owns the graph-aware acquisition contract and its concrete
partial-knowledge-gradient implementation. An acquisition computes an
unweighted gain for a node action. Independent helpers in this module handle
action generation, local risk and cost weighting, optimization, and selection.
"""

from __future__ import annotations

import gc
import math
import warnings
from collections.abc import Callable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NoReturn,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np
import pandas as pd
from attrs import Attribute, define, evolve, field
from attrs.converters import optional as optional_converter
from attrs.validators import and_, ge, gt, instance_of, not_
from attrs.validators import optional as optional_validator
from typing_extensions import override

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import (
    IncompatibilityError,
    InfeasibilityError,
    ModelNotTrainedError,
)
from baybe.objectives.base import Objective
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.targets.numerical import NumericalTarget
from baybe.utils.validation import finite_float

from .dag import MultiStepDAG
from .data import (
    MultiStepData,
    _ActionContext,
    _ExecutablePlan,
    _identifier_number,
    _PendingAction,
    contains_action,
)

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from botorch.models.model import Model
    from torch import Tensor

    from .model import MultiStepModel
    from .node import MultiStepNode


GainScale = Literal["linear", "log"]
"""The representation used by an acquisition function's raw gain."""


@runtime_checkable
class GraphAwareAcquisitionProtocol(Protocol):
    """Structural contract for acquisitions that score concrete node actions.

    Implementations prepare one global gain context per recommendation call and
    then return a raw, differentiable gain for candidate local inputs. They do
    not generate executable actions, apply costs or local success risk, optimize
    candidates, select an action, or mutate campaign state.
    """

    __slots__ = ()

    gain_scale: GainScale
    """The representation of the returned raw gain."""

    supports_signed_gains: bool
    """Whether finite negative raw gains are valid for this acquisition."""

    def validate_context(
        self, dag: MultiStepDAG, data: MultiStepData, batch_size: int
    ) -> None:
        """Validate static prerequisites before fitting models."""
        ...

    def validate_model(self, model: MultiStepModel) -> None:
        """Validate fitted-model capabilities required by this acquisition."""
        ...

    def make_gain_function(
        self,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...] = (),
    ) -> Callable[[_ActionContext, MultiStepNode, Tensor], Tensor]:
        """Prepare the global gain context and return a local node-gain function.

        The returned function accepts a node and candidate local inputs with
        shape ``(batch..., 1, d_node)`` and returns one raw gain value per
        candidate batch.
        """
        ...


@define(frozen=True)
class PartialKnowledgeGradientFN(AcquisitionFunction):
    """Partial-knowledge-gradient policy for function networks.

    The policy evaluates a local node measurement by its expected improvement of
    the best terminal decision over a finite set of global reference
    configurations. Continuous node-owned inputs are optimized against that
    reference set by the shared recommendation layer.

    The policy returns raw gains only. Action generation, success-risk
    weighting, remaining-cost accounting, optimization, and final selection are
    performed by the shared multi-step orchestration layer.

    Nodes with feasibility conditions are supported through independently
    learned success-probability models. Their local success risk is applied once
    by the shared recommendation layer. Optionally, terminal decisions also
    account for network success and an explicitly configured failure utility.
    The local weighting remains in place in either case.
    """

    abbreviation: ClassVar[str] = "pKGFN"
    """An alternative name for type resolution."""

    supports_multi_output: ClassVar[bool] = False
    """Whether this acquisition function can handle models with multiple outputs."""

    gain_scale: ClassVar[GainScale] = "linear"
    """p-KGFN returns a nonnegative linear gain after fantasy averaging."""

    supports_signed_gains: ClassVar[bool] = False
    """The completed p-KGFN estimate is clamped to zero before weighting."""

    _integer_validator: ClassVar[Callable[[Any, Attribute, int], None]] = and_(
        instance_of(int),
        not_(instance_of(bool), msg="Expected an integer, not a Boolean."),
    )
    """The attrs validator shared by the integer configuration fields."""

    num_fantasies: int = field(default=8, validator=[_integer_validator, gt(0)])
    """The number of fantasy observations per measurement action."""

    num_network_samples: int = field(default=64, validator=[_integer_validator, gt(0)])
    """The number of network-path samples used for the terminal evaluation."""

    sampling_seed: int = field(default=0, validator=[_integer_validator, ge(0)])
    """The base seed for reproducible fantasies and network-path samples."""

    num_terminal_samples: int = field(default=64, validator=[_integer_validator, gt(0)])
    """Sobol reference points per discrete global assignment for continuous spaces."""

    max_terminal_candidates: int = field(
        default=256, validator=[_integer_validator, gt(0)]
    )
    """The maximum number of terminal configurations that may be enumerated."""

    terminal_failure_utility: float | None = field(
        default=None,
        converter=optional_converter(float),
        validator=optional_validator(finite_float),
        kw_only=True,
    )
    """Utility assigned to failed terminal decisions, or ``None`` to disable it.

    When set, each sampled terminal utility is weighted by the corresponding
    product of node success probabilities. The remaining probability is assigned
    this failure utility. This models conditional node successes as independent
    given the sampled network path; it does not update classifiers in fantasies.
    """

    @staticmethod
    def _compute_pkgfn_gain(
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
            IncompatibilityError: If the input shape, baseline, or gain
                is invalid or non-finite.
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

    @staticmethod
    def _compute_terminal_means_from_network_samples(
        dag: MultiStepDAG,
        samples: Tensor,
        *,
        success_probabilities: Tensor | None = None,
        failure_utility: float | None = None,
    ) -> Tensor:
        """Compute terminal utility means from raw network samples.

        Args:
            dag: The DAG defining the terminal objective.
            samples: Raw samples with shape
                ``(n_paths, batch..., q, num_outputs)``.
            success_probabilities: Optional network success probabilities for each
                sample, with shape ``(n_paths, batch..., q)``.
            failure_utility: Utility of a failed terminal decision, required when
                success probabilities are provided.

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

        if (success_probabilities is None) != (failure_utility is None):
            raise IncompatibilityError(
                "Terminal success probabilities and failure utility must be "
                "provided together."
            )
        if success_probabilities is not None:
            assert failure_utility is not None
            if (
                success_probabilities.shape != utilities.shape
                or not torch.isfinite(success_probabilities).all()
                or (success_probabilities < 0).any()
                or (success_probabilities > 1).any()
                or not math.isfinite(failure_utility)
            ):
                raise IncompatibilityError(
                    "Terminal success probabilities must match the utility shape "
                    "and lie in [0, 1], and failure utility must be finite."
                )
            probabilities = success_probabilities.to(utilities)
            utilities = (
                probabilities * utilities + (1 - probabilities) * failure_utility
            )

        return utilities.squeeze(-1).mean(dim=0)

    def _compute_terminal_means(
        self,
        model: MultiStepModel,
        points: Tensor,
    ) -> Tensor:
        """Evaluate terminal decisions with the learned network.

        Args:
            model: The current or hypothetical network model.
            points: Terminal reference configurations.

        Returns:
            Mean utility per decision.
        """
        probabilities = None
        if self.terminal_failure_utility is None:
            samples = model.draw_network_samples(
                points,
                self.num_network_samples,
                sampling_seed=self.sampling_seed,
            )
        else:
            samples, probabilities = model.draw_network_outcomes(
                points,
                self.num_network_samples,
                sampling_seed=self.sampling_seed,
            )
        return self._compute_terminal_means_from_network_samples(
            model.dag,
            samples,
            success_probabilities=probabilities,
            failure_utility=self.terminal_failure_utility,
        )

    def _pending_branches(  # noqa: DOC502
        self,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...],
        reference_points: Tensor,
        baseline: Tensor,
    ) -> tuple[tuple[MultiStepModel, Tensor], ...]:
        """Create one conditioned network model per joint pending fantasy.

        All pending executions of a node are sampled jointly so that their GP
        correlations and observation noise are retained. Nodes are independent
        given their local data, so the joint network fantasy combines the node
        samples by index rather than by a Cartesian product.

        Args:
            model: The fitted real network model.
            pending: The reserved local executions.
            reference_points: The global terminal reference configurations.
            baseline: The baseline utility without pending conditioning.

        Returns:
            One ``(conditioned_model, baseline_value)`` pair per pending fantasy.
            Without pending actions, the original model and baseline are
            returned unchanged.

        Raises:
            IncompatibilityError: If a pending node is unknown or a local input
                cannot be transformed.
        """
        import torch

        if not pending:
            return ((model, baseline),)

        conditioned_models = model.condition_on_pending(
            pending,
            num_fantasies=self.num_fantasies,
            sampling_seed=self.sampling_seed,
        )
        branches: list[tuple[MultiStepModel, Tensor]] = []
        for conditioned in conditioned_models:
            with torch.no_grad():
                value = self._compute_terminal_means(
                    conditioned, reference_points
                ).max()
            branches.append((conditioned, value))
        return tuple(branches)

    def _make_terminal_reference_tensor(
        self, dag: MultiStepDAG, reference: Tensor
    ) -> Tensor:
        """Create reproducible terminal reference configurations.

        Args:
            dag: The DAG defining the global search space.
            reference: A tensor supplying the model's device and dtype.

        Returns:
            Reference points in computational representation, with a singleton
            q-axis.
        """
        import torch
        from botorch.utils.sampling import draw_sobol_samples

        searchspace = dag.full_searchspace
        discrete_space = searchspace.discrete
        terminal_exp = (
            discrete_space.exp_rep
            if not discrete_space.is_empty
            else pd.DataFrame(index=[0])
        )
        continuous = searchspace.continuous
        if not continuous.is_empty:
            bounds = torch.as_tensor(continuous.comp_rep_bounds.to_numpy()).to(
                reference
            )
            samples = draw_sobol_samples(
                bounds=bounds,
                n=self.num_terminal_samples,
                q=1,
                seed=self.sampling_seed,
            ).squeeze(-2)
            continuous_exp = pd.DataFrame(
                samples.cpu().numpy(), columns=continuous.parameter_names
            )
            terminal_exp = terminal_exp.merge(continuous_exp, how="cross")

        terminal_comp = searchspace.transform(terminal_exp)
        return (
            torch.as_tensor(terminal_comp.to_numpy(copy=True))
            .to(reference)
            .unsqueeze(-2)
        )

    def _validate_fitted_models(self, model: MultiStepModel) -> None:
        """Validate the fitted-model capabilities required by p-KGFN.

        Args:
            model: The fitted network model.

        Raises:
            IncompatibilityError: If a node model is batched or cannot be conditioned.
            ModelNotTrainedError: If a configured feasibility model is missing.
        """
        import torch

        if model.missing_feasibility_nodes:
            raise ModelNotTrainedError(
                "p-KGFN requires fitted local feasibility models for configured "
                f"nodes {list(model.missing_feasibility_nodes)}."
            )

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

    def make_gain_function(  # noqa: DOC502
        self,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...] = (),
    ) -> Callable[[_ActionContext, MultiStepNode, Tensor], Tensor]:
        """Prepare the global p-KGFN context and return a node-gain function.

        The terminal reference points, unconditional baseline, and pending
        fantasy branches are prepared once. The returned function then evaluates
        raw local gains for arbitrary nodes and candidate batches in that same
        global context.

        Args:
            model: The fitted network model, including the DAG and objective.
            pending: Already reserved actions whose outcomes are fantasized.

        Returns:
            A differentiable function mapping a node and local candidate inputs
            to raw p-KGFN gains.

        Raises:
            IncompatibilityError: If the acquisition or model capabilities are
                unsupported.
        """
        import torch

        self._validate_fitted_models(model)
        reference = next(model.parameters())
        reference_points = self._make_terminal_reference_tensor(model.dag, reference)
        with torch.no_grad():
            baseline = self._compute_terminal_means(model, reference_points).max()
        pending_branches = self._pending_branches(
            model,
            pending,
            reference_points,
            baseline,
        )

        def gain(
            _context: _ActionContext, node: MultiStepNode, inputs: Tensor
        ) -> Tensor:
            """Compute p-KGFN's raw utility gain before action weighting."""
            import torch
            from botorch.settings import propagate_grads
            from gpytorch.settings import fast_pred_var

            if inputs.dim() < 2 or inputs.shape[-2] != 1:
                raise IncompatibilityError(
                    "p-KGFN candidate inputs must have shape "
                    f"'(batch..., 1, d_node)', got {tuple(inputs.shape)}."
                )
            candidate_batch = tuple(inputs.shape[:-2])
            fantasy_means: list[Tensor] = []
            fantasy_baselines: list[Tensor] = []
            for conditioned_model, branch_baseline in pending_branches:
                with propagate_grads(True), fast_pred_var(False):
                    fantasy_model = conditioned_model.make_node_fantasies(
                        node, inputs, self.num_fantasies, self.sampling_seed
                    )
                fantasy_batch = tuple(
                    fantasy_model.get_node_model(node.name).batch_shape
                )
                expected_batch = (self.num_fantasies, *candidate_batch)
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
                    means = self._compute_terminal_means(fantasy_model, points)
                if output_batch != fantasy_batch:
                    means = means.squeeze(-1)
                fantasy_means.append(means.movedim(1, 0))
                fantasy_baselines.append(
                    branch_baseline.expand(self.num_fantasies, *candidate_batch)
                )

            return self._compute_pkgfn_gain(
                torch.cat(fantasy_means, dim=0),
                torch.cat(fantasy_baselines, dim=0),
            )

        return gain

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
            IncompatibilityError: Always, because the policy is not a stock
                point acquisition function.
        """
        raise IncompatibilityError(
            f"'{self.__class__.__name__}' is a multi-step policy. "
            "Use 'MultiStepCampaign.recommend()' instead."
        )

    def validate_context(
        self, dag: MultiStepDAG, data: MultiStepData, batch_size: int
    ) -> None:
        """Validate the policy's static prerequisites before fitting.

        Args:
            dag: The DAG topology, including the objective.
            data: The measurements determining executable actions.
            batch_size: The requested number of recommendations.

        Raises:
            IncompatibilityError: If the batch size, objective, parameter types,
                surrogate configurations, or terminal candidate limit is
                unsupported.
        """
        if batch_size != 1:
            raise IncompatibilityError(
                f"p-KGFN supports only a single recommendation, got {batch_size=}."
            )

        objective = dag.objective
        if objective.is_multi_output or not all(
            isinstance(target, NumericalTarget) for target in objective.targets
        ):
            raise IncompatibilityError(
                "p-KGFN requires a scalar objective over numerical targets, but "
                f"cannot handle '{objective.__class__.__name__}' with targets "
                f"{[target.name for target in objective.targets]}."
            )

        unsupported = sorted(
            parameter.name
            for node in dag.nodes
            for parameter in node.inputs
            if not isinstance(
                parameter, (DiscreteParameter, NumericalContinuousParameter)
            )
        )
        if unsupported:
            raise IncompatibilityError(
                "p-KGFN supports only discrete and numerical continuous "
                "parameters, got: "
                f"{unsupported}."
            )
        has_continuous = any(
            isinstance(parameter, NumericalContinuousParameter)
            for parameter in dag.parameter_by_name.values()
        )
        n_discrete_terminal = math.prod(
            len(parameter.values)
            for parameter in dag.parameter_by_name.values()
            if isinstance(parameter, DiscreteParameter)
        )
        n_terminal = n_discrete_terminal * (
            self.num_terminal_samples if has_continuous else 1
        )
        if n_terminal > self.max_terminal_candidates:
            raise IncompatibilityError(
                f"The terminal reference set has {n_terminal} candidates, which "
                f"exceeds the configured limit of {self.max_terminal_candidates}."
            )

    def validate_model(self, model: MultiStepModel) -> None:  # noqa: DOC502
        """Validate the fitted model capabilities required by p-KGFN.

        Args:
            model: The fitted network model.

        Raises:
            IncompatibilityError: If a node model is batched or cannot be conditioned.
            ModelNotTrainedError: If a configured feasibility model is missing.
        """
        self._validate_fitted_models(model)


ScoreScale = GainScale
"""The representation used by an acquisition function's gain."""

_PENDING_FANTASIES: int = 8
"""Pending outcome branches used when averaging a stock acquisition."""

_PENDING_FANTASY_SEED: int = 0
"""Deterministic seed for stock pending-outcome sampling."""

_N_RESTARTS: int = 10
"""Multi-start optimization restarts for continuous local inputs."""

_N_RAW_SAMPLES: int = 64
"""Raw samples used to initialize continuous local optimization."""

_OPTIMIZER_SEED: int = 0
"""Deterministic seed for continuous local optimization."""


def _resolve_score_scale(acquisition: object) -> ScoreScale:
    """Resolve the declared gain scale of a supported stock acquisition.

    Args:
        acquisition: The concrete BayBE acquisition function.

    Returns:
        The representation used by the acquisition's nonnegative gain.

    Raises:
        IncompatibilityError: If the acquisition does not have a supported gain
            contract.
    """
    from baybe.acquisition import (
        qExpectedImprovement,
        qLogEI,
        qLogNEHVI,
        qLogNEI,
        qLogNParEGO,
        qNEHVI,
        qNEI,
    )

    if isinstance(
        acquisition,
        (
            qExpectedImprovement,
            qNEI,
            qNEHVI,
        ),
    ):
        return "linear"
    if isinstance(
        acquisition,
        (
            qLogEI,
            qLogNEI,
            qLogNEHVI,
            qLogNParEGO,
        ),
    ):
        return "log"

    raise IncompatibilityError(
        f"The acquisition function '{acquisition.__class__.__name__}' does not "
        "declare a supported nonnegative gain scale for multi-step action scoring. "
        "Use a supported Monte Carlo gain acquisition."
    )


def _score_actions(
    acquisition_values: Tensor,
    success_probabilities: Tensor,
    cost: float,
    *,
    scale: ScoreScale,
    allow_negative_gains: bool = False,
) -> Tensor:
    """Apply local execution risk and cost once to acquisition gains.

    Args:
        acquisition_values: Raw acquisition gains, in linear or log form.
        success_probabilities: Local next-measurement success probabilities,
            with exactly the same shape, dtype, and device as the gains.
        cost: Positive cost of the next node measurement.
        scale: Representation of the raw gains.
        allow_negative_gains: Permit finite signed linear estimates for gains
            whose implementation explicitly declares them.

    Returns:
        The risk- and cost-adjusted gains. A probability of exactly zero maps to
        ``-inf`` and is therefore unselectable, with no epsilon substitution.

    Raises:
        IncompatibilityError: If values, probability, cost, scale, or shapes are
            invalid.
    """
    import torch

    if scale not in ("linear", "log"):
        raise IncompatibilityError(f"Unsupported acquisition score scale: {scale!r}.")
    if acquisition_values.shape != success_probabilities.shape:
        raise IncompatibilityError(
            "Acquisition values and success probabilities must have identical "
            f"shapes, got {tuple(acquisition_values.shape)} and "
            f"{tuple(success_probabilities.shape)}."
        )
    if (
        acquisition_values.device != success_probabilities.device
        or acquisition_values.dtype != success_probabilities.dtype
    ):
        raise IncompatibilityError(
            "Acquisition values and success probabilities must share dtype and device."
        )
    if not math.isfinite(cost) or cost <= 0:
        raise IncompatibilityError("Action cost must be finite and strictly positive.")

    if scale == "linear":
        if not torch.isfinite(acquisition_values).all():
            raise IncompatibilityError("Linear acquisition gains must be finite.")
        if not allow_negative_gains and (acquisition_values < 0).any():
            raise IncompatibilityError(
                "Linear gain acquisitions must not return negative values."
            )
        scores = acquisition_values * success_probabilities / cost
    else:
        if (
            torch.isnan(acquisition_values).any()
            or torch.isposinf(acquisition_values).any()
        ):
            raise IncompatibilityError(
                "Log acquisition gains may be finite or -inf, but not NaN or +inf."
            )
        # Avoid evaluating log(0): even an unselected torch.where branch can
        # otherwise introduce infinite gradients. This is not an epsilon rescue;
        # the zero-probability branch is still represented by exactly -inf.
        safe_probabilities = torch.where(
            success_probabilities > 0,
            success_probabilities,
            torch.ones_like(success_probabilities),
        )
        scores = acquisition_values + safe_probabilities.log() - math.log(cost)

    return torch.where(
        success_probabilities > 0,
        scores,
        torch.full_like(scores, -torch.inf),
    )


def _reduce_score_branches(branch_values: Tensor, *, scale: ScoreScale) -> Tensor:
    """Average pending branches in the gain's own representation.

    Args:
        branch_values: Raw acquisition values with a nonempty leading branch axis.
        scale: Representation of the acquisition values.

    Returns:
        The arithmetic mean of linear gains or log of the arithmetic mean of
        log-represented gains.

    Raises:
        IncompatibilityError: If branch values are empty, malformed, or non-finite
            outside the valid ``-inf`` log-gain case.
    """
    import torch

    if branch_values.ndim < 1 or branch_values.shape[0] < 1:
        raise IncompatibilityError(
            "At least one pending acquisition branch is required."
        )
    if scale == "linear":
        if not torch.isfinite(branch_values).all():
            raise IncompatibilityError("Linear pending gains must be finite.")
        return branch_values.mean(dim=0)
    if scale == "log":
        if torch.isnan(branch_values).any() or torch.isposinf(branch_values).any():
            raise IncompatibilityError(
                "Log pending gains may be finite or -inf, but not NaN or +inf."
            )
        return torch.logsumexp(branch_values, dim=0) - math.log(branch_values.shape[0])
    raise IncompatibilityError(f"Unsupported acquisition score scale: {scale!r}.")


def _make_tensor_score_acquisition(
    *,
    model: Model,
    raw_score: Callable[[Tensor], Tensor],
    success_probability: Callable[[Tensor], Tensor],
    cost: float,
    scale: ScoreScale,
    feasibility_threshold: float = 0.0,
    allow_negative_gains: bool = False,
) -> BoAcquisitionFunction:
    """Create a lazy BoTorch adapter that weights a raw score before optimization.

    The local success probability is evaluated before the expensive raw-score
    (fantasy) computation. Candidates below the configured threshold, and
    candidates with probability exactly zero, never invoke the raw-score
    function.

    Args:
        model: The model supplied to the native BoTorch acquisition base class.
        raw_score: The differentiable, unweighted score function.
        success_probability: The differentiable local action success predictor.
        cost: Cost of the next node measurement.
        scale: Representation of the raw score.
        feasibility_threshold: Minimum positive local success probability.
        allow_negative_gains: Permit signed linear gains whose implementation
            explicitly declares them.

    Returns:
        A BoTorch acquisition function evaluating the shared weighted score.

    Raises:
        IncompatibilityError: If the threshold is invalid.
    """
    from attrs import define, setters
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from botorch.utils.transforms import t_batch_mode_transform

    if not math.isfinite(feasibility_threshold) or feasibility_threshold < 0:
        raise IncompatibilityError(
            "The local feasibility threshold must be finite and nonnegative."
        )

    @define(eq=False, repr=False, slots=False, init=False, on_setattr=setters.NO_OP)
    class _TensorScoreAcquisition(BoAcquisitionFunction):
        """Adapt a raw tensor score to shared multi-step action weighting."""

        @override
        @cast(
            "Callable[[Callable[..., Tensor]], Callable[..., Tensor]]",
            t_batch_mode_transform(expected_q=1),
        )
        def forward(self, X: Tensor) -> Tensor:
            import torch

            probabilities = success_probability(X)
            if (
                not torch.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or (probabilities > 1).any()
            ):
                raise IncompatibilityError(
                    "Local success probabilities must be finite values in [0, 1]."
                )
            allowed = probabilities > 0
            if feasibility_threshold > 0:
                allowed = allowed & (probabilities >= feasibility_threshold)
            if not bool(allowed.any()):
                return torch.full(
                    probabilities.shape,
                    -torch.inf,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
            if bool(allowed.all()):
                return _score_actions(
                    raw_score(X),
                    probabilities,
                    cost,
                    scale=scale,
                    allow_negative_gains=allow_negative_gains,
                )

            subset = X[allowed.unsqueeze(-1).unsqueeze(-1).expand_as(X)].reshape(
                -1, 1, X.shape[-1]
            )
            scores = _score_actions(
                raw_score(subset),
                probabilities[allowed],
                cost,
                scale=scale,
                allow_negative_gains=allow_negative_gains,
            )
            result = torch.full(
                probabilities.shape,
                -torch.inf,
                dtype=scores.dtype,
                device=scores.device,
            )
            result[allowed] = scores
            return result

    return _TensorScoreAcquisition(model=model)


def _make_node_actions(
    dag: MultiStepDAG,
    plans: tuple[_ExecutablePlan, ...],
    pending: tuple[_PendingAction, ...] = (),
) -> tuple[_ExecutablePlan, ...]:
    """Enumerate fixed local input problems from executable plans.

    Continuous own inputs stay free for optimization. Discrete own inputs and
    observed parent outputs are fixed. Each problem retains its concrete
    experiment context and the remaining cost of its continuation.

    Args:
        dag: The DAG topology.
        plans: The executable actions prepared from the current campaign state.
        pending: Already reserved actions. An identical action is not offered
            again.

    Returns:
        The local problems, in plan and input order.

    """
    problems: list[_ExecutablePlan] = []
    for plan in plans:
        name = plan.node
        node = dag.node_by_name[name]
        own_inputs = (
            node.searchspace.discrete.exp_rep
            if node.inputs and not node.searchspace.discrete.is_empty
            else pd.DataFrame(index=[0])
        )
        own_rows: list[dict[str, Any]] = (
            [
                {str(key): value for key, value in row.items()}
                for row in own_inputs.to_dict("records")
            ]
            if len(own_inputs.columns)
            else [{} for _ in range(len(own_inputs))]
        )
        for own in own_rows:
            fixed_inputs = {**own, **plan.inputs}
            if contains_action(pending, node=name, inputs=fixed_inputs, dag=dag):
                continue
            problems.append(evolve(plan, inputs=fixed_inputs))
    return tuple(problems)


def _exclude_pending_local_actions(
    model: MultiStepModel,
    node: MultiStepNode,
    inputs: Tensor,
    probabilities: Tensor,
    pending: tuple[_PendingAction, ...],
) -> Tensor:
    """Exclude pending local inputs before evaluating an acquisition value."""
    import torch

    columns = model.get_node_model_columns(node.name)
    index = {column: i for i, column in enumerate(columns)}
    tolerances = inputs.new_zeros(len(columns))
    for parameter in node.inputs:
        if parameter.is_numerical:
            for column in parameter.comp_rep_columns:
                tolerances[index[column]] = (
                    float(getattr(parameter, "tolerance", 0.0)) + 1e-12
                )

    for action in pending:
        if action.node != node.name:
            continue
        pending_inputs = model.transform_node_inputs(
            node.name, pd.DataFrame([dict(action.inputs)])
        ).to(inputs)
        matches = (
            (inputs - pending_inputs.reshape(*([1] * (inputs.ndim - 1)), -1))
            .abs()
            .le(tolerances)
            .all(dim=-1)
            .squeeze(-1)
        )
        probabilities = torch.where(
            matches, torch.zeros_like(probabilities), probabilities
        )
    return probabilities


def _optimize_node_actions(
    *,
    model: MultiStepModel,
    problems: tuple[_ExecutablePlan, ...],
    gain_function: Callable[[_ActionContext, MultiStepNode, Tensor], Tensor],
    gain_scale: ScoreScale,
    supports_signed_gains: bool,
    feasibility_threshold: float,
    pending: tuple[_PendingAction, ...],
) -> tuple[_PendingAction, Tensor] | None:
    """Optimize one action context's discrete or mixed local search.

    The fully weighted score is optimized, so both the raw gain and the local
    success probability influence the selected candidate.

    Args:
        model: The fitted network model.
        problems: The executable local problems sharing one experiment context.
        gain_function: The acquisition's raw node-gain evaluator.
        gain_scale: Representation of the raw gain.
        supports_signed_gains: Whether the acquisition may return finite negative
            linear gains.
        feasibility_threshold: Minimum local success probability for an
            admissible candidate.
        pending: Already reserved actions that cannot be recommended again.

    Returns:
        The optimized action and its weighted tensor score, or ``None`` if every
        candidate is excluded by the local success threshold.

    Raises:
        IncompatibilityError: If the optimizer returns invalid values, a candidate
            outside the parameter bounds, or changes fixed inputs.
    """
    import torch
    from botorch.exceptions.warnings import OptimizationWarning
    from botorch.optim import optimize_acqf_discrete, optimize_acqf_mixed

    node = model.dag.node_by_name[problems[0].node]
    model_columns = model.get_node_model_columns(node.name)
    free_parameters = {
        parameter.name: parameter
        for parameter in node.inputs
        if isinstance(parameter, NumericalContinuousParameter)
    }
    local_frame = pd.DataFrame([problem.inputs for problem in problems]).assign(
        **{name: parameter.bounds.lower for name, parameter in free_parameters.items()}
    )
    choices = model.transform_node_inputs(node.name, local_frame)
    acquisition = _make_tensor_score_acquisition(
        model=model,
        raw_score=lambda inputs: gain_function(
            _ActionContext(problems[0].experiment_id, problems[0].parent_refs),
            node,
            inputs,
        ),
        success_probability=lambda inputs: _exclude_pending_local_actions(
            model,
            node,
            inputs,
            model._predict_feasibility_tensor(node.name, inputs),
            pending,
        ),
        cost=problems[0].remaining_cost,
        scale=gain_scale,
        feasibility_threshold=feasibility_threshold,
        allow_negative_gains=supports_signed_gains,
    )

    fixed_indices = [
        index
        for index, column in enumerate(model_columns)
        if column not in free_parameters
    ]
    if free_parameters:
        bounds = torch.stack((choices.amin(dim=0), choices.amax(dim=0)))
        for name, parameter in free_parameters.items():
            bounds[:, model_columns.index(name)] = choices.new_tensor(
                (parameter.bounds.lower, parameter.bounds.upper)
            )
        fixed_features = [
            {index: row[index].item() for index in fixed_indices} for row in choices
        ]
        with torch.random.fork_rng(), warnings.catch_warnings():
            torch.manual_seed(_OPTIMIZER_SEED)
            # BoTorch otherwise returns partial results when some fixed
            # feature combinations fail. A recommendation must be all-or-error.
            warnings.filterwarnings(
                "error", "Candidate generation failed for", OptimizationWarning
            )
            candidate, score = optimize_acqf_mixed(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                fixed_features_list=fixed_features,
                num_restarts=_N_RESTARTS,
                raw_samples=_N_RAW_SAMPLES,
                options={"seed": _OPTIMIZER_SEED},
            )
        if not ((candidate >= bounds[0]) & (candidate <= bounds[1])).all():
            raise IncompatibilityError(
                f"The optimizer returned a candidate outside the bounds of "
                f"node '{node.name}'."
            )
    else:
        candidate, score = optimize_acqf_discrete(
            acq_function=acquisition, q=1, choices=choices
        )

    if torch.isnan(score).any() or torch.isposinf(score).any():
        raise IncompatibilityError(
            f"The optimizer returned an invalid score for node '{node.name}'."
        )
    if not torch.isfinite(candidate).all():
        raise IncompatibilityError(
            f"The optimizer's candidate for node '{node.name}' is not finite."
        )
    if torch.isneginf(score).all():
        return None

    matches = (choices[:, fixed_indices] == candidate[0, fixed_indices]).all(dim=-1)
    if not matches.any():
        raise IncompatibilityError(
            f"The optimizer changed fixed inputs of node '{node.name}'."
        )
    selected_problem = problems[int(matches.nonzero()[0].item())]
    selected_inputs = dict(selected_problem.inputs)
    selected_inputs.update(
        {
            name: candidate[0, model_columns.index(name)].item()
            for name in free_parameters
        }
    )
    return (
        _PendingAction(
            node=node.name,
            inputs=selected_inputs,
            parent_refs=selected_problem.parent_refs,
            experiment_id=selected_problem.experiment_id,
        ),
        score,
    )


def select_scored_actions(
    scored: tuple[tuple[_PendingAction, Tensor], ...],
    *,
    batch_size: int,
    pending: tuple[_PendingAction, ...],
    dag: MultiStepDAG,
) -> list[_PendingAction]:
    """Select the highest-scoring distinct pending-free actions.

    Args:
        scored: One weighted score per optimized action context.
        batch_size: The number of actions requested.
        pending: Already reserved actions that must not be duplicated.
        dag: The DAG topology.

    Returns:
        The selected actions in descending score order.

    Raises:
        IncompatibilityError: If a score is invalid or fewer distinct
            pending-free actions than requested are available.
        InfeasibilityError: If every optimized action was excluded by the local
            success threshold.
    """
    options: list[tuple[float, int, _PendingAction]] = []
    rejected_for_risk = False
    for index, (action, score) in enumerate(scored):
        value = float(score.detach().cpu().reshape(-1)[0])
        if math.isnan(value) or value == math.inf:
            raise IncompatibilityError(
                f"The graph-aware acquisition returned an invalid score "
                f"{value!r} for node '{action.node}'."
            )
        if math.isinf(value) and value < 0:
            rejected_for_risk = True
            continue
        options.append((value, index, action))

    if not options:
        if rejected_for_risk:
            raise InfeasibilityError(
                "No executable action reaches the local success threshold."
            )
        raise IncompatibilityError(
            "The graph-aware acquisition returned no executable action."
        )

    options.sort(key=lambda item: (-item[0], item[1]))
    selected: list[_PendingAction] = []
    contexts: set[tuple[str, str]] = set()
    for _, _, action in options:
        context = (action.experiment_id, action.node)
        if context in contexts:
            continue
        if contains_action(
            (*pending, *selected),
            node=action.node,
            inputs=action.inputs,
            dag=dag,
        ):
            continue
        contexts.add(context)
        selected.append(action)
        if len(selected) == batch_size:
            break

    if len(selected) < batch_size:
        raise IncompatibilityError(
            f"Only {len(selected)} distinct pending-free executable actions "
            f"are available, but {batch_size=} were requested. Report pending "
            "measurements or reduce the batch size."
        )
    return selected


def recommend_graph_aware(
    model: MultiStepModel,
    plans: tuple[_ExecutablePlan, ...],
    pending: tuple[_PendingAction, ...] = (),
    *,
    acquisition: GraphAwareAcquisitionProtocol,
    batch_size: int,
    feasibility_threshold: float = 1e-6,
) -> list[_PendingAction]:
    """Score and select executable actions with a graph-aware acquisition.

    The executable action set is derived from the plans independently of the
    acquisition. The acquisition prepares its global gain context once and then
    supplies raw local node gains, which are weighted and optimized centrally.

    Args:
        model: The fitted network model.
        plans: The executable actions prepared from the current campaign state.
        pending: Already reserved actions. Their outcomes are fantasized, and an
            identical action is not offered again.
        acquisition: The graph-aware gain contract implementation.
        batch_size: The number of actions requested.
        feasibility_threshold: Minimum local success probability for an
            admissible candidate.

    Returns:
        The selected actions in descending weighted-score order.

    Raises:
        IncompatibilityError: If no executable action is available, a selected
            action duplicates pending state, or fewer distinct actions than
            requested are available.
        InfeasibilityError: If no executable action reaches the local success
            threshold.
    """
    problems = _make_node_actions(model.dag, plans, pending)
    if not problems:
        raise IncompatibilityError(
            "The graph-aware acquisition found no executable local problem. Every "
            "node requires real observations for all parent inputs."
        )

    gain_function = acquisition.make_gain_function(model, pending)
    grouped: dict[tuple[Any, ...], list[_ExecutablePlan]] = {}
    for problem in problems:
        grouped.setdefault(
            (
                problem.node,
                problem.experiment_id,
                tuple(problem.parent_refs.items()),
                problem.remaining_cost,
            ),
            [],
        ).append(problem)

    scored: list[tuple[_PendingAction, Tensor]] = []
    for group in grouped.values():
        optimized = _optimize_node_actions(
            model=model,
            problems=tuple(group),
            gain_function=gain_function,
            gain_scale=acquisition.gain_scale,
            supports_signed_gains=acquisition.supports_signed_gains,
            feasibility_threshold=feasibility_threshold,
            pending=pending,
        )
        if optimized is None:
            continue
        scored.append(optimized)

    if not scored:
        raise InfeasibilityError(
            "No executable action reaches the local success threshold "
            f"{feasibility_threshold}."
        )
    return select_scored_actions(
        tuple(scored),
        batch_size=batch_size,
        pending=pending,
        dag=model.dag,
    )


def _project_candidate(
    plan: _ExecutablePlan, candidate: pd.Series, *, dag: MultiStepDAG
) -> _PendingAction:
    """Project one global configuration onto the plan's executable node.

    Args:
        plan: The executable node context.
        candidate: One global DAG configuration.
        dag: The DAG topology.

    Returns:
        The concrete local action carrying the node's own parameters and the
        plan's real parent outputs.
    """
    node = dag.node_by_name[plan.node]
    inputs = {parameter.name: candidate[parameter.name] for parameter in node.inputs}
    inputs.update(plan.inputs)
    return _PendingAction(
        node=node.name,
        inputs=inputs,
        parent_refs=plan.parent_refs,
        experiment_id=plan.experiment_id,
    )


def _make_stock_acquisition_branches(
    model: MultiStepModel,
    pending: tuple[_PendingAction, ...],
    acquisition_function: AcquisitionFunction,
    measurements: pd.DataFrame,
) -> tuple[BoAcquisitionFunction, ...]:
    """Prepare stock acquisition functions over the fitted network model."""
    from attrs import evolve

    from baybe.objectives.desirability import DesirabilityObjective

    from .model import _AcquisitionSurrogateAdapter

    objective = model.dag.objective
    if isinstance(objective, DesirabilityObjective) and objective.as_pre_transformation:
        # The network models raw node outputs; apply every target transformation
        # and scalarization at acquisition time.
        objective = evolve(objective, as_pre_transformation=False)

    conditioned_models = model.condition_on_pending(
        pending,
        num_fantasies=_PENDING_FANTASIES,
        sampling_seed=_PENDING_FANTASY_SEED,
    )
    if not conditioned_models:
        raise IncompatibilityError(
            "Pending conditioning returned no stock acquisition branches."
        )

    return tuple(
        acquisition_function.to_botorch(
            _AcquisitionSurrogateAdapter(conditioned),
            model.dag.full_searchspace,
            objective,
            measurements,
        )
        for conditioned in conditioned_models
    )


def _stock_fixed_parameters(
    plan: _ExecutablePlan,
    experiments: Mapping[str, Any],
    data: MultiStepData,
    dag: MultiStepDAG,
) -> dict[str, Any]:
    """Return global parameters already fixed by the concrete process context."""
    if plan.is_new:
        return {}
    experiment = experiments[plan.experiment_id]
    fixed: dict[str, Any] = {}
    for node_name, ref in experiment.bindings.items():
        measurement = data.get_evaluation(dag, ref)
        for parameter in dag.node_by_name[node_name].inputs:
            fixed[parameter.name] = measurement.inputs[parameter.name]
    return fixed


def _stock_discrete_candidates(
    searchspace: SearchSpace, fixed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Enumerate discrete global assignments compatible with a process context."""
    if searchspace.discrete.is_empty:
        return [{}]
    frame = searchspace.discrete.exp_rep
    for parameter in searchspace.discrete.parameters:
        if parameter.name not in fixed:
            continue
        value = fixed[parameter.name]
        if parameter.is_numerical:
            matches = np.isclose(
                frame[parameter.name].to_numpy(dtype=float),
                float(value),
                rtol=0.0,
                atol=float(getattr(parameter, "tolerance", 0.0)) + 1e-12,
            )
        else:
            matches = frame[parameter.name].eq(value).to_numpy(dtype=bool)
        frame = frame.loc[matches]
    return [
        {str(key): value for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _stock_action_probability(
    model: MultiStepModel,
    plan: _ExecutablePlan,
    global_inputs: Tensor,
    pending: tuple[_PendingAction, ...],
) -> Tensor:
    """Predict and gate local risk using one optimized global configuration."""
    import torch

    node = model.dag.node_by_name[plan.node]
    columns = model.get_node_model_columns(node.name)
    global_indices = {
        name: index
        for index, name in enumerate(model.dag.full_searchspace.comp_rep_columns)
    }
    history_comp: dict[str, float] = {}
    for feature in model.dag.context._features(node):
        value = plan.inputs[feature.name]
        if feature.parameter is None:
            history_comp[feature.name] = float(value)
        else:
            parameter = feature.make_parameter()
            if isinstance(parameter, DiscreteParameter):
                encoded = parameter.transform(pd.Series([value], name=parameter.name))
                history_comp.update(
                    {str(k): float(v) for k, v in encoded.iloc[0].items()}
                )
            else:
                history_comp[parameter.name] = float(value)
    local_columns: list[Tensor] = []
    for column in columns:
        if column in history_comp:
            local_columns.append(
                global_inputs.new_full(
                    (*global_inputs.shape[:-1], 1), history_comp[column]
                )
            )
        else:
            local_columns.append(
                global_inputs[..., global_indices[column]].unsqueeze(-1)
            )
    local_inputs = torch.cat(local_columns, dim=-1)
    probabilities = model._predict_feasibility_tensor(node.name, local_inputs)

    return _exclude_pending_local_actions(
        model, node, local_inputs, probabilities, pending
    )


def _optimize_stock_context(
    *,
    model: MultiStepModel,
    plan: _ExecutablePlan,
    discrete_candidate: Mapping[str, Any],
    fixed_parameters: Mapping[str, Any],
    branches: tuple[BoAcquisitionFunction, ...],
    score_scale: ScoreScale,
    pending: tuple[_PendingAction, ...],
    feasibility_threshold: float,
) -> tuple[pd.Series, Tensor] | None:
    """Optimize the shared risk/cost score over one executable global context."""
    import torch
    from botorch.exceptions.warnings import OptimizationWarning
    from botorch.optim import optimize_acqf_mixed

    searchspace = model.dag.full_searchspace
    reference = next(model.parameters())
    bounds = torch.as_tensor(searchspace.comp_rep_bounds.to_numpy()).T.to(reference)
    column_index = {
        name: index for index, name in enumerate(searchspace.comp_rep_columns)
    }
    discrete = pd.DataFrame([dict(discrete_candidate)])
    discrete_comp = (
        searchspace.discrete.transform(discrete).iloc[0].to_dict()
        if not searchspace.discrete.is_empty
        else {}
    )
    fixed_features = {
        column_index[str(column)]: float(value)
        for column, value in discrete_comp.items()
    }
    fixed_features.update(
        {
            column_index[name]: float(value)
            for name, value in fixed_parameters.items()
            if name in column_index and name in searchspace.continuous.parameter_names
        }
    )

    def raw_score(inputs: Tensor) -> Tensor:
        """Average pending branches in the acquisition's declared scale."""
        return _reduce_score_branches(
            torch.stack([branch(inputs) for branch in branches]), scale=score_scale
        )

    acquisition = _make_tensor_score_acquisition(
        model=branches[0].model,
        raw_score=raw_score,
        success_probability=lambda inputs: _stock_action_probability(
            model, plan, inputs, pending
        ),
        cost=plan.remaining_cost,
        scale=score_scale,
        feasibility_threshold=feasibility_threshold,
    )

    free_continuous = [
        parameter.name
        for parameter in searchspace.continuous.parameters
        if parameter.name not in fixed_parameters
    ]
    if free_continuous:
        with torch.random.fork_rng(), warnings.catch_warnings():
            torch.manual_seed(_OPTIMIZER_SEED)
            warnings.filterwarnings(
                "error", "Candidate generation failed for", OptimizationWarning
            )
            candidate, score = optimize_acqf_mixed(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                fixed_features_list=[fixed_features],
                num_restarts=_N_RESTARTS,
                raw_samples=_N_RAW_SAMPLES,
                options={"seed": _OPTIMIZER_SEED},
            )
    else:
        candidate = bounds[0].new_zeros((1, 1, bounds.shape[-1]))
        for column, value in fixed_features.items():
            candidate[..., column] = value
        score = acquisition(candidate)

    if (
        not torch.isfinite(candidate).all()
        or torch.isnan(score).any()
        or torch.isposinf(score).any()
    ):
        raise IncompatibilityError(
            f"Stock acquisition optimization returned invalid values for node "
            f"'{plan.node}'."
        )
    if torch.isneginf(score).all():
        return None

    values = dict(discrete_candidate)
    values.update(
        {
            parameter.name: candidate[0, 0, column_index[parameter.name]].item()
            for parameter in searchspace.continuous.parameters
        }
    )
    return pd.Series(values), score.reshape(-1)[0]


def _next_extra_identifier(
    first_new_id: int, plans: tuple[_ExecutablePlan, ...]
) -> int:
    """Return the first identifier available for additional planned actions.

    Args:
        first_new_id: The first identifier available without the plans.
        plans: The executable plans already carrying prepared identifiers.

    Returns:
        One larger than every identifier reserved by the plans or known state.
    """
    return max(
        [first_new_id]
        + [_identifier_number(plan.experiment_id) + 1 for plan in plans if plan.is_new]
    )


def recommend_stock(
    model: MultiStepModel,
    data: MultiStepData,
    experiments: Mapping[str, Any],
    plans: tuple[_ExecutablePlan, ...],
    pending: tuple[_PendingAction, ...],
    *,
    acquisition_function: AcquisitionFunction,
    score_scale: ScoreScale,
    batch_size: int,
    feasibility_threshold: float = 1e-6,
    first_new_id: int = 1,
    execution_mode: Literal["process", "functional"] = "process",
) -> list[_PendingAction]:
    """Optimize stock acquisitions over executable actions using shared scoring.

    A stock acquisition retains its global-configuration semantics, while each
    executable context constrains already measured parameters and projects the
    optimized configuration onto its concrete local action. Pending fantasy
    branches, local feasibility, cost weighting, and final selection use the same
    gain and selector contracts as graph-aware acquisitions.

    This projection is an orchestration rule, not an unchanged local
    information-value acquisition.

    Args:
        model: The fitted network model.
        data: The real measurement store.
        experiments: All known experiment snapshots.
        plans: The executable node contexts.
        pending: The currently reserved local actions, whose outcomes are
            fantasized when scoring.
        acquisition_function: The resolved stock acquisition function,
            including its actual default for this objective.
        score_scale: The acquisition function's declared gain representation.
        batch_size: The number of node actions to recommend.
        feasibility_threshold: Minimum local success probability for an
            admissible candidate.
        first_new_id: The first identifier available for a new experiment.
        execution_mode: Whether global reference rows must be process-complete or
            may be composed from successful real references across experiments.

    Returns:
        The selected executable actions in descending score order.

    Raises:
        IncompatibilityError: If the DAG search space is unsupported, required
            reference measurements are unavailable, an acquisition score is
            invalid, or fewer distinct executable actions than requested exist.
        InfeasibilityError: If no candidate reaches the local success threshold.
    """
    searchspace = model.dag.full_searchspace
    if searchspace.is_constrained:
        raise IncompatibilityError(
            "Stock acquisitions do not support constrained DAG search spaces."
        )
    if not plans:
        raise IncompatibilityError("No executable node remains in any open experiment.")

    measurements = data.complete_measurement_frame(
        model.dag, execution_mode=execution_mode
    )
    if measurements.empty:
        raise IncompatibilityError(
            "The selected stock acquisition requires at least one complete, "
            "reference-consistent set of real DAG measurements."
        )

    branches = _make_stock_acquisition_branches(
        model, pending, acquisition_function, measurements
    )
    next_identifier = _next_extra_identifier(first_new_id, plans)
    scored: list[tuple[_PendingAction, Tensor]] = []
    for plan in plans:
        fixed = _stock_fixed_parameters(plan, experiments, data, model.dag)
        has_new_action = False
        for discrete in _stock_discrete_candidates(searchspace, fixed):
            optimized = _optimize_stock_context(
                model=model,
                plan=plan,
                discrete_candidate=discrete,
                fixed_parameters=fixed,
                branches=branches,
                score_scale=score_scale,
                pending=pending,
                feasibility_threshold=feasibility_threshold,
            )
            if optimized is None:
                continue
            candidate, score = optimized
            action = _project_candidate(plan, candidate, dag=model.dag)
            if plan.is_new and has_new_action:
                action = _PendingAction(
                    node=action.node,
                    inputs=action.inputs,
                    parent_refs=action.parent_refs,
                    experiment_id=f"exp-{next_identifier}",
                )
                next_identifier += 1
            scored.append((action, score))
            has_new_action = True

    if not scored:
        raise InfeasibilityError(
            "No executable action reaches the local success threshold "
            f"{feasibility_threshold}."
        )
    return select_scored_actions(
        tuple(scored), batch_size=batch_size, pending=pending, dag=model.dag
    )


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
