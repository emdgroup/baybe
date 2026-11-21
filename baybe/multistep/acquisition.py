"""Acquisition policy for partial-evaluation function networks."""

from __future__ import annotations

import gc
import math
import warnings
from collections.abc import Callable, Mapping
from itertools import groupby
from operator import attrgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

import pandas as pd
from attrs import Attribute, define, field
from attrs.validators import and_, ge, gt, instance_of, le, not_
from typing_extensions import override

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import IncompatibilityError, ModelNotTrainedError
from baybe.objectives.base import Objective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget

from ._experiments import _EvaluationRef, _PendingAction, contains_action
from ._scoring import _make_tensor_score_acquisition as _make_action_score_acquisition

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from torch import Tensor

    from .dag import MultiStepDAG
    from .data import MultiStepData
    from .model import MultiStepModel
    from .node import MultiStepNode


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only copy of a mapping."""
    return MappingProxyType(dict(value))


@define(frozen=True)
class _LocalProblem:
    """A node and its fixed discrete and observed-parent inputs."""

    node: str = field()
    """The node to evaluate."""

    fixed_inputs: Mapping[str, Any] = field(converter=_freeze_mapping)
    """The discrete own inputs and observed parent outputs kept fixed."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """The concrete parent evaluations supplying the parent outputs."""


def _condition_on_pending(
    model: MultiStepModel,
    pending: tuple[_PendingAction, ...],
    *,
    num_fantasies: int,
    sampling_seed: int,
) -> tuple[MultiStepModel, ...]:
    """Create one conditioned network model per joint pending fantasy.

    Pending executions of a node are sampled jointly so that their GP
    correlations and observation noise are retained. Nodes are independent
    given their local data, so joint network fantasies combine node samples by
    index instead of enumerating their Cartesian product.

    Args:
        model: The fitted real network model.
        pending: The reserved local executions.
        num_fantasies: Number of joint pending outcome paths.
        sampling_seed: Base seed for the deterministic predictive draws.

    Returns:
        One conditioned model per pending fantasy. Without pending actions,
        the original model is returned as the only element.

    Raises:
        IncompatibilityError: If a pending node is unknown or a local input
            cannot be transformed.
    """
    if not pending:
        return (model,)

    grouped: dict[str, list[_PendingAction]] = {}
    for action in pending:
        grouped.setdefault(action.node, []).append(action)

    node_inputs: dict[str, Tensor] = {}
    node_samples: dict[str, Tensor] = {}
    for index, (node_name, actions) in enumerate(grouped.items()):
        node = model.dag.node_by_name.get(node_name)
        if node is None:
            raise IncompatibilityError(
                f"Pending action references unknown node '{node_name}'."
            )
        frame = pd.DataFrame(
            [dict(action.inputs) for action in actions],
            columns=list(node.model_input_names),
        )
        inputs = model.transform_node_inputs(node_name, frame).unsqueeze(-2)
        node_inputs[node_name] = inputs
        node_samples[node_name] = model.sample_node_observations(
            node_name,
            inputs,
            num_fantasies,
            sampling_seed + 2 * index,
        )

    return tuple(
        model.condition_on_node_observations(
            {
                node_name: (
                    node_inputs[node_name],
                    node_samples[node_name][fantasy_index],
                )
                for node_name in grouped
            }
        )
        for fantasy_index in range(num_fantasies)
    )


@define(frozen=True)
class PartialKnowledgeGradientFN(AcquisitionFunction):
    """Partial-knowledge-gradient policy for function networks.

    The policy selects one node and its complete local input. It weights the
    expected gain of the best terminal decision by the local measurement's
    success probability and divides once by that measurement's cost, instead of
    maximizing a local node output. Continuous node-owned
    inputs are optimized against a finite terminal reference set. The
    configuration is only valid for
    :class:`~baybe.multistep.campaign.MultiStepCampaign`; it is not a stock
    BoTorch point acquisition function.

    The policy owns action generation, terminal utility evaluation, scoring, and
    optimization. The network model provides training, input transformation,
    sampling, and fantasy conditioning without implementing policy decisions.

    Nodes with feasibility conditions are supported through independently learned
    success-probability models. A terminal decision whose predicted joint success
    probability falls below :attr:`feasibility_threshold` is treated as failed and
    yields no improvement over the best observed successful utility, which acts as
    the outside option of the policy.
    """

    abbreviation: ClassVar[str] = "pKGFN"
    """An alternative name for type resolution."""

    supports_multi_output: ClassVar[bool] = False
    """Whether this acquisition function can handle models with multiple outputs."""

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

    n_restarts: int = field(default=10, validator=[_integer_validator, gt(0)])
    """The number of multi-start optimization restarts for continuous inputs."""

    n_raw_samples: int = field(default=64, validator=[_integer_validator, gt(0)])
    """The number of raw samples used to initialize continuous optimization."""

    max_terminal_candidates: int = field(
        default=256, validator=[_integer_validator, gt(0)]
    )
    """The maximum number of terminal configurations that may be enumerated."""

    max_node_candidates: int = field(default=256, validator=[_integer_validator, gt(0)])
    """The maximum number of measurement actions that may be enumerated."""

    feasibility_threshold: float = field(
        default=0.5, validator=[instance_of(float), ge(0.0), le(1.0)], kw_only=True
    )
    """Minimum joint success probability of a terminal decision.

    If the learned network success probability of a terminal decision falls below
    this value, the decision is treated as failed and its utility falls back to the
    best observed successful utility. The threshold only applies when at least one
    node configures feasibility conditions.
    """

    def __attrs_post_init__(self) -> None:
        """Validate the relationship between optimizer sample counts.

        Raises:
            ValueError: If the raw sample count is smaller than the restart count.
        """
        if self.n_raw_samples < self.n_restarts:
            raise ValueError(
                f"'n_raw_samples' must be greater than or equal to 'n_restarts', "
                f"got {self.n_raw_samples} and {self.n_restarts}."
            )

    @staticmethod
    def _compute_pkgfn_gain(
        fantasy_terminal_means: Tensor,
        baseline: Tensor,
        fantasy_weights: Tensor | None = None,
    ) -> Tensor:
        """Reduce fantasy terminal means to an unweighted p-KGFN gain estimate.

        Args:
            fantasy_terminal_means: Network-averaged utilities with shape
                ``(num_fantasies, num_terminal_candidates, batch...)``.
            baseline: The baseline utility of the current data. A scalar applies
                to every fantasy; a tensor provides one baseline per fantasy and
                must broadcast to the best-terminal means.
            fantasy_weights: Optional outcome probabilities matching the fantasy
                and candidate batch axes. Without weights, use a uniform average.

        Returns:
            One finite, potentially signed gain estimate per candidate batch.

        Raises:
            IncompatibilityError: If the input shape, baseline, weights, or gain
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

        gains = best - baseline
        if fantasy_weights is None:
            expected = gains.mean(dim=0)
        else:
            if (
                fantasy_weights.shape[0] != gains.shape[0]
                or not torch.isfinite(fantasy_weights).all()
            ):
                raise IncompatibilityError(
                    "The fantasy weights must be finite and match the fantasy axis."
                )
            expanded = fantasy_weights.reshape(
                fantasy_weights.shape[0],
                *([1] * (gains.dim() - fantasy_weights.dim())),
                *fantasy_weights.shape[1:],
            )
            try:
                expected = (gains * expanded).sum(dim=0)
            except RuntimeError as ex:
                raise IncompatibilityError(
                    "The fantasy weights do not broadcast to the gain shape."
                ) from ex
        if not torch.isfinite(expected).all():
            raise IncompatibilityError("The p-KGFN gain estimate is not finite.")
        return expected

    @staticmethod
    def _compute_terminal_means_from_network_samples(
        dag: MultiStepDAG, samples: Tensor
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
            samples.shape[-2:] != (1, len(dag.final_target_names))
            or not torch.isfinite(samples).all()
        ):
            raise IncompatibilityError(
                "Terminal samples must be finite, have q=1, and match the DAG outputs."
            )

        utilities = dag.objective.to_botorch()(samples, None)
        if utilities.shape[-1] != 1 or not torch.isfinite(utilities).all():
            raise IncompatibilityError(
                "The objective must produce one finite utility per terminal sample."
            )

        return utilities.squeeze(-1).mean(dim=0)

    def _compute_terminal_means(
        self,
        model: MultiStepModel,
        points: Tensor,
        incumbent: Tensor | None,
    ) -> Tensor:
        """Evaluate terminal decisions with learned network feasibility.

        Args:
            model: The current or hypothetical network model.
            points: Terminal reference configurations.
            incumbent: Best observed successful utility, if feasibility is modeled.

        Returns:
            Mean utility per decision, retaining the incumbent outside option.
        """
        import torch

        samples, probabilities = model.draw_network_outcomes(
            points,
            self.num_network_samples,
            sampling_seed=self.sampling_seed,
        )
        means = self._compute_terminal_means_from_network_samples(model.dag, samples)
        if incumbent is None:
            return means
        feasible = probabilities.mean(dim=0).squeeze(-1) >= self.feasibility_threshold
        return torch.where(feasible, means.clamp_min(incumbent), incumbent)

    def _make_node_actions(
        self,
        dag: MultiStepDAG,
        data: MultiStepData,
        pending: tuple[_PendingAction, ...] = (),
    ) -> tuple[_LocalProblem, ...]:
        """Enumerate fixed local input problems currently executable.

        Continuous own inputs stay free for optimization. Discrete own inputs and
        observed parent outputs are fixed. Parent outputs are drawn from real,
        successful evaluations and may be combined across independent
        experiments. Numerically equal parent outputs are collapsed for action
        generation only; the underlying observations remain distinct.

        Args:
            dag: The DAG topology.
            data: The measurement store.
            pending: Already reserved actions that must not be offered again.

        Returns:
            The local problems, in topological and input order.
        """
        problems: list[_LocalProblem] = []
        for name in dag.topological_order:
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
            parent_choices: list[list[tuple[Any, _EvaluationRef]]] = []
            for parent in node.parents:
                refs = data.evaluation_refs(dag, parent.name)
                choices: list[tuple[Any, _EvaluationRef]] = []
                seen: set[Any] = set()
                for ref in refs:
                    evaluation = data.get_evaluation(dag, ref)
                    value = evaluation.output
                    if value in seen:
                        continue
                    seen.add(value)
                    choices.append((value, ref))
                if not choices:
                    break
                parent_choices.append(choices)

            if len(parent_choices) != len(node.parents):
                continue

            combinations: list[tuple[dict[str, Any], dict[str, _EvaluationRef]]] = [
                (dict(inputs), {}) for inputs in own_rows
            ]
            for parent, choices in zip(node.parents, parent_choices):
                combinations = [
                    (
                        {**inputs, parent.output: value},
                        {**refs, parent.name: ref},
                    )
                    for inputs, refs in combinations
                    for value, ref in choices
                ]

            for fixed_inputs, parent_refs in combinations:
                if contains_action(pending, node=name, inputs=fixed_inputs):
                    continue
                problems.append(
                    _LocalProblem(
                        node=name,
                        fixed_inputs=fixed_inputs,
                        parent_refs=parent_refs,
                    )
                )
        return tuple(problems)

    def _pending_branches(  # noqa: DOC502
        self,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...],
        reference_points: Tensor,
        incumbent: Tensor | None,
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
            incumbent: Best observed successful utility, if applicable.
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

        conditioned_models = _condition_on_pending(
            model,
            pending,
            num_fantasies=self.num_fantasies,
            sampling_seed=self.sampling_seed,
        )
        branches: list[tuple[MultiStepModel, Tensor]] = []
        for conditioned in conditioned_models:
            with torch.no_grad():
                value = self._compute_terminal_means(
                    conditioned, reference_points, incumbent
                ).max()
            branches.append((conditioned, value))
        return tuple(branches)

    def _make_tensor_score_acquisition(
        self,
        *,
        model: MultiStepModel,
        node: MultiStepNode,
        reference_points: Tensor,
        baseline: Tensor,
        pending: tuple[_PendingAction, ...] = (),
        incumbent: Tensor | None = None,
    ) -> BoAcquisitionFunction:
        """Create the BoTorch tensor-score adapter for one node.

        Args:
            model: The fitted network model.
            node: The node whose measurement is evaluated.
            reference_points: The global terminal reference configurations.
            baseline: The best current terminal utility without pending data.
            pending: Reserved local executions whose outcomes are fantasized.
            incumbent: Best observed successful utility, if applicable.

        Returns:
            An acquisition function delegating scoring to this policy.
        """
        from botorch.settings import propagate_grads
        from gpytorch.settings import fast_pred_var

        policy = self
        pending_branches = self._pending_branches(
            model,
            pending,
            reference_points,
            incumbent,
            baseline,
        )

        def raw_score(X: Tensor) -> Tensor:
            """Compute p-KGFN's raw utility gain before action weighting."""
            import torch

            candidate_batch = tuple(X.shape[:-2])
            fantasy_means: list[Tensor] = []
            fantasy_baselines: list[Tensor] = []
            for conditioned_model, branch_baseline in pending_branches:
                with propagate_grads(True), fast_pred_var(False):
                    fantasy_model = conditioned_model.make_node_fantasies(
                        node, X, policy.num_fantasies, policy.sampling_seed
                    )
                fantasy_batch = tuple(
                    fantasy_model.get_node_model(node.name).batch_shape
                )
                expected_batch = (policy.num_fantasies, *candidate_batch)
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
                    means = policy._compute_terminal_means(
                        fantasy_model, points, incumbent
                    )
                if output_batch != fantasy_batch:
                    means = means.squeeze(-1)
                fantasy_means.append(means.movedim(1, 0))
                fantasy_baselines.append(
                    branch_baseline.expand(policy.num_fantasies, *candidate_batch)
                )

            return policy._compute_pkgfn_gain(
                torch.cat(fantasy_means, dim=0),
                torch.cat(fantasy_baselines, dim=0),
            )

        return _make_action_score_acquisition(
            model=model,
            raw_score=raw_score,
            success_probability=lambda inputs: model._predict_feasibility_tensor(
                node.name, inputs
            ),
            cost=node.cost,
            scale="linear",
            allow_negative_gains=True,
        )

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

    @staticmethod
    def _observed_incumbent(
        data: MultiStepData, model: MultiStepModel
    ) -> Tensor | None:
        """Return the best observed successful terminal utility.

        Only fully successful complete rows are candidates: derived rows are
        assembled exclusively from successful node measurements, and imported rows
        pass every configured feasibility condition by validation.

        Args:
            data: The measurement store.
            model: The fitted network model.

        Returns:
            The best observed terminal utility, or ``None`` when no node configures
            feasibility conditions or no successful complete row exists yet.

        Raises:
            IncompatibilityError: If the objective does not produce one finite
                utility per complete row.
        """
        import torch

        from baybe.utils.dataframe import to_tensor

        dag = model.dag
        if not any(node.feasibility_conditions for node in dag.nodes):
            return None
        complete = data.complete_measurements
        if complete.empty:
            return None

        reference = next(model.parameters())
        samples = (
            to_tensor(complete[list(dag.final_target_names)])
            .to(reference)
            .unsqueeze(-2)
        )
        utilities = dag.objective.to_botorch()(samples, None)
        if utilities.shape[-1] != 1 or not torch.isfinite(utilities).all():
            raise IncompatibilityError(
                "The objective must produce one finite utility per complete row."
            )
        return utilities.reshape(-1).max()

    def _optimize_node_actions(
        self,
        model: MultiStepModel,
        problems: tuple[_LocalProblem, ...],
        reference_points: Tensor,
        baseline: Tensor,
        pending: tuple[_PendingAction, ...] = (),
        incumbent: Tensor | None = None,
    ) -> tuple[_PendingAction, Tensor]:
        """Delegate a node's discrete or mixed action search to BoTorch.

        Args:
            model: The fitted network model.
            problems: The executable local problems for one node.
            reference_points: The global terminal reference configurations.
            baseline: The best current terminal utility without pending data.
            pending: Reserved local executions whose outcomes are fantasized.
            incumbent: Best observed successful utility, when feasibility is modeled.

        Returns:
            The optimized action and its tensor score.

        Raises:
            IncompatibilityError: If the optimizer returns non-finite values or a
                candidate outside the parameter bounds or changes fixed inputs.
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
        local_frame = pd.DataFrame(
            [problem.fixed_inputs for problem in problems]
        ).assign(
            **{
                name: parameter.bounds.lower
                for name, parameter in free_parameters.items()
            }
        )
        choices = model.transform_node_inputs(node.name, local_frame)
        acquisition = self._make_tensor_score_acquisition(
            model=model,
            node=node,
            reference_points=reference_points,
            baseline=baseline,
            pending=pending,
            incumbent=incumbent,
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
                torch.manual_seed(self.sampling_seed)
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
                    num_restarts=self.n_restarts,
                    raw_samples=self.n_raw_samples,
                    options={"seed": self.sampling_seed},
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

        if not torch.isfinite(candidate).all() or not torch.isfinite(score).all():
            raise IncompatibilityError(
                f"The optimizer's candidate or score for node '{node.name}' "
                "is not finite."
            )
        matches = (choices[:, fixed_indices] == candidate[0, fixed_indices]).all(dim=-1)
        if not matches.any():
            raise IncompatibilityError(
                f"The optimizer changed fixed inputs of node '{node.name}'."
            )
        selected_problem = problems[int(matches.nonzero()[0].item())]
        selected_inputs = dict(selected_problem.fixed_inputs)
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
            ),
            score,
        )

    def _validate_fitted_models(self, model: MultiStepModel) -> None:
        """Validate the fitted GP and likelihood contract required by p-KGFN.

        Args:
            model: The fitted network model.

        Raises:
            IncompatibilityError: If a node model is batched, is not a
                ``SingleTaskGP``, or does not use a plain ``GaussianLikelihood``.
            ModelNotTrainedError: If a configured feasibility model is missing.
        """
        import torch
        from botorch.models import SingleTaskGP
        from gpytorch.likelihoods import GaussianLikelihood

        if model.missing_feasibility_nodes:
            raise ModelNotTrainedError(
                "p-KGFN requires fitted local feasibility models for configured "
                f"nodes {list(model.missing_feasibility_nodes)}."
            )

        for name in model.dag.topological_order:
            gp = model.get_node_model(name)
            if (
                not isinstance(gp, SingleTaskGP)
                or gp.batch_shape != torch.Size()
                or not isinstance(gp.likelihood, GaussianLikelihood)
            ):
                raise IncompatibilityError(
                    f"p-KGFN requires an unbatched '{SingleTaskGP.__name__}' with "
                    f"a plain '{GaussianLikelihood.__name__}' for node '{name}', "
                    f"but found '{type(gp).__name__}'."
                )

    def select_action(
        self,
        data: MultiStepData,
        model: MultiStepModel,
        pending: tuple[_PendingAction, ...] = (),
    ) -> _PendingAction:
        """Select the best currently executable p-KGFN measurement action.

        No training is performed and neither campaign state nor real models are
        modified. Discrete inputs and observed parent outputs remain fixed while
        continuous own inputs are optimized. Ties keep the first action.

        Args:
            data: The measurement store determining executable actions.
            model: The fitted network model, including the DAG and objective.
            pending: Already reserved actions. Their outcomes are fantasized, and
                an identical action is not offered again.

        Returns:
            The highest-scoring node and its complete local input.

        Raises:
            IncompatibilityError: If the context or fitted models are unsupported,
                no action is executable, or a score is non-finite.
        """
        import torch

        dag = model.dag
        self.validate_context(dag, data, batch_size=1)
        self._validate_fitted_models(model)
        problems = self._make_node_actions(dag, data, pending)
        if not problems:
            raise IncompatibilityError(
                "The p-KGFN policy found no executable local problem. Every node "
                "requires real observations for all parent inputs."
            )

        reference = next(model.parameters())
        reference_points = self._make_terminal_reference_tensor(dag, reference)
        incumbent = self._observed_incumbent(data, model)
        with torch.no_grad():
            baseline = self._compute_terminal_means(
                model, reference_points, incumbent
            ).max()

        selected = max(
            (
                self._optimize_node_actions(
                    model,
                    tuple(node_problems),
                    reference_points,
                    baseline,
                    pending=pending,
                    incumbent=incumbent,
                )
                for _, node_problems in groupby(problems, key=attrgetter("node"))
            ),
            key=lambda result: result[1].item(),
        )[0]
        if contains_action(pending, node=selected.node, inputs=selected.inputs):
            raise IncompatibilityError(
                f"The selected continuous action of node '{selected.node}' is "
                "identical to an already pending action. Provide a different "
                "search context or report the pending result first."
            )
        return selected

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
                surrogate configurations, or candidate limits are unsupported.
        """
        if batch_size != 1:
            raise IncompatibilityError(
                f"p-KGFN supports only a single recommendation, got {batch_size=}."
            )

        objective = dag.objective
        if (
            not isinstance(objective, SingleTargetObjective)
            or not isinstance(objective.targets[0], NumericalTarget)
            or dag.final_output_dim != 1
        ):
            raise IncompatibilityError(
                "p-KGFN requires a numerical single-target objective and one "
                "final output."
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
        non_gp = sorted(
            node.name
            for node in dag.nodes
            if not isinstance(node.surrogate_model, GaussianProcessSurrogate)
        )
        if non_gp:
            raise IncompatibilityError(
                f"p-KGFN requires '{GaussianProcessSurrogate.__name__}' configurations "
                f"for nodes {non_gp}."
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

        n_local_problems = sum(
            math.prod(
                len(parameter.values)
                for parameter in node.inputs
                if isinstance(parameter, DiscreteParameter)
            )
            * math.prod(
                data.node_training_data(dag, parent.name)[parent.output].nunique()
                for parent in node.parents
            )
            for node in dag.nodes
        )
        if n_local_problems > self.max_node_candidates:
            raise IncompatibilityError(
                f"The DAG has {n_local_problems} fixed local measurement problems, "
                f"which exceeds the configured limit of {self.max_node_candidates}."
            )


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
