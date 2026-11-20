"""Acquisition policy for partial-evaluation function networks."""

from __future__ import annotations

import gc
import math
import warnings
from collections.abc import Callable
from itertools import groupby
from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn, cast

import pandas as pd
from attrs import Attribute, define, field
from attrs.validators import and_, ge, gt, instance_of, le, not_
from typing_extensions import override

from baybe.acquisition.base import AcquisitionFunction
from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.objectives.single import SingleTargetObjective
from baybe.parameters.base import DiscreteParameter
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget

if TYPE_CHECKING:
    from botorch.acquisition.acquisition import (
        AcquisitionFunction as BoAcquisitionFunction,
    )
    from torch import Tensor

    from .dag import MultiStepDAG
    from .data import MultiStepData
    from .model import MultiStepModel
    from .node import MultiStepNode


@define(frozen=True)
class _OpenAction:
    """A selected node measurement with its complete local input."""

    node: str = field()
    """The node to execute."""

    inputs: dict[str, Any] = field()
    """The node's own parameter values and observed parent outputs."""


@define(frozen=True)
class _LocalProblem:
    """A node and its fixed discrete and observed-parent inputs."""

    node: str = field()
    """The node to evaluate."""

    fixed_inputs: dict[str, Any] = field()
    """The discrete own inputs and observed parent outputs kept fixed."""


@define(frozen=True)
class PartialKnowledgeGradientFN(AcquisitionFunction):
    """Partial-knowledge-gradient policy for function networks.

    The policy selects one node and its complete local input. It scores the
    expected gain of the best terminal decision per cost of that single
    measurement, instead of maximizing a local node output. Continuous node-owned
    inputs are optimized against a finite terminal reference set. The
    configuration is only valid for
    :class:`~baybe.multistep.campaign.MultiStepCampaign`; it is not a stock
    BoTorch point acquisition function.

    The policy owns action generation, terminal utility evaluation, scoring, and
    optimization. The network model provides training, input transformation,
    sampling, and fantasy conditioning without implementing policy decisions.

    Feasibility affects a measurement action through its current predicted success
    probability: successful output fantasies are weighted by ``p`` and the
    no-output failure branch by ``1 - p``. The policy therefore accounts for the
    information gain about successful outputs through its output fantasies, but it
    does not model the information gain about feasibility from a hypothetical
    local observation. Only real measurements update the feasibility classifier.
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
    """Minimum network success probability required for a new terminal decision.

    The best actually observed successful terminal result remains an available
    decision. Failure observations update feasibility but never the output model.
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
    def _compute_pkgfn_score(
        fantasy_terminal_means: Tensor,
        baseline: Tensor,
        cost: float,
        fantasy_weights: Tensor | None = None,
    ) -> Tensor:
        """Reduce fantasy terminal means to a cost-normalized p-KGFN score.

        Args:
            fantasy_terminal_means: Network-averaged utilities with shape
                ``(num_fantasies, num_terminal_candidates, batch...)``.
            baseline: The scalar baseline utility of the current network.
            cost: The positive cost of the selected node.
            fantasy_weights: Optional outcome probabilities matching the fantasy
                and candidate batch axes. Without weights, use a uniform average.

        Returns:
            One finite tensor score per candidate batch.

        Raises:
            IncompatibilityError: If the input shape, baseline, cost, or score is
                invalid or non-finite.
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
            or baseline.dim() != 0
            or not torch.isfinite(baseline)
            or not math.isfinite(cost)
            or cost <= 0
        ):
            raise IncompatibilityError(
                "p-KGFN requires finite means, a finite scalar baseline, and a "
                "positive finite cost."
            )

        best = fantasy_terminal_means.amax(dim=1)
        expected = (
            best.mean(dim=0)
            if fantasy_weights is None
            else (best * fantasy_weights).sum(dim=0)
        )
        score = (expected - baseline) / cost
        if not torch.isfinite(score).all():
            raise IncompatibilityError("The p-KGFN score is not finite.")
        return score

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

    @staticmethod
    def _combine_branch_means(
        success_means: Tensor,
        incumbent: Tensor,
        probability: Tensor,
        num_fantasies: int,
    ) -> tuple[Tensor, Tensor]:
        """Combine output fantasies with the no-output failure branch.

        The success branch keeps one terminal mean per output fantasy and is
        weighted by the node's predicted success probability. The failure branch
        performs no output update: every terminal decision keeps the incumbent
        value and receives the complementary probability. No feasibility
        observation is invented, so the classifier is not conditioned on a
        hypothetical outcome.

        Args:
            success_means: Terminal means from the fantasized output model with
                shape ``(num_fantasies, num_terminal_candidates, batch...)``.
            incumbent: The best observed successful utility as a scalar tensor.
            probability: The node's predicted success probability with shape
                ``(batch...,)``.
            num_fantasies: The number of output fantasies.

        Returns:
            The concatenated branch means with shape
            ``(num_fantasies + 1, num_terminal_candidates, batch...)`` and the
            matching branch weights with shape ``(num_fantasies + 1, batch...,)``.

        Raises:
            IncompatibilityError: If the branch shapes are inconsistent or a
                value is non-finite.
        """
        import torch

        if success_means.dim() < 2:
            raise IncompatibilityError(
                "Success-branch means must have shape "
                "(fantasies, terminal candidates, batch...), got "
                f"{tuple(success_means.shape)}."
            )
        if success_means.shape[0] != num_fantasies:
            raise IncompatibilityError(
                f"The success branch contains {success_means.shape[0]} fantasies, "
                f"but {num_fantasies} were requested."
            )
        expected_batch_shape = success_means.shape[2:]
        if probability.shape != expected_batch_shape:
            raise IncompatibilityError(
                f"The success probability has shape {tuple(probability.shape)}, "
                f"but the success branch expects {expected_batch_shape}."
            )
        if incumbent.dim() != 0 or not torch.isfinite(incumbent):
            raise IncompatibilityError("The incumbent utility must be a finite scalar.")
        if (
            not torch.isfinite(success_means).all()
            or not torch.isfinite(probability).all()
        ):
            raise IncompatibilityError(
                "The success branch and probability must be finite."
            )

        failure_means = incumbent.expand(1, *success_means.shape[1:])
        success_weights = (
            probability.expand(num_fantasies, *probability.shape) / num_fantasies
        )
        failure_weights = (1.0 - probability).unsqueeze(0)
        means = torch.cat((success_means, failure_means), dim=0)
        weights = torch.cat((success_weights, failure_weights), dim=0)
        return means, weights

    def _make_node_actions(  # noqa: DOC502
        self, dag: MultiStepDAG, data: MultiStepData
    ) -> tuple[_LocalProblem, ...]:
        """Enumerate fixed local input problems currently executable.

        Continuous own inputs stay free for optimization. Discrete own inputs and
        observed parent outputs are fixed.

        Args:
            dag: The DAG topology.
            data: The measurement store.

        Returns:
            The local problems, in topological and input order.
        """
        problems: list[_LocalProblem] = []
        for name in dag.topological_order:
            node = dag.node_by_name[name]
            inputs = (
                node.searchspace.discrete.exp_rep
                if node.inputs and not node.searchspace.discrete.is_empty
                else pd.DataFrame(index=[0])
            )
            for parent in node.parents:
                frame = data.node_training_data(dag, parent.name)
                if frame.empty:
                    break
                inputs = inputs.merge(
                    frame[[parent.output]].drop_duplicates(), how="cross"
                )
            else:
                rows = inputs.to_dict("records") if len(inputs.columns) else [{}]
                problems.extend(
                    _LocalProblem(node=name, fixed_inputs=cast(dict[str, Any], row))
                    for row in rows
                )
        return tuple(problems)

    def _make_tensor_score_acquisition(
        self,
        *,
        model: MultiStepModel,
        node: MultiStepNode,
        reference_points: Tensor,
        baseline: Tensor,
        incumbent: Tensor | None = None,
    ) -> BoAcquisitionFunction:
        """Create the BoTorch tensor-score adapter for one node.

        Args:
            model: The fitted network model.
            node: The node whose measurement is evaluated.
            reference_points: The global terminal reference configurations.
            baseline: The best current terminal utility.
            incumbent: The best actually observed successful utility, if applicable.

        Returns:
            An acquisition function delegating scoring to this policy.
        """
        from attrs import setters
        from botorch.acquisition.acquisition import (
            AcquisitionFunction as BoAcquisitionFunction,
        )
        from botorch.settings import propagate_grads
        from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
        from gpytorch.settings import fast_pred_var

        policy = self

        @define(eq=False, repr=False, slots=False, init=False, on_setattr=setters.NO_OP)
        class _TensorScoreAcquisition(BoAcquisitionFunction):
            """Adapt the policy's tensor score to the BoTorch optimizer."""

            @override
            @cast(
                "Callable[[Callable[..., Tensor]], Callable[..., Tensor]]",
                t_batch_mode_transform(expected_q=1),
            )
            def forward(self, X: Tensor) -> Tensor:
                fantasy_inputs = X.expand(policy.num_fantasies, *X.shape)
                fantasy_points = match_batch_shape(
                    reference_points.reshape(
                        len(reference_points),
                        *(1 for _ in fantasy_inputs.shape[:-2]),
                        *reference_points.shape[-2:],
                    ),
                    fantasy_inputs,
                )
                with propagate_grads(True), fast_pred_var(False):
                    fantasy_model = model.make_node_fantasies(
                        node, X, policy.num_fantasies, policy.sampling_seed
                    )
                    means = policy._compute_terminal_means(
                        fantasy_model, fantasy_points, incumbent
                    ).movedim(1, 0)
                    weights = None
                    if node.feasibility_conditions:
                        if incumbent is None:
                            raise IncompatibilityError(
                                f"Node '{node.name}' requires an incumbent utility "
                                "for its failure branch."
                            )
                        probability = (
                            model._feasibility_models[node.name]
                            .posterior_probabilities(X)
                            .reshape(X.shape[:-2])
                        )
                        means, weights = policy._combine_branch_means(
                            means,
                            incumbent,
                            probability,
                            policy.num_fantasies,
                        )
                    return policy._compute_pkgfn_score(
                        means, baseline, node.cost, weights
                    )

        return _TensorScoreAcquisition(model=model)

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

    def _optimize_node_actions(
        self,
        model: MultiStepModel,
        problems: tuple[_LocalProblem, ...],
        reference_points: Tensor,
        baseline: Tensor,
        incumbent: Tensor | None = None,
    ) -> tuple[_OpenAction, Tensor]:
        """Delegate a node's discrete or mixed action search to BoTorch.

        Args:
            model: The fitted network model.
            problems: The executable local problems for one node.
            reference_points: The global terminal reference configurations.
            baseline: The best current terminal utility.
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
        selected_inputs = dict(problems[int(matches.nonzero()[0].item())].fixed_inputs)
        selected_inputs.update(
            {
                name: candidate[0, model_columns.index(name)].item()
                for name in free_parameters
            }
        )
        return _OpenAction(node=node.name, inputs=selected_inputs), score

    def _validate_fitted_models(self, model: MultiStepModel) -> None:
        """Validate the fitted GP and likelihood contract required by p-KGFN.

        Args:
            model: The fitted network model.

        Raises:
            IncompatibilityError: If a node model is batched, is not a
                ``SingleTaskGP``, or does not use a plain ``GaussianLikelihood``.
        """
        import torch
        from botorch.models import SingleTaskGP
        from gpytorch.likelihoods import GaussianLikelihood

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

    def select_action(self, data: MultiStepData, model: MultiStepModel) -> _OpenAction:
        """Select the best currently executable p-KGFN measurement action.

        No training is performed and neither campaign state nor real models are
        modified. Discrete inputs and observed parent outputs remain fixed while
        continuous own inputs are optimized. Ties keep the first action.

        Args:
            data: The measurement store determining executable actions.
            model: The fitted network model, including the DAG and objective.

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
        problems = self._make_node_actions(dag, data)
        if not problems:
            raise IncompatibilityError(
                "The p-KGFN policy found no executable local problem. Every node "
                "requires real observations for all parent inputs."
            )

        reference = next(model.parameters())
        reference_points = self._make_terminal_reference_tensor(dag, reference)
        incumbent = None
        if any(node.feasibility_conditions for node in dag.nodes):
            from baybe.utils.dataframe import to_tensor

            terminal = data.node_training_data(dag, dag.final_nodes[0])
            observed = to_tensor(terminal[list(dag.final_target_names)]).to(reference)
            incumbent = dag.objective.to_botorch()(observed.unsqueeze(-2), None).max()
        with torch.no_grad():
            baseline = self._compute_terminal_means(
                model, reference_points, incumbent
            ).max()

        return max(
            (
                self._optimize_node_actions(
                    model, tuple(node_problems), reference_points, baseline, incumbent
                )
                for _, node_problems in groupby(problems, key=attrgetter("node"))
            ),
            key=lambda result: result[1].item(),
        )[0]

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
