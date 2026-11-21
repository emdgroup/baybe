"""Multi-step campaign facade for DAG-structured experiments.

The facade owns the multi-step measurements and fits a single
:class:`~baybe.multistep.model.MultiStepModel` that is composed of one model per
DAG node. For p-KGFN it delegates action selection to
:class:`~baybe.multistep.acquisition.PartialKnowledgeGradientFN`; for stock
acquisition functions it hands the model to a
:class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`
directly. A recommendation sees the DAG-wide search space and the final-node
objective; upstream-only measurements still train their node models.

The campaign keeps an explicit snapshot for every experiment. A recommendation
automatically reserves its concrete local action as pending and assigns or reuses
a compatible experiment identifier. Measurements are dataframes that carry the
identifier back, may arrive in any order, and are matched by
``(experiment_id, node)``. Once all nodes of an experiment are bound to successful
real evaluations with globally consistent inputs, the campaign assembles a
derived complete DAG row. Inconsistent reused branches remain valid local
observations without a complete-row projection. Pending
actions are never treated as completed terminal measurements.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Any, cast

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field, setters
from attrs.converters import optional
from attrs.validators import ge, instance_of, le

from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import IncompatibilityError, InfeasibilityError
from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.serialization import SerialMixin, converter

from ._columns import (
    COMPLETE_SOURCE,
    COMPLETE_SOURCE_DERIVED,
    EXPERIMENT_ID,
    NODE,
)
from ._experiments import (
    _EvaluationRef,
    _Experiment,
    _PendingAction,
    contains_action,
    experiment_actions,
    next_executable_node,
    pending_actions,
    register_action,
)
from ._scoring import (
    ScoreScale,
    _reduce_score_branches,
    _resolve_score_scale,
    _score_actions,
)
from .acquisition import (
    PartialKnowledgeGradientFN,
    _condition_on_pending,
)
from .dag import MultiStepDAG
from .data import MultiStepData, NodeMeasurement, _is_successful, _validate_node_inputs
from .model import MultiStepModel, _AcquisitionSurrogateAdapter
from .node import MultiStepNode


def _is_missing(value: Any) -> bool:
    """Return whether a scalar measurement value is absent."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _values_match(parameter: Any, measured: Any, expected: Any) -> bool:
    """Return whether a measured own input matches the reserved action.

    Numerical parameters allow their configured tolerance. Non-numerical
    parameters require exact equality.

    Args:
        parameter: The parameter owning the input.
        measured: The value reported in the measurement frame.
        expected: The value stored in the pending action.

    Returns:
        Whether the values describe the same local input.
    """
    if getattr(parameter, "is_numerical", False):
        if isinstance(measured, bool) or not isinstance(
            measured, (int, float, np.integer, np.floating)
        ):
            return False
        tolerance = float(getattr(parameter, "tolerance", 0.0))
        return bool(
            np.isclose(
                float(measured),
                float(expected),
                rtol=0.0,
                atol=tolerance + 1e-12,
            )
        )
    return measured == expected


_PENDING_FANTASIES: int = 8
"""Pending outcome branches used when averaging a stock acquisition."""

_PENDING_FANTASY_SEED: int = 0
"""Deterministic seed for stock pending-outcome sampling."""


@define(frozen=True)
class _ExecutablePlan:
    """One executable next node with its concrete parent evaluations.

    A plan either continues an existing experiment or, when
    ``experiment_id`` is ``None``, starts a fresh experiment from a root node.
    """

    experiment_id: str | None = field()
    """The experiment to continue, or ``None`` for a new context."""

    node: str = field()
    """The node that the plan makes executable."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=lambda value: MappingProxyType(dict(value))
    )
    """The concrete real parent evaluations supplying the local inputs."""


@define(slots=False)
class MultiStepCampaign(SerialMixin):
    """Stateful facade for multi-step Bayesian optimization.

    Node surrogates that produce ensemble posteriors, such as
    :class:`~baybe.surrogates.random_forest.RandomForestSurrogate`, can drive
    recommendations over discrete search spaces. Because ensemble posteriors are
    not differentiable, continuous search spaces require gradient-capable node
    surrogates.

    The facade fits a single multi-step model composed of one model per node.
    The p-KGFN policy selects actions directly; stock acquisition functions use a
    :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`
    directly. The acquisition function is configurable; other recommender
    settings are not exposed.

    Every recommendation is registered as a pending local execution. Several
    experiments may therefore be open simultaneously, and their pending actions
    are reported independently through :meth:`add_measurements`.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    # ``add_measurements`` validates the store explicitly because the accepted
    # numerical tolerance is a per-call argument, so the default setattr validation
    # would enforce the strict default instead of the call's setting.
    data: MultiStepData = field(factory=MultiStepData.empty, on_setattr=setters.convert)
    """The measurement store holding node measurements and complete rows."""

    acquisition_function: AcquisitionFunction | None = field(
        default=None, converter=optional(convert_acqf)
    )
    """The p-KGFN policy or stock acquisition function.

    ``None`` selects the default Bayesian acquisition, also with feasibility conditions.
    Random design is used only while the required initial observations are missing.
    """

    feasibility_threshold: float = field(
        default=0.5, validator=[instance_of(float), ge(0.0), le(1.0)], kw_only=True
    )
    """Minimum network success probability for stock acquisition candidates.

    p-KGFN uses the threshold configured on its acquisition policy instead.
    """

    _experiments: dict[str, _Experiment] = field(factory=dict, init=False, eq=False)
    """All evaluation contexts in creation order, keyed by identifier."""

    _experiment_counter: int = field(default=0, init=False, eq=False)
    """Monotonic counter for generating experiment identifiers."""

    @data.validator
    def _validate_data(  # noqa: DOC101, DOC103
        self, _: Any, value: MultiStepData
    ) -> None:
        """Validate the measurement store against the DAG."""
        value.validate(self.dag)

    @classmethod
    def from_dag(
        cls,
        dag: MultiStepDAG,
        *,
        acquisition_function: AcquisitionFunction | str | None = None,
        data: MultiStepData | None = None,
        feasibility_threshold: float = 0.5,
    ) -> MultiStepCampaign:
        """Create a campaign for the given DAG.

        Args:
            dag: The DAG topology, including the optimization objective.
            acquisition_function: Optional acquisition function or its name.
            data: Optional initial measurement store.
            feasibility_threshold: Network success threshold for stock acquisitions.

        Returns:
            The configured campaign.
        """
        return cls(
            dag=dag,
            data=MultiStepData.empty() if data is None else data,
            acquisition_function=acquisition_function,
            feasibility_threshold=feasibility_threshold,
        )

    @cached_property
    def model(self) -> MultiStepModel:
        """The multi-step model owned by the campaign."""
        return self.dag.make_model()

    @property
    def pending_experiments(self) -> pd.DataFrame:
        """Return a defensive view of all currently pending local actions.

        Returns:
            One row per reserved action, carrying ``experiment_id``, ``node``,
            and the complete known local input. Modifying the frame does not
            affect campaign state.
        """
        columns = [EXPERIMENT_ID, NODE]
        rows: list[dict[str, Any]] = []
        for experiment in self._experiments.values():
            for action in experiment_actions(experiment):
                rows.append(
                    {
                        EXPERIMENT_ID: experiment.experiment_id,
                        NODE: action.node,
                        **dict(action.inputs),
                    }
                )
                for name in action.inputs:
                    if name not in columns:
                        columns.append(name)
        return pd.DataFrame(rows).reindex(columns=columns)

    def add_measurements(
        self,
        data: pd.DataFrame,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Record measured node outputs for any active experiment.

        The dataframe carries the experiment context back to the campaign. It uses
        the format returned by :meth:`recommend`, extended with the observed result
        columns. Rows may arrive in any order and may cover only a subset of the
        outstanding actions.

        Measurements are matched by ``(experiment_id, node)``. Own parameter values
        must agree with the recommendation that reserved the action. Parent outputs
        are resolved from the concrete stored evaluations bound to that experiment;
        repeated parent-output columns in the dataframe are optional but must agree.
        Successful measurements and explicit failures are stored once per context.

        Args:
            data: One row per node measurement, holding ``experiment_id``, ``node``,
                the node's own parameters, optionally the outputs of its parents, the
                node-output column, and any auxiliary columns referenced by
                feasibility conditions. Missing result values are allowed only when
                another observed condition establishes failure.
            numerical_measurements_must_be_within_tolerance: Flag indicating if
                numerical parameters need to be within their tolerances.

        Raises:
            IncompatibilityError: If the context or node is unknown, a node has no
                pending action, a duplicate or conflicting row is supplied, or the
                values violate the measurement contract.
            IncompleteMeasurementsError: If condition values are missing.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        if data.empty:
            raise IncompatibilityError("Cannot add an empty measurement batch.")
        if not data.columns.is_unique:
            raise IncompatibilityError("Measurement columns must be unique.")

        # Validate the existing store once; dataframe-backed views remain mutable.
        self.data.validate(
            self.dag, numerical_measurements_must_be_within_tolerance=False
        )

        missing_context = [
            column for column in (NODE, EXPERIMENT_ID) if column not in data
        ]
        if missing_context:
            raise IncompatibilityError(
                f"Measurements must carry the experiment context. Missing columns: "
                f"{missing_context}."
            )

        experiments = dict(self._experiments)
        store = self.data
        affected: set[str] = set()
        responded: set[tuple[str, str]] = set()

        for values in data.to_dict("records"):
            raw_experiment_id = values[EXPERIMENT_ID]
            if _is_missing(raw_experiment_id):
                raise IncompatibilityError(
                    "Every measurement row requires an experiment identifier."
                )
            experiment_id = str(raw_experiment_id)
            experiment = experiments.get(experiment_id)
            if experiment is None:
                raise IncompatibilityError(
                    f"There is no active experiment with identifier '{experiment_id}'."
                )

            raw_node = values[NODE]
            if _is_missing(raw_node):
                raise IncompatibilityError(
                    "Every measurement row requires a node name."
                )
            node = self.dag.node_by_name.get(str(raw_node))
            if node is None:
                raise IncompatibilityError(
                    f"Node '{raw_node}' is not part of the DAG. Known nodes: "
                    f"{sorted(self.dag.node_by_name)}."
                )
            if (experiment_id, node.name) in responded:
                raise IncompatibilityError(
                    f"Node '{node.name}' was reported more than once in the same "
                    f"batch for experiment '{experiment_id}'."
                )
            responded.add((experiment_id, node.name))
            if node.name in experiment.bindings:
                raise IncompatibilityError(
                    f"Node '{node.name}' has already been measured in experiment "
                    f"'{experiment_id}'."
                )
            action = experiment.pending.get(node.name)
            if action is None:
                raise IncompatibilityError(
                    f"Node '{node.name}' has no pending action in experiment "
                    f"'{experiment_id}'. Call 'recommend()' before reporting its "
                    "measurement."
                )

            measurement = self._make_experiment_measurement(
                node,
                cast(Mapping[str, Any], values),
                action,
                experiment_id,
                numerical_measurements_must_be_within_tolerance,
            )
            ref = _EvaluationRef(
                source="measurement", index=len(store.measurements), node=node.name
            )
            store = store.add(measurement)
            experiments[experiment_id] = experiment.with_binding(
                node.name, ref, action.parent_refs
            ).without_pending(node.name)
            affected.add(experiment_id)

        # Per-row checks use this call's tolerance, while the existing store was
        # validated before the batch. Derived rows are assembled from those values.
        for experiment_id in affected:
            experiment = experiments[experiment_id]
            if self._has_derived_complete_row(store, experiment_id):
                continue
            row = store.make_complete_row(
                self.dag, experiment.bindings, _skip_invalid=True
            )
            if row is None:
                continue
            row[EXPERIMENT_ID] = experiment_id
            store = store.add_complete(row, derived=True)

        # Publish only after the complete batch has been validated.
        self.data = store
        self._experiments = experiments

    def _has_derived_complete_row(
        self, data: MultiStepData, experiment_id: str
    ) -> bool:
        """Return whether the derived completion view already exists.

        Args:
            data: The measurement store.
            experiment_id: The context to inspect.

        Returns:
            ``True`` if a derived complete row already represents the context.
        """
        if EXPERIMENT_ID not in data.complete_measurements:
            return False

        frame = data.complete_measurements
        return bool(
            (
                (frame[EXPERIMENT_ID] == experiment_id)
                & (frame[COMPLETE_SOURCE] == COMPLETE_SOURCE_DERIVED)
            ).any()
        )

    def _make_experiment_measurement(
        self,
        node: MultiStepNode,
        values: Mapping[str, Any],
        action: _PendingAction,
        experiment_id: str,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> NodeMeasurement:
        """Resolve one measured row into an observation of a context.

        Args:
            node: The observed node.
            values: One row of the measurement dataframe, including the node and
                experiment identifier.
            action: The pending action that reserved this execution.
            experiment_id: The experiment the observation belongs to.
            numerical_measurements_must_be_within_tolerance: If ``True``, numerical
                parameter values must lie within the parameter range or tolerance.

        Returns:
            The resolved node observation.

        Raises:
            IncompatibilityError: If the columns, the own parameter values, or the
                repeated parent outputs are invalid or disagree with the action.
            IncompleteMeasurementsError: If condition values are missing.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        auxiliary_columns = {
            condition.column
            for condition in node.feasibility_conditions
            if condition.column != node.output
        }
        allowed = {
            NODE,
            EXPERIMENT_ID,
            *node.model_input_names,
            node.output,
            *auxiliary_columns,
        }
        foreign = {
            column
            for column, value in values.items()
            if column not in allowed and not _is_missing(value)
        }
        if foreign:
            raise IncompatibilityError(
                f"Measurements for node '{node.name}' contain unexpected columns "
                f"{sorted(foreign)}. Allowed columns: {sorted(allowed)}."
            )

        own_parameters = tuple(node.inputs)
        own_names = tuple(parameter.name for parameter in own_parameters)
        missing = sorted(name for name in own_names if name not in values)
        if missing:
            raise IncompatibilityError(
                f"Measurements for node '{node.name}' are missing the own parameter "
                f"columns {missing}."
            )

        inputs = dict(action.inputs)
        for parameter in own_parameters:
            measured = values[parameter.name]
            expected = action.inputs[parameter.name]
            if not _values_match(parameter, measured, expected):
                raise IncompatibilityError(
                    f"The value {measured!r} of parameter '{parameter.name}' does not "
                    f"match the recommended value {expected!r} for node "
                    f"'{node.name}' in experiment '{experiment_id}'."
                )
            inputs[parameter.name] = measured

        for parent in node.parents:
            if parent.output not in values or _is_missing(values[parent.output]):
                continue
            if values[parent.output] != action.inputs[parent.output]:
                raise IncompatibilityError(
                    f"The parent output '{parent.output}' in the measurements for node "
                    f"'{node.name}' does not match the value of parent "
                    f"'{parent.name}' bound to the recommended action in experiment "
                    f"'{experiment_id}'."
                )

        _validate_node_inputs(
            node,
            pd.DataFrame([inputs]),
            numerical_measurements_must_be_within_tolerance,
        )

        output = values.get(node.output)
        measurement = NodeMeasurement(
            node=node.name,
            inputs=inputs,
            output=None if _is_missing(output) else output,
            experiment_id=experiment_id,
            condition_values={
                column: values[column]
                for column in auxiliary_columns
                if column in values
            },
        )
        if measurement.output is None and _is_successful(node, measurement):
            raise IncompatibilityError(
                f"Successful observations of node '{node.name}' require an output."
            )
        return measurement

    def recommend(self, batch_size: int = 1) -> pd.DataFrame:
        """Recommend and reserve the next node actions.

        With :class:`~baybe.multistep.acquisition.PartialKnowledgeGradientFN`,
        select one local measurement action by weighting the expected terminal
        utility gain with local success probability and node cost. The policy
        supports discrete, continuous, and hybrid node-owned inputs, and returns
        the complete local input including real observed parent outputs. It does
        not require a complete DAG row.
        Pending real actions are fantasized when a new action is scored.

        With supported stock acquisition functions, score configurations in the
        complete DAG search space, project them onto executable local actions,
        and rank those actions after applying their local success probability
        and node cost. This route requires a purely discrete DAG search space.
        Pending local actions are conditioned on hypothetical node models, and
        the stock acquisition is averaged over their outcomes in linear or log
        gain space as appropriate.

        A random initial design is used only while the required output or
        feasibility models cannot be fitted. Thereafter, the configured
        acquisition selects the action; ``None`` selects the default Bayesian
        acquisition.

        Every returned action is registered as pending. Two actions of the same
        node cannot share an experiment, so a batch that repeats one node uses
        several experiment identifiers. New identifiers are published only after
        every computation and registration succeeded.

        Args:
            batch_size: The number of node actions to recommend.

        Returns:
            A dataframe holding ``experiment_id``, ``node``, and the complete
            local input of each recommended node (its own parameters and parent
            outputs).

        Raises:
            ValueError: If ``batch_size`` is smaller than 1.
            IncompatibilityError: If the legacy route receives a non-discrete DAG
                search space, or if the selected policy does not support the DAG,
                objective, model, or requested batch size, or if no executable
                action is available.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but "
                f"provided {batch_size=}."
            )

        pkgfn_config = (
            self.acquisition_function
            if isinstance(self.acquisition_function, PartialKnowledgeGradientFN)
            else None
        )
        resolved_acquisition: AcquisitionFunction | None = None
        score_scale: ScoreScale | None = None
        if pkgfn_config is None:
            acquisition_resolver = BotorchRecommender(
                acquisition_function=self.acquisition_function
            )
            resolved_acquisition = acquisition_resolver._get_acquisition_function(
                self.dag.objective
            )
            # Validate the gain contract before fitting models or running an
            # optimizer. Unknown signed utilities cannot be made into gains by
            # multiplying them with risk and cost factors.
            score_scale = _resolve_score_scale(resolved_acquisition)

        if pkgfn_config is not None:
            pkgfn_config.validate_context(self.dag, self.data, batch_size)

        pending = pending_actions(self._experiments)
        if pkgfn_config is None:
            self._validate_stock_pending(pending)
        self._fit_node_models()
        needs_initial_data = bool(
            self.model.missing_nodes or self.model.missing_feasibility_nodes
        )

        if pkgfn_config is not None and not needs_initial_data:
            actions = [pkgfn_config.select_action(self.data, self.model, pending)]
        elif needs_initial_data or self.data.complete_measurements.empty:
            actions = self._recommend_random(batch_size, pending)
        else:
            assert resolved_acquisition is not None
            assert score_scale is not None
            actions = self._recommend_optimum(
                batch_size,
                pending,
                acquisition_function=resolved_acquisition,
                score_scale=score_scale,
            )

        experiments = dict(self._experiments)
        counter = self._experiment_counter
        columns = [EXPERIMENT_ID, NODE]
        rows: list[dict[str, Any]] = []
        for action in actions:
            experiments, counter, experiment_id = register_action(
                experiments,
                counter,
                node=action.node,
                inputs=action.inputs,
                parent_refs=action.parent_refs,
                data=self.data,
                dag=self.dag,
            )
            rows.append(
                {
                    EXPERIMENT_ID: experiment_id,
                    NODE: action.node,
                    **dict(action.inputs),
                }
            )
            for name in self.dag.node_by_name[action.node].model_input_names:
                if name not in columns:
                    columns.append(name)

        # Publish experiment state only after the complete batch has been built.
        result = pd.DataFrame(rows).reindex(columns=columns)
        self._experiments = experiments
        self._experiment_counter = counter
        return result

    def _executable_plans(self) -> tuple[_ExecutablePlan, ...]:
        """Collect executable next nodes from all open experiments and fresh plans.

        Returns:
            Plans continuing an existing experiment first, followed by fresh
            root-node plans and fresh non-root plans whose parents already have
            successful real evaluations, both in topological order.
        """
        plans: list[_ExecutablePlan] = []
        for experiment in self._experiments.values():
            next_node = next_executable_node(experiment, self.dag, self.data)
            if next_node is not None:
                node, parent_refs = next_node
                plans.append(
                    _ExecutablePlan(
                        experiment_id=experiment.experiment_id,
                        node=node,
                        parent_refs=parent_refs,
                    )
                )
        for name in self.dag.topological_order:
            if not self.dag.parents_of(name):
                plans.append(_ExecutablePlan(experiment_id=None, node=name))
        for name in self.dag.topological_order:
            parents = self.dag.parents_of(name)
            if not parents:
                continue
            parent_bindings: dict[str, _EvaluationRef] = {}
            for parent in parents:
                refs = self.data.evaluation_refs(self.dag, parent)
                if not refs:
                    break
                parent_bindings[parent] = refs[0]
            else:
                plans.append(
                    _ExecutablePlan(
                        experiment_id=None, node=name, parent_refs=parent_bindings
                    )
                )
        return tuple(plans)

    def _parent_outputs(
        self, parent_refs: Mapping[str, _EvaluationRef]
    ) -> dict[str, Any]:
        """Resolve the real parent outputs of a concrete action.

        Args:
            parent_refs: The parent evaluations supplying the local input.

        Returns:
            The observed parent outputs keyed by their measurement column.
        """
        return {
            self.dag.node_by_name[parent].output: self.data.get_evaluation(
                self.dag, ref
            ).output
            for parent, ref in parent_refs.items()
        }

    def _random_actions_for_plan(
        self,
        plan: _ExecutablePlan,
        batch_size: int,
        pending: tuple[_PendingAction, ...],
    ) -> list[_PendingAction] | None:
        """Draw distinct random actions for one executable plan.

        Args:
            plan: The executable node context.
            batch_size: The number of distinct actions requested.
            pending: Already reserved actions that must not be duplicated.

        Returns:
            The random actions, or ``None`` if the plan cannot provide the
            requested number of pending-free distinct actions.
        """
        node = self.dag.node_by_name[plan.node]
        parent_outputs = self._parent_outputs(plan.parent_refs)

        if not node.inputs:
            if batch_size != 1:
                return None
            inputs = dict(parent_outputs)
            if contains_action(pending, node=node.name, inputs=inputs):
                return None
            return [
                _PendingAction(
                    node=node.name,
                    inputs=inputs,
                    parent_refs=plan.parent_refs,
                )
            ]

        searchspace = node.searchspace
        if searchspace.continuous.is_empty:
            candidates = searchspace.discrete.exp_rep.drop_duplicates()
            available: list[dict[str, Any]] = [
                {str(key): value for key, value in raw_row.items()}
                for raw_row in candidates.to_dict("records")
                if not contains_action(
                    pending,
                    node=node.name,
                    inputs={
                        **{str(key): value for key, value in raw_row.items()},
                        **parent_outputs,
                    },
                )
            ]
            if len(available) < batch_size:
                return None
            selected = pd.DataFrame(available).sample(n=batch_size, replace=False)
        else:
            sampled = RandomRecommender().recommend(
                batch_size=max(2 * batch_size, batch_size),
                searchspace=searchspace,
            )
            rows: list[dict[str, Any]] = []
            seen: set[tuple[tuple[str, Any], ...]] = set()
            for raw_row in sampled.to_dict("records"):
                row = {str(key): value for key, value in raw_row.items()}
                key = tuple(sorted(row.items()))
                if key in seen or contains_action(
                    pending, node=node.name, inputs={**row, **parent_outputs}
                ):
                    continue
                seen.add(key)
                rows.append(row)
                if len(rows) == batch_size:
                    break
            if len(rows) < batch_size:
                return None
            selected = pd.DataFrame(rows)

        selected_records = [
            {str(key): value for key, value in raw_row.items()}
            for raw_row in selected.to_dict("records")
        ]
        return [
            _PendingAction(
                node=node.name,
                inputs={**row, **parent_outputs},
                parent_refs=plan.parent_refs,
            )
            for row in selected_records
        ]

    def _recommend_random(
        self, batch_size: int, pending: tuple[_PendingAction, ...]
    ) -> list[_PendingAction]:
        """Select pending-free random actions for initial data collection.

        Args:
            batch_size: The number of actions requested.
            pending: Already reserved actions that must not be duplicated.

        Returns:
            The selected random actions.

        Raises:
            IncompatibilityError: If no executable plan can provide the requested
                number of distinct pending-free actions.
        """
        missing = set(self.model.missing_nodes) | set(
            self.model.missing_feasibility_nodes
        )
        order = {name: index for index, name in enumerate(self.dag.topological_order)}
        plans = sorted(
            self._executable_plans(),
            key=lambda plan: (
                plan.experiment_id is None,
                plan.node not in missing,
                order[plan.node],
            ),
        )
        for plan in plans:
            actions = self._random_actions_for_plan(plan, batch_size, pending)
            if actions is not None:
                return actions
        raise IncompatibilityError(
            f"No executable node provides {batch_size} distinct pending-free "
            "random actions. Report pending measurements or reduce the batch size."
        )

    def _fit_node_models(self) -> None:
        """Fit the node models from the current measurements.

        The fit is partial-safe, so upstream-only measurements already train
        their node models before complete DAG rows exist.
        """
        if not self.data.measurements and self.data.complete_measurements.empty:
            return
        self.model.fit_node_data(
            {
                node.name: self.data.node_training_data(self.dag, node.name)
                for node in self.dag.nodes
            },
            allow_partial=True,
            feasibility_frames={
                node.name: self.data.node_feasibility_training_data(self.dag, node.name)
                for node in self.dag.nodes
                if node.feasibility_conditions
            },
        )

    def _recommend_optimum(
        self,
        batch_size: int,
        pending: tuple[_PendingAction, ...],
        *,
        acquisition_function: AcquisitionFunction,
        score_scale: ScoreScale,
    ) -> list[_PendingAction]:
        """Project globally scored configurations onto executable node actions.

        The configured stock acquisition is evaluated on the final DAG posterior
        over the complete discrete search space. Pending local actions are not
        treated as complete terminal measurements; instead, one hypothetical
        network model is conditioned per joint pending outcome and the
        acquisition values are averaged in their declared linear or logarithmic
        gain representation. Each global configuration is projected onto each
        compatible executable plan; that concrete local action is then weighted
        by its own-node success probability and cost before ranking.

        This projection is an orchestration rule, not an unchanged local
        information-value acquisition. The local risk factor uses only the
        selected node's own parameters and the real parent outputs of its
        concrete plan.

        Args:
            batch_size: The number of node actions to recommend.
            pending: The currently reserved local actions, whose outcomes are
                fantasized when scoring.
            acquisition_function: The resolved stock acquisition function,
                including its actual default for this objective.
            score_scale: The acquisition function's declared gain representation.

        Returns:
            The selected executable actions in descending score order.

        Raises:
            IncompatibilityError: If the DAG search space is not purely discrete,
                an acquisition score is invalid, pending actions cannot be
                expressed by the adapter, or fewer distinct executable actions
                than requested are available.
            InfeasibilityError: If no candidate reaches the network success threshold.
        """
        searchspace = self.dag.full_searchspace
        if not searchspace.continuous.is_empty:
            raise IncompatibilityError(
                f"{self.__class__.__name__} scores every configuration of the "
                f"DAG and therefore requires a purely discrete search space."
            )

        candidates = searchspace.discrete.exp_rep.reset_index(drop=True)
        if candidates.empty:
            raise IncompatibilityError(
                "The discrete DAG search space contains no configurations."
            )

        if any(node.feasibility_conditions for node in self.dag.nodes):
            import torch

            from baybe.utils.dataframe import to_tensor

            points = to_tensor(searchspace.transform(candidates)).unsqueeze(-2)
            with torch.no_grad():
                _, probabilities = self.model.draw_network_outcomes(
                    points, self.model.num_network_samples
                )
            feasible = probabilities.mean(dim=0).squeeze(-1).cpu().numpy()
            candidates = candidates.loc[feasible >= self.feasibility_threshold]
            candidates = candidates.reset_index(drop=True)
            if candidates.empty:
                raise InfeasibilityError(
                    "No candidate reaches the network feasibility threshold "
                    f"{self.feasibility_threshold}."
                )

        values = self._stock_acquisition_values(
            candidates, pending, acquisition_function, score_scale
        )
        import torch

        values_tensor = torch.as_tensor(values)
        plans = self._executable_plans()
        if not plans:
            raise IncompatibilityError(
                "No executable node remains in any open experiment."
            )

        options: list[tuple[float, int, int, _PendingAction]] = []
        for plan_index, plan in enumerate(plans):
            candidate_indices = np.flatnonzero(
                self._compatible_candidates(candidates, plan)
            )
            actions = [
                self._project_candidate(plan, candidates.iloc[candidate_index])
                for candidate_index in candidate_indices
            ]
            if not actions:
                continue
            action_probabilities = self._action_success_probabilities(actions)
            candidate_index_tensor = torch.as_tensor(candidate_indices)
            action_values = values_tensor.index_select(0, candidate_index_tensor)
            probability_tensor = action_values.new_tensor(action_probabilities)
            costs = action_values.new_full(
                action_values.shape, self.dag.node_by_name[plan.node].cost
            )
            weighted_scores = _score_actions(
                action_values,
                probability_tensor,
                costs,
                scale=score_scale,
            )
            for index, (candidate_index, action) in enumerate(
                zip(candidate_indices, actions)
            ):
                if action_probabilities[index] == 0.0:
                    continue
                options.append(
                    (
                        float(weighted_scores[index].item()),
                        plan_index,
                        int(candidate_index),
                        action,
                    )
                )

        options.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected: list[_PendingAction] = []
        seen: set[tuple[str, tuple[tuple[str, Any], ...]]] = set()
        for _, _, _, action in options:
            key = (action.node, tuple(sorted(action.inputs.items())))
            if key in seen or contains_action(
                pending, node=action.node, inputs=action.inputs
            ):
                continue
            seen.add(key)
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

    def _compatible_candidates(
        self, candidates: pd.DataFrame, plan: _ExecutablePlan
    ) -> np.ndarray:
        """Return the global configurations compatible with an executable plan.

        A configuration is compatible when it agrees with every own parameter
        value already measured in the plan's experiment. Parent outputs are
        predictions rather than candidate columns and therefore remain supplied
        by the plan's real evaluation references.

        Args:
            candidates: The global DAG configurations.
            plan: The executable node context.

        Returns:
            A Boolean mask over the candidate rows.
        """
        compatible = np.ones(len(candidates), dtype=bool)
        if plan.experiment_id is None:
            return compatible

        experiment = self._experiments[plan.experiment_id]
        for node_name, ref in experiment.bindings.items():
            evaluation = self.data.get_evaluation(self.dag, ref)
            node = self.dag.node_by_name[node_name]
            for parameter in node.inputs:
                expected = evaluation.inputs[parameter.name]
                values = candidates[parameter.name]
                if parameter.is_numerical:
                    numeric = values.to_numpy(dtype=float, na_value=np.nan)
                    compatible &= np.isfinite(numeric) & np.isclose(
                        numeric,
                        float(expected),
                        rtol=0.0,
                        atol=float(getattr(parameter, "tolerance", 0.0)) + 1e-12,
                    )
                else:
                    compatible &= values.eq(expected).fillna(False).to_numpy(dtype=bool)
        return compatible

    def _project_candidate(
        self, plan: _ExecutablePlan, candidate: pd.Series
    ) -> _PendingAction:
        """Project one global configuration onto the plan's executable node.

        Args:
            plan: The executable node context.
            candidate: One global DAG configuration.

        Returns:
            The concrete local action carrying the node's own parameters and the
            plan's real parent outputs.
        """
        node = self.dag.node_by_name[plan.node]
        inputs = {
            parameter.name: candidate[parameter.name] for parameter in node.inputs
        }
        inputs.update(self._parent_outputs(plan.parent_refs))
        return _PendingAction(
            node=node.name, inputs=inputs, parent_refs=plan.parent_refs
        )

    def _action_success_probabilities(
        self, actions: list[_PendingAction]
    ) -> np.ndarray:
        """Return local success probabilities for concrete next actions.

        Args:
            actions: Actions of one node, each carrying own inputs and real parents.

        Returns:
            One probability per action; unconfigured nodes receive probability one.
        """
        node = self.dag.node_by_name[actions[0].node]
        if not node.feasibility_conditions:
            return np.ones(len(actions), dtype=float)
        predictions = self.model.predict_feasibility(
            node.name, pd.DataFrame([dict(action.inputs) for action in actions])
        )
        return predictions.to_numpy(dtype=float)

    def _stock_acquisition_values(
        self,
        candidates: pd.DataFrame,
        pending: tuple[_PendingAction, ...],
        acquisition_function: AcquisitionFunction,
        score_scale: ScoreScale,
    ) -> np.ndarray:
        """Average the stock acquisition over hypothetical pending outcomes.

        Args:
            candidates: The global DAG configurations to score.
            pending: The currently reserved local actions.
            acquisition_function: The resolved stock acquisition function.
            score_scale: The acquisition function's declared gain representation.

        Returns:
            One averaged acquisition value per candidate.

        Raises:
            IncompatibilityError: If a pending context is unsupported or an
                acquisition value is invalid outside the valid ``-inf`` log-gain
                case.
        """
        self._validate_stock_pending(pending)
        models = _condition_on_pending(
            self.model,
            pending,
            num_fantasies=_PENDING_FANTASIES,
            sampling_seed=_PENDING_FANTASY_SEED,
        )
        if not models:
            raise IncompatibilityError(
                "Pending conditioning returned no stock acquisition branches."
            )
        import torch

        branches: list[Any] = []
        for model in models:
            recommender = BotorchRecommender(
                surrogate_model=_AcquisitionSurrogateAdapter(model),
                acquisition_function=acquisition_function,
            )
            branch = recommender.acquisition_values(
                candidates,
                self.dag.full_searchspace,
                self.dag.objective,
                self.data.complete_measurements,
            ).to_numpy(dtype=float)
            branches.append(torch.as_tensor(branch))
        reduced = _reduce_score_branches(torch.stack(branches), scale=score_scale)
        return reduced.detach().cpu().numpy()

    def _validate_stock_pending(self, pending: tuple[_PendingAction, ...]) -> None:
        """Reject terminal pending that the stock reference adapter cannot express.

        Upstream pending supports averaging over conditioned local models.
        Terminal pending also changes observed incumbents/reference data, which
        this adapter cannot represent without a valid hypothetical global row.

        Args:
            pending: The reserved local actions.

        Raises:
            IncompatibilityError: If a terminal execution is pending.
        """
        if any(action.node in self.dag.final_nodes for action in pending):
            raise IncompatibilityError(
                "Stock acquisitions do not support terminal pending actions: "
                "the adapter cannot update hypothetical global reference data "
                "or incumbents. Report terminal measurements before requesting "
                "another stock recommendation. Upstream pending remains supported."
            )


_campaign_unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    MultiStepCampaign, converter, _cattrs_include_init_false=True
)
_campaign_structure_hook = cattrs.gen.make_dict_structure_fn(
    MultiStepCampaign, converter, _cattrs_include_init_false=True
)


def _structure_multistep_campaign(
    specs: dict[str, Any], cls: type[MultiStepCampaign]
) -> MultiStepCampaign:
    """Restore a campaign snapshot without fitted models.

    Args:
        specs: The dictionary representation of the campaign.
        cls: The campaign class to instantiate.

    Returns:
        The restored campaign.

    Raises:
        TypeError: If the serialized type marker names a different class.
    """
    specs = dict(specs)
    type_ = specs.pop("type", None)
    if type_ is not None and type_ != cls.__name__:
        raise TypeError(
            f"The type field '{type_}' does not match the target class "
            f"'{cls.__name__}'."
        )
    return _campaign_structure_hook(specs, cls)


converter.register_unstructure_hook(MultiStepCampaign, _campaign_unstructure_hook)
converter.register_structure_hook(MultiStepCampaign, _structure_multistep_campaign)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
