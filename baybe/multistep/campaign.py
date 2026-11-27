"""Multi-step campaign facade for DAG-structured experiments.

The facade owns the real node measurements and fits a single
:class:`~baybe.multistep.model.MultiStepModel` composed of one node model per DAG
node. A node model may expose multiple independently fitted output models, while
the campaign records their measurements and feasibility as one execution.
Executable node actions are derived from the stored process state, independently
of the configured acquisition function. Graph-aware acquisitions score those
actions directly; stock acquisition functions score global configurations that
are projected onto the same executable actions. A shared scoring layer then
applies local success risk and the selected scoring costs before the final action is
selected.

Every recommendation reserves its concrete local action as pending and carries
an explicit experiment identifier. Measurements are dataframes that carry the
identifier back, may arrive in any order, and are matched by
``(experiment_id, node)``. Complete DAG rows are never stored as a second data
source; they are temporary views derived from the real measurements. Pending
actions are never treated as completed terminal measurements.
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Mapping
from copy import deepcopy
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field, setters
from attrs.validators import ge, in_, instance_of, le

from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import (
    IncompatibilityError,
    IncompatibleAcquisitionFunctionError,
)
from baybe.recommenders.meta.sequential import TwoPhaseMetaRecommender
from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender
from baybe.recommenders.pure.experimental_feasibility import InfeasibleMeasurementMode
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.serialization import SerialMixin, converter

from ._columns import EXPERIMENT_ID, NODE
from ._model_reference_ei import (
    ModelReferenceExpectedImprovement,
    _draw_model_reference_seed,
)
from .acquisition import (
    CostMode,
    GraphAwareAcquisitionProtocol,
    PartialKnowledgeGradientFN,
    ScoreScale,
    _make_node_actions,
    _resolve_score_scale,
    recommend_graph_aware,
    recommend_stock,
)
from .dag import MultiStepDAG
from .data import (
    DuplicateMeasurementWarning,
    ExecutionMode,
    MultiStepData,
    NodeMeasurement,
    RepeatedMeasurementWarning,
    _EvaluationRef,
    _ExecutablePlan,
    _Experiment,
    _get_feasibility_label,
    _get_output_values,
    _history_inputs,
    _identifier_number,
    _inputs_match,
    _PendingAction,
    _validate_node_inputs,
    contains_action,
    executable_nodes,
    experiment_actions,
    failed_nodes,
    make_executable_plans,
    pending_actions,
    register_action,
)
from .node import MultiStepNode

if TYPE_CHECKING:
    from .model import MultiStepModel


def _is_missing(value: Any) -> bool:
    """Return whether a scalar measurement value is absent."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _convert_acquisition(
    value: AcquisitionFunction | GraphAwareAcquisitionProtocol | str | None,
) -> AcquisitionFunction | GraphAwareAcquisitionProtocol | None:
    """Resolve acquisition names and accept structural graph-aware policies."""
    if value is None or isinstance(
        value, (AcquisitionFunction, GraphAwareAcquisitionProtocol)
    ):
        return value
    if isinstance(value, str):
        return convert_acqf(value)
    raise TypeError("Expected an acquisition function, graph-aware policy, or name.")


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


@define(slots=False)
class MultiStepCampaign(SerialMixin):
    """Stateful facade for multi-step Bayesian optimization.

    Node surrogates that produce ensemble posteriors, such as
    :class:`~baybe.surrogates.random_forest.RandomForestSurrogate`, can drive
    recommendations over discrete search spaces. Because ensemble posteriors are
    not differentiable, continuous search spaces require gradient-capable node
    surrogates.

    Executable actions are derived from the DAG, the real measurements, and the
    configured execution mode. They are the same for graph-aware and stock
    acquisition functions. The acquisition scores are weighted once by local
    success risk and the selected scoring costs before the final action is selected.

    In process mode, actions continue their concrete experiments; independent
    fork branches are offered together, and a join waits for all required real
    parent outputs. In functional mode, successful real parent outputs can be
    combined across experiments. Every recommendation is registered as a pending
    local execution in an explicitly assigned experiment.

    The default two-phase recommender starts with random executable actions and
    switches to the acquisition after one real node measurement. Unobserved nodes
    use unconditioned output and feasibility GP priors after the switch.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    # ``add_measurements`` validates the store explicitly because the accepted
    # numerical tolerance is a per-call argument, so the default setattr validation
    # would enforce the strict default instead of the call's setting.
    data: MultiStepData = field(factory=MultiStepData.empty, on_setattr=setters.convert)
    """The measurement store holding the real node observations."""

    acquisition_function: AcquisitionFunction | GraphAwareAcquisitionProtocol | None = (
        field(default=None, converter=_convert_acquisition)
    )
    """The graph-aware policy or stock acquisition function.

    ``None`` selects :class:`~baybe.multistep.PartialKnowledgeGradientFN`.
    This default requires ``batch_size=1``, a scalar numerical objective, and node
    surrogates that support GP priors and fantasy conditioning.
    Graph-aware policies that do not derive from BayBE's
    :class:`~baybe.acquisition.base.AcquisitionFunction` can be used at runtime,
    but cannot currently be serialized as part of a campaign.
    """

    recommender: TwoPhaseMetaRecommender = field(
        factory=TwoPhaseMetaRecommender,
        validator=instance_of(TwoPhaseMetaRecommender),
        kw_only=True,
    )
    """Two-phase selector; its random stage is adapted to executable node actions.

    Only ``RandomRecommender`` and ``BotorchRecommender`` stages are supported.
    The latter selects this campaign's acquisition, or its own explicitly configured
    acquisition if the campaign has none. Other BoTorch settings are rejected.
    ``switch_after`` counts recorded node measurements, including failures and unknown
    feasibility labels, but excludes pending actions.
    """

    execution_mode: ExecutionMode = field(
        default="process",
        validator=in_(("process", "functional")),
    )
    """How executable actions are derived from the recorded process state."""

    cost_mode: CostMode = field(
        default="remaining", validator=in_(("node", "remaining")), kw_only=True
    )
    """Which existing cost value weights each acquisition gain.

    ``"node"`` uses the cost of the next node execution. ``"remaining"`` uses
    the precomputed cost remaining for the executable process and is the default.
    """

    allow_repeats: bool = field(default=True, validator=instance_of(bool))
    """Whether a repeated real execution is stored and trained as a new observation.

        When disabled, a repeated result is discarded with a specific warning and
        the context continues from an already-known observation when its concrete
        parent references match.
    """

    feasibility_threshold: float = field(
        default=1e-6,
        validator=[instance_of(float), ge(0.0), le(1.0)],
        kw_only=True,
    )
    """Minimum local feasibility scoring factor required for an executable action.

    The threshold is applied to the shared joint-success classifier before expensive
    candidate fantasies. A probability of exactly zero is never admissible, even when
    the threshold is zero.
    """

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
        acquisition_function: (
            AcquisitionFunction | GraphAwareAcquisitionProtocol | str | None
        ) = None,
        data: MultiStepData | None = None,
        execution_mode: ExecutionMode = "process",
        allow_repeats: bool = True,
        feasibility_threshold: float = 1e-6,
        cost_mode: CostMode = "remaining",
        recommender: TwoPhaseMetaRecommender | None = None,
    ) -> MultiStepCampaign:
        """Create a campaign for the given DAG.

        Args:
            dag: The DAG topology, including the optimization objective.
            acquisition_function: Optional acquisition function or its name.
            data: Optional initial measurement store.
            execution_mode: Whether actions continue concrete experiments
                (``"process"``) or combine known parent outputs across experiments
                (``"functional"``).
            allow_repeats: Whether repeated real executions are stored and trained.
            feasibility_threshold: Minimum local feasibility scoring factor for an
                executable action.
            cost_mode: Whether to score with the next node's cost or the remaining
                process cost. Defaults to ``"remaining"``.
            recommender: Optional two-phase selector for the random and acquisition
                stages. Defaults to switching after one recorded node measurement.

        Returns:
            The configured campaign.
        """
        return cls(
            dag=dag,
            data=MultiStepData.empty() if data is None else data,
            acquisition_function=acquisition_function,
            execution_mode=execution_mode,
            allow_repeats=allow_repeats,
            feasibility_threshold=feasibility_threshold,
            cost_mode=cost_mode,
            recommender=TwoPhaseMetaRecommender()
            if recommender is None
            else recommender,
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
        for experiment in self.data._contexts.values():
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
        must agree with the recommendation that reserved the action. Historical
        features are resolved from the concrete referenced observations; repeated
        history columns in the dataframe are optional but must agree.
        Successful measurements and explicit failures are stored once per identity.
        A repeated local execution in another experiment is stored only when
        repeats are enabled.

        Args:
            data: One row per node measurement, holding ``experiment_id``, ``node``,
                the node's own parameters, optional history columns, every declared
                node-output column, and any auxiliary columns referenced by
                feasibility conditions. Missing condition values produce an unknown
                joint label unless another observed condition establishes failure;
                missing outputs remain allowed only after a known failure.
            numerical_measurements_must_be_within_tolerance: Flag indicating if
                numerical parameters need to be within their tolerances.

        Raises:
            IncompatibilityError: If the context or node is unknown, a node has no
                pending action, a duplicate or conflicting row is supplied, or the
                values violate the measurement contract.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        if data.empty:
            raise IncompatibilityError("Cannot add an empty measurement batch.")
        if not data.columns.is_unique:
            raise IncompatibilityError("Measurement columns must be unique.")

        # Validate the existing store once; the batch itself is validated per row
        # with the tolerance setting of this call.
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

        experiments = dict(self.data._contexts)
        store = self.data
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

            # A binding may also be an adopted parent evaluation from another
            # experiment, so identity is determined by the measurements actually
            # stored for this experiment.
            stored_bindings = store.experiment_bindings(experiment_id)
            if node.name in stored_bindings:
                existing = store.get_evaluation(self.dag, stored_bindings[node.name])
                candidate = self._make_experiment_measurement(
                    node,
                    cast(Mapping[str, Any], values),
                    existing.inputs,
                    experiment_id,
                    numerical_measurements_must_be_within_tolerance,
                    parent_refs=existing.parent_refs,
                )
                if not self._measurements_match(node, existing, candidate):
                    raise IncompatibilityError(
                        f"The measurement of node '{node.name}' in experiment "
                        f"'{experiment_id}' conflicts with the already stored "
                        "observation of the same identity."
                    )
                warnings.warn(
                    DuplicateMeasurementWarning(
                        f"Node '{node.name}' in experiment '{experiment_id}' was "
                        "reported again with identical data. The duplicate was "
                        "ignored; no additional observation was stored."
                    ),
                    stacklevel=2,
                )
                continue

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
                action.inputs,
                experiment_id,
                numerical_measurements_must_be_within_tolerance,
                parent_refs=action.parent_refs,
            )

            if (
                not self.allow_repeats
                and (
                    matched := store.matching_evaluation(
                        self.dag, node.name, measurement.inputs
                    )
                )
                is not None
            ):
                known_ref, known_measurement = matched
                warnings.warn(
                    RepeatedMeasurementWarning(
                        f"Node '{node.name}' in experiment '{experiment_id}' repeats "
                        "an already stored local execution. Repeats are disabled, "
                        "so the new result was discarded."
                    ),
                    stacklevel=2,
                )
                updated = experiment.without_pending(node.name)
                if _get_feasibility_label(node, measurement) is True and dict(
                    known_measurement.parent_refs
                ) == dict(action.parent_refs):
                    updated = updated.with_binding(
                        node.name, known_ref, action.parent_refs
                    )
                else:
                    updated = updated.closed_copy()
                experiments[experiment_id] = updated
                continue

            ref = _EvaluationRef(index=len(store.measurements), node=node.name)
            store = store.add(measurement)
            experiments[experiment_id] = experiment.with_binding(
                node.name, ref, action.parent_refs
            ).without_pending(node.name)

        # Publish only after the complete batch has been validated.
        self.data = store.with_process_state(experiments, self.data._experiment_counter)

    def _measurements_match(
        self,
        node: MultiStepNode,
        stored: NodeMeasurement,
        candidate: NodeMeasurement,
    ) -> bool:
        """Return whether two reports describe the same stored observation.

        Args:
            node: The observed node.
            stored: The already stored observation.
            candidate: The freshly resolved duplicate report.

        Returns:
            Whether the reports agree within the local-input tolerance and exactly
            on the real output and auxiliary condition values.
        """
        return (
            stored.node == candidate.node
            and _inputs_match(node, stored.inputs, candidate.inputs)
            and _get_output_values(node, stored) == _get_output_values(node, candidate)
            and dict(stored.condition_values) == dict(candidate.condition_values)
            and dict(stored.parent_refs) == dict(candidate.parent_refs)
        )

    def _make_experiment_measurement(
        self,
        node: MultiStepNode,
        values: Mapping[str, Any],
        expected_inputs: Mapping[str, Any],
        experiment_id: str,
        numerical_measurements_must_be_within_tolerance: bool = True,
        *,
        parent_refs: Mapping[str, _EvaluationRef] | None = None,
    ) -> NodeMeasurement:
        """Resolve one measured row into an observation of a context.

        Args:
            node: The observed node.
            values: One row of the measurement dataframe, including the node and
                experiment identifier.
            expected_inputs: The complete local input reserved by the pending
                action or already stored for the identity.
            experiment_id: The experiment the observation belongs to.
            numerical_measurements_must_be_within_tolerance: If ``True``, numerical
                parameter values must lie within the parameter range or tolerance.
            parent_refs: The concrete successful parent observations used by the
                reserved action.

        Returns:
            The resolved node observation.

        Raises:
            IncompatibilityError: If the columns, the own parameter values, or the
                repeated parent outputs are invalid or disagree with the action.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        auxiliary_columns = {
            condition.column
            for condition in node.feasibility_conditions
            if condition.column not in node.output_names
        }
        allowed = {
            NODE,
            EXPERIMENT_ID,
            *self.dag.context.input_names(node),
            *node.output_names,
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

        inputs = dict(expected_inputs)
        for parameter in own_parameters:
            measured = values[parameter.name]
            expected = expected_inputs[parameter.name]
            if not _values_match(parameter, measured, expected):
                raise IncompatibilityError(
                    f"The value {measured!r} of parameter '{parameter.name}' does not "
                    f"match the recommended value {expected!r} for node "
                    f"'{node.name}' in experiment '{experiment_id}'."
                )
            inputs[parameter.name] = measured

        for name in self.dag.context.input_names(node):
            if name in own_names or name not in values or _is_missing(values[name]):
                continue
            if values[name] != expected_inputs[name]:
                raise IncompatibilityError(
                    f"The history feature '{name}' in the measurements for node "
                    f"'{node.name}' does not match the value bound to the "
                    f"recommended action in experiment "
                    f"'{experiment_id}'."
                )

        _validate_node_inputs(
            node,
            pd.DataFrame([inputs]),
            numerical_measurements_must_be_within_tolerance,
            context=self.dag.context,
        )

        output_values = {
            name: None if _is_missing(values.get(name)) else values.get(name)
            for name in node.output_names
        }
        measurement = NodeMeasurement(
            node=node.name,
            inputs=inputs,
            output=(
                output_values[node.output_names[0]]
                if len(node.output_names) == 1
                else output_values
            ),
            experiment_id=experiment_id,
            condition_values={
                column: values[column]
                for column in auxiliary_columns
                if column in values and not _is_missing(values[column])
            },
            parent_refs={} if parent_refs is None else parent_refs,
        )
        _get_feasibility_label(node, measurement)
        return measurement

    def _validate_configuration_prerequisites(self) -> None:
        """Validate the shared requirements of the fixed-configuration API."""
        if self.execution_mode != "process":
            raise IncompatibilityError(
                "Full-configuration attempts require process execution mode."
            )
        if not self.allow_repeats:
            raise IncompatibilityError(
                "Full-configuration attempts require allow_repeats=True."
            )
        unsupported_modes = [
            node.name
            for node in self.dag.nodes
            if node.infeasible_measurement_mode
            not in (
                InfeasibleMeasurementMode.OBSERVED,
                InfeasibleMeasurementMode.EXCLUDE,
            )
        ]
        if unsupported_modes:
            raise IncompatibilityError(
                "Full-configuration attempts support only OBSERVED and EXCLUDE "
                f"failure training modes; unsupported nodes: {unsupported_modes}."
            )
        self.dag._configuration_space.validate_numerical_parameters()

    def _has_open_process_context(self) -> bool:
        """Return whether any known process context is incomplete and active."""
        identifiers = set(self.data._contexts)
        identifiers.update(
            measurement.experiment_id
            for measurement in self.data.measurements
            if measurement.experiment_id is not None
        )
        for experiment_id in identifiers:
            context = self.data._contexts.get(experiment_id)
            if context is None:
                bindings = self.data.experiment_bindings(experiment_id)
                failed = any(
                    _get_feasibility_label(
                        self.dag.node_by_name[name],
                        self.data.get_evaluation(self.dag, ref),
                    )
                    is False
                    for name, ref in bindings.items()
                )
                if not failed and not self.data.is_experiment_complete(
                    self.dag, experiment_id
                ):
                    return bool(bindings)
                continue

            if context.closed or failed_nodes(context, self.dag, self.data):
                continue
            if self.data.is_experiment_complete(self.dag, experiment_id):
                continue
            if context.configuration is not None:
                return True
            if context.bindings or context.pending:
                return True
        return False

    def _has_open_configuration(self) -> bool:
        """Return whether a reserved complete configuration is still active."""
        return any(
            context.configuration is not None
            and not context.closed
            and not failed_nodes(context, self.dag, self.data)
            and not self.data.is_experiment_complete(self.dag, experiment_id)
            for experiment_id, context in self.data._contexts.items()
        )

    def register_configuration(self, configuration: Mapping[str, Any]) -> pd.DataFrame:
        """Reserve one complete parameter configuration without a local action.

        Args:
            configuration: Exactly the DAG's own parameter names and values.

        Returns:
            A one-row frame containing the new ``experiment_id`` and all own
            parameters in DAG declaration order.

        Raises:
            IncompatibilityError: If prerequisites, the configuration, or current
                process state are invalid.
            TypeError: If the input is not a mapping.
        """
        self._validate_configuration_prerequisites()
        if self._has_open_process_context():
            raise IncompatibilityError(
                "A new full configuration cannot start while another process "
                "context is open."
            )
        if not isinstance(configuration, Mapping):
            raise TypeError("A configuration must be a mapping of parameter values.")
        configuration_copy = dict(configuration)
        self.dag._configuration_space.validate(configuration_copy)

        identifier_number = self._first_new_identifier()
        experiment_id = f"exp-{identifier_number}"
        context = _Experiment(
            experiment_id=experiment_id,
            configuration=configuration_copy,
        )
        contexts = dict(self.data._contexts)
        contexts[experiment_id] = context
        updated = self.data.with_process_state(contexts, identifier_number)
        self.data = updated
        return pd.DataFrame(
            [{EXPERIMENT_ID: experiment_id, **configuration_copy}],
            columns=[EXPERIMENT_ID, *self.dag._configuration_space.parameter_names],
        )

    def sample_configuration(self, *, sampling_seed: int) -> pd.DataFrame:
        """Uniformly sample and reserve one complete numeric configuration.

        Args:
            sampling_seed: Non-negative integer seed for the local sampler.

        Returns:
            A one-row frame containing the new ``experiment_id`` and all own
            parameters in DAG declaration order.

        Raises:
            IncompatibilityError: If the configuration prerequisites or process state
                are invalid.
            TypeError: If ``sampling_seed`` is not an integer.
            ValueError: If ``sampling_seed`` is negative.
        """
        self._validate_configuration_prerequisites()
        if self._has_open_process_context():
            raise IncompatibilityError(
                "A new full configuration cannot start while another process "
                "context is open."
            )
        configuration = self.dag._configuration_space.sample(
            sampling_seed=sampling_seed
        )
        return self.register_configuration(configuration.iloc[0].to_dict())

    def get_next_configuration_action(self, experiment_id: str) -> pd.DataFrame:
        """Reserve or return the first executable action for a fixed configuration.

        Args:
            experiment_id: The identifier returned when the configuration was
                registered.

        Returns:
            A one-row local action, or an empty frame for a completed or failed
            process.

        Raises:
            IncompatibilityError: If the identifier is unknown, the attempt is not a
                configuration, feasibility is unknown, or stored state is inconsistent.
        """
        self.data.validate(
            self.dag, numerical_measurements_must_be_within_tolerance=False
        )
        context = self.data._contexts.get(experiment_id)
        if context is None:
            raise IncompatibilityError(
                f"Unknown configuration experiment identifier '{experiment_id}'."
            )
        if context.configuration is None:
            raise IncompatibilityError(
                f"Experiment '{experiment_id}' is not a full-configuration attempt."
            )

        for node_name, ref in context.bindings.items():
            node = self.dag.node_by_name[node_name]
            label = _get_feasibility_label(
                node, self.data.get_evaluation(self.dag, ref)
            )
            if label is None:
                raise IncompatibilityError(
                    f"Feasibility of node '{node_name}' in configuration "
                    f"'{experiment_id}' is unknown; execution is blocked."
                )
            if label is False:
                return pd.DataFrame()

        actions = experiment_actions(context)
        if actions:
            if len(actions) != 1:
                raise IncompatibilityError(
                    f"Configuration '{experiment_id}' has multiple pending actions."
                )
            return self._configuration_action_frame(actions[0])
        if self.data.is_experiment_complete(self.dag, experiment_id):
            return pd.DataFrame()
        if context.closed:
            raise IncompatibilityError(
                f"Configuration '{experiment_id}' is closed without a known failure."
            )

        executable = executable_nodes(context, self.dag, self.data)
        if not executable:
            raise IncompatibilityError(
                f"Configuration '{experiment_id}' has no executable node but is not "
                "complete or failed."
            )
        node_name, parent_refs = executable[0]
        node = self.dag.node_by_name[node_name]
        inputs = {
            parameter.name: context.configuration[parameter.name]
            for parameter in node.inputs
        }
        inputs.update(_history_inputs(self.dag, self.data, node_name, parent_refs))
        contexts, counter = register_action(
            self.data._contexts,
            self.data._experiment_counter,
            experiment_id=experiment_id,
            node=node_name,
            inputs=inputs,
            parent_refs=parent_refs,
            data=self.data,
            dag=self.dag,
        )
        self.data = self.data.with_process_state(contexts, counter)
        return self._configuration_action_frame(
            contexts[experiment_id].pending[node_name]
        )

    def _configuration_action_frame(self, action: _PendingAction) -> pd.DataFrame:
        """Format one pending local action without exposing context internals."""
        node = self.dag.node_by_name[action.node]
        columns = [EXPERIMENT_ID, NODE, *self.dag.context.input_names(node)]
        row = {EXPERIMENT_ID: action.experiment_id, NODE: action.node}
        row.update(action.inputs)
        return pd.DataFrame([row], columns=columns)

    def recommend(self, batch_size: int = 1) -> pd.DataFrame:
        """Recommend and reserve the next node actions.

        Executable actions are derived from the DAG, the real measurements, and the
        configured execution mode, independently of the acquisition function. A
        graph-aware acquisition scores those actions directly. A stock acquisition
        scores the global DAG configurations and the values are projected onto the
        same executable actions. The shared scoring layer applies the local success
        probability and the selected cost basis before the final selection.

        Pending real actions are treated as hypothetical model states when
        scoring. They are not counted as observed real measurements.

        The initial random stage bypasses model fitting and acquisition scoring. The
        acquisition stage combines node models with local feasibility predictions.

        Every returned action is registered as pending with its explicit experiment
        identifier. New identifiers are published only after every computation and
        registration succeeded.

        Args:
            batch_size: The number of node actions to recommend.

        Returns:
            A dataframe holding ``experiment_id``, ``node``, and the complete
            local input of each recommended node (its own parameters and the
            configured historical features).

        Raises:
            ValueError: If ``batch_size`` is smaller than 1.
            IncompatibilityError: If the selected acquisition does not support the
                DAG, objective, model, or requested batch size, if required
                acquisition references are unavailable, or if no executable action
                is available.
            InfeasibilityError: If no executable action reaches the local success
                threshold.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but "
                f"provided {batch_size=}."
            )

        if self._has_open_configuration():
            raise IncompatibilityError(
                "Local node recommendations are unavailable while a full "
                "configuration attempt is open."
            )

        self.data.validate(
            self.dag, numerical_measurements_must_be_within_tolerance=False
        )

        if (
            type(self.recommender.initial_recommender) is not RandomRecommender
            or type(self.recommender.recommender) is not BotorchRecommender
        ):
            raise IncompatibilityError(
                "Multi-step two-phase selection requires a RandomRecommender initial "
                "stage and a BotorchRecommender acquisition stage."
            )
        bayesian = cast(BotorchRecommender, self.recommender.recommender)
        if bayesian != BotorchRecommender(
            acquisition_function=bayesian.acquisition_function
        ):
            raise IncompatibilityError(
                "Multi-step campaigns support only the acquisition_function setting "
                "of the BotorchRecommender; configure node surrogates on the DAG."
            )
        if (
            self.acquisition_function is not None
            and bayesian.acquisition_function is not None
        ):
            raise IncompatibilityError(
                "Configure the acquisition either on the campaign or on its "
                "BotorchRecommender, not both."
            )
        selected = self.recommender.select_recommender(
            batch_size=batch_size,
            measurements=pd.DataFrame(index=range(len(self.data.measurements))),
        )
        if isinstance(selected, RandomRecommender):
            return self._recommend_random(batch_size, selected)

        acquisition_function = (
            self.acquisition_function
            if self.acquisition_function is not None
            else bayesian.acquisition_function
        )

        graph_aware = acquisition_function is None or isinstance(
            acquisition_function, GraphAwareAcquisitionProtocol
        )
        policy: GraphAwareAcquisitionProtocol | None = None
        resolved_acquisition: AcquisitionFunction | None = None
        score_scale: ScoreScale | None = None

        if graph_aware:
            selected_policy = (
                cast(GraphAwareAcquisitionProtocol, PartialKnowledgeGradientFN())
                if acquisition_function is None
                else cast(GraphAwareAcquisitionProtocol, acquisition_function)
            )
            policy = selected_policy
            policy.validate_context(self.dag, self.data, batch_size)
        else:
            assert acquisition_function is None or isinstance(
                acquisition_function, AcquisitionFunction
            )
            resolver = BotorchRecommender(acquisition_function=acquisition_function)
            resolved_acquisition = resolver._get_acquisition_function(
                self.dag.objective
            )
            if (
                self.dag.objective.is_multi_output
                and not resolved_acquisition.supports_multi_output
            ):
                raise IncompatibleAcquisitionFunctionError(
                    f"You attempted to use the single-output acquisition function "
                    f"'{resolved_acquisition.__class__.__name__}' for a "
                    f"{len(self.dag.objective.targets)}-target multi-output "
                    "objective."
                )
            if resolved_acquisition.is_analytic:
                raise IncompatibilityError(
                    f"The stock acquisition "
                    f"'{resolved_acquisition.__class__.__name__}' is analytic, but "
                    "the multi-step network posterior is sample-based. Use a "
                    "q-batched, Monte-Carlo acquisition function instead."
                )
            score_scale = _resolve_score_scale(resolved_acquisition)
            if isinstance(resolved_acquisition, ModelReferenceExpectedImprovement):
                resolved_acquisition.validate_context(self.dag, self.data, batch_size)

        call_seed = None
        if isinstance(resolved_acquisition, ModelReferenceExpectedImprovement):
            call_seed = _draw_model_reference_seed()
        self._fit_node_models()
        pending = pending_actions(self.data._contexts)
        plans = self._executable_plans()

        if graph_aware:
            assert policy is not None
            policy.validate_model(self.model)
            actions = recommend_graph_aware(
                self.model,
                plans,
                pending,
                acquisition=policy,
                batch_size=batch_size,
                feasibility_threshold=self.feasibility_threshold,
                cost_mode=self.cost_mode,
            )
        else:
            assert resolved_acquisition is not None
            assert score_scale is not None
            actions = recommend_stock(
                self.model,
                self.data,
                self.data._contexts,
                plans,
                pending,
                acquisition_function=resolved_acquisition,
                score_scale=score_scale,
                batch_size=batch_size,
                feasibility_threshold=self.feasibility_threshold,
                first_new_id=self._first_new_identifier(),
                execution_mode=self.execution_mode,
                cost_mode=self.cost_mode,
                call_seed=call_seed,
            )

        experiments = dict(self.data._contexts)
        counter = self.data._experiment_counter
        columns = [EXPERIMENT_ID, NODE]
        rows: list[dict[str, Any]] = []
        for action in actions:
            experiments, counter = register_action(
                experiments,
                counter,
                experiment_id=action.experiment_id,
                node=action.node,
                inputs=action.inputs,
                parent_refs=action.parent_refs,
                data=self.data,
                dag=self.dag,
            )
            rows.append(
                {
                    EXPERIMENT_ID: action.experiment_id,
                    NODE: action.node,
                    **dict(action.inputs),
                }
            )
            for name in self.dag.context.input_names(
                self.dag.node_by_name[action.node]
            ):
                if name not in columns:
                    columns.append(name)

        # Publish experiment state only after the complete batch has been built.
        result = pd.DataFrame(rows).reindex(columns=columns)
        self.data = self.data.with_process_state(experiments, counter)
        return result

    def _recommend_random(
        self, batch_size: int, recommender: RandomRecommender
    ) -> pd.DataFrame:
        """Sample executable node actions without fitting or scoring models."""
        experiments = dict(self.data._contexts)
        counter = self.data._experiment_counter
        first_new_id = self._first_new_identifier()
        actions: list[_PendingAction] = []
        for _ in range(batch_size):
            pending = pending_actions(experiments)
            plans = make_executable_plans(
                self.dag,
                self.data,
                experiments,
                mode=self.execution_mode,
                first_new_id=max(first_new_id, counter + 1),
            )
            problems = _make_node_actions(self.dag, plans, pending)
            if not problems:
                raise IncompatibilityError(
                    "The random stage found no distinct executable local action."
                )
            names = sorted({problem.node for problem in problems})
            name = str(np.random.choice(names))
            candidates = [problem for problem in problems if problem.node == name]
            plan = candidates[int(np.random.randint(len(candidates)))]
            node = self.dag.node_by_name[name]
            inputs = dict(plan.inputs)
            if node.inputs and not node.searchspace.continuous.is_empty:
                # Discrete coordinates were selected above from the remaining actions.
                discrete = pd.DataFrame(
                    [
                        {
                            key: inputs[key]
                            for key in node.searchspace.discrete.parameter_names
                        }
                    ]
                )
                for _attempt in range(100):
                    values = recommender._recommend_hybrid(
                        node.searchspace, discrete, 1
                    ).iloc[0]
                    inputs.update(values.to_dict())
                    if not contains_action(
                        pending, node=name, inputs=inputs, dag=self.dag
                    ):
                        break
                else:
                    raise IncompatibilityError(
                        f"Could not sample a distinct pending action for node '{name}'."
                    )
            action = _PendingAction(name, inputs, plan.experiment_id, plan.parent_refs)
            experiments, counter = register_action(
                experiments,
                counter,
                experiment_id=action.experiment_id,
                node=action.node,
                inputs=action.inputs,
                parent_refs=action.parent_refs,
                data=self.data,
                dag=self.dag,
            )
            actions.append(action)
        columns = [EXPERIMENT_ID, NODE]
        rows = []
        for action in actions:
            rows.append(
                {
                    EXPERIMENT_ID: action.experiment_id,
                    NODE: action.node,
                    **action.inputs,
                }
            )
            for name in self.dag.context.input_names(
                self.dag.node_by_name[action.node]
            ):
                if name not in columns:
                    columns.append(name)
        result = pd.DataFrame(rows).reindex(columns=columns)
        self.data = self.data.with_process_state(experiments, counter)
        return result

    def _first_new_identifier(self) -> int:
        """Return the first numeric experiment suffix not known to be in use.

        Returns:
            One larger than the largest counter or known experiment suffix.
        """
        numbers = [
            self.data._experiment_counter,
            *(
                _identifier_number(experiment_id)
                for experiment_id in self.data._contexts
            ),
            *(
                _identifier_number(measurement.experiment_id)
                for measurement in self.data.measurements
                if measurement.experiment_id is not None
            ),
        ]
        return max(numbers, default=0) + 1

    def _executable_plans(self) -> tuple[_ExecutablePlan, ...]:
        """Collect every executable action from the current process state.

        Returns:
            The executable plans, including their remaining objective costs.
        """
        return make_executable_plans(
            self.dag,
            self.data,
            self.data._contexts,
            mode=self.execution_mode,
            first_new_id=self._first_new_identifier(),
        )

    def _fit_node_models(self) -> None:
        """Fit the node models from the current measurements.

        Missing outputs receive GP priors, and upstream-only measurements update
        only the nodes for which real data exists.
        """
        self.model._fit_node_data_with_priors(
            {
                node.name: self.data.node_training_data(self.dag, node.name)
                for node in self.dag.nodes
            },
            feasibility_frames={
                node.name: self.data.node_feasibility_training_data(self.dag, node.name)
                for node in self.dag.nodes
                if node.feasibility_conditions
            },
        )


def _unstructure_acquisition(
    value: AcquisitionFunction | GraphAwareAcquisitionProtocol | None,
) -> dict[str, Any] | None:
    """Serialize BayBE acquisitions; reject unsupported external policies."""
    if value is None:
        return None
    if not isinstance(value, AcquisitionFunction):
        raise NotImplementedError(
            f"Serializing graph-aware policies of type '{type(value).__name__}' "
            "is not supported."
        )
    return converter.unstructure(value, unstructure_as=AcquisitionFunction)


def _structure_acquisition(
    value: dict[str, Any] | str | None, _: Any
) -> AcquisitionFunction | None:
    """Restore an optional serialized BayBE acquisition."""
    return None if value is None else converter.structure(value, AcquisitionFunction)


_campaign_unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    MultiStepCampaign,
    converter,
    _cattrs_include_init_false=True,
    acquisition_function=cattrs.gen.override(unstruct_hook=_unstructure_acquisition),
)
_campaign_structure_hook = cattrs.gen.make_dict_structure_fn(
    MultiStepCampaign,
    converter,
    _cattrs_include_init_false=True,
    acquisition_function=cattrs.gen.override(struct_hook=_structure_acquisition),
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
    specs = deepcopy(specs)
    type_ = specs.pop("type", None)
    if type_ is not None and type_ != cls.__name__:
        raise TypeError(
            f"The type field '{type_}' does not match the target class "
            f"'{cls.__name__}'."
        )
    legacy_epsilon = specs.pop("feasibility_epsilon", 0.0)
    if legacy_epsilon != 0.0:
        raise IncompatibilityError(
            "This campaign snapshot uses the removed 'feasibility_epsilon' scoring "
            f"transformation ({legacy_epsilon!r}). Loading it would silently change "
            "recommendation semantics. Migrate the snapshot explicitly by setting "
            "'feasibility_epsilon' to 0.0 only if you accept direct joint-success "
            "probability scoring, then load it again."
        )
    return _campaign_structure_hook(specs, cls)


converter.register_unstructure_hook(MultiStepCampaign, _campaign_unstructure_hook)
converter.register_structure_hook(MultiStepCampaign, _structure_multistep_campaign)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
