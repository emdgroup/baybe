"""Measurement store for multi-step campaigns.

A :class:`NodeMeasurement` is one real observation of one node, including all
outputs produced by that execution. Its inputs are the full input of the node
model: the values of the node's own parameters together with the configured
historical features. Node training views are derived from the stored measurements
on demand. Complete DAG rows are never stored as a second data source; they are
temporary views assembled from concrete observations.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

import cattrs
import numpy as np
import pandas as pd
from attrs import define, evolve, field
from attrs.validators import and_, deep_iterable, ge, instance_of, min_len, optional

from baybe.exceptions import (
    IncompatibilityError,
    IncompleteMeasurementsError,
    InvalidTargetValueError,
)
from baybe.parameters.base import Parameter
from baybe.recommenders.pure.experimental_feasibility import (
    InfeasibleMeasurementMode,
)
from baybe.serialization import SerialMixin, converter
from baybe.utils.basic import to_tuple
from baybe.utils.validation import validate_parameter_input

from ._columns import (
    EXPERIMENT_GROUP,
    EXPERIMENT_ID,
    FEASIBILITY_LABEL,
    MEASUREMENT_ID,
)
from .context import MultiStepContext
from .dag import MultiStepDAG
from .node import FeasibilityCondition, MultiStepNode

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace

ExecutionMode = Literal["process", "functional"]
"""How executable actions are derived from recorded experiments."""


class DuplicateMeasurementWarning(UserWarning):
    """An identical observation was reported again for the same identity."""


class RepeatedMeasurementWarning(UserWarning):
    """A repeated execution was discarded because repeats are disabled."""


def _condition_success(
    node: MultiStepNode,
    values: Mapping[str, Any],
    conditions: tuple[FeasibilityCondition, ...] | None = None,
) -> bool:
    """Evaluate selected node conditions, allowing missing data only after failure.

    Args:
        node: The node whose execution conditions are evaluated.
        values: Observed output and auxiliary values keyed by measurement column.
        conditions: Conditions to evaluate, defaulting to all node conditions.

    Returns:
        Whether all selected conditions pass.

    Raises:
        IncompleteMeasurementsError: If a condition value is missing and no other
            observed condition establishes failure.
        InvalidTargetValueError: If a measured value has an incompatible type or
            is non-finite.
    """
    passed = True
    missing: list[str] = []
    for condition in node.feasibility_conditions if conditions is None else conditions:
        value = values.get(condition.column)
        if (
            value is None
            or value is pd.NA
            or (isinstance(value, (float, np.floating)) and bool(np.isnan(value)))
        ):
            missing.append(condition.column)
            continue
        if condition.operator == "is_true":
            if not isinstance(value, (bool, np.bool_)):
                raise InvalidTargetValueError(
                    f"Feasibility column '{condition.column}' of node '{node.name}' "
                    "must contain Boolean values."
                )
            passed = passed and bool(value)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise InvalidTargetValueError(
                f"Feasibility column '{condition.column}' of node '{node.name}' "
                "must contain numeric values."
            )
        if not np.isfinite(value):
            raise InvalidTargetValueError(
                f"Feasibility column '{condition.column}' of node '{node.name}' "
                "must contain finite values."
            )
        threshold = condition.threshold
        assert threshold is not None
        condition_passed = (
            value <= threshold if condition.operator == "<=" else value >= threshold
        )
        passed = passed and bool(condition_passed)
    if passed and missing:
        raise IncompleteMeasurementsError(
            f"Feasibility values for node '{node.name}' are missing from columns "
            f"{sorted(set(missing))}."
        )
    return passed


def _extract_feasibility_values(
    node: MultiStepNode,
    measurements: pd.DataFrame,
    conditions: tuple[FeasibilityCondition, ...] | None = None,
) -> pd.Series:
    """Derive Boolean labels from selected conditions on each measurement row.

    Args:
        node: The node whose conditions are evaluated.
        measurements: Rows containing the node output and auxiliary measurements.
        conditions: Conditions to evaluate, defaulting to all node conditions.

    Returns:
        A Boolean label for each row, preserving index and using the internal label
        column name.
    """
    labels: list[Any] = []
    for row in measurements.to_dict(orient="records"):
        try:
            label = _condition_success(node, cast(Mapping[str, Any], row), conditions)
        except IncompleteMeasurementsError:
            label = pd.NA
        labels.append(label)
    return pd.Series(labels, index=measurements.index, name=FEASIBILITY_LABEL)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only copy of a mapping."""
    return MappingProxyType(dict(value))


def _finite_output(value: Any, *, node: str) -> float:
    """Validate and convert one real output value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise IncompatibilityError(
            f"The output of node '{node}' must be numeric, got {value!r}."
        )
    if not np.isfinite(value):
        raise IncompatibilityError(
            f"The output of node '{node}' must be finite, got {value!r}."
        )
    return float(value)


def _normalize_measurement_output(
    value: float | Mapping[str, float | None] | None,
) -> float | Mapping[str, float | None] | None:
    """Copy and validate a scalar or named output bundle."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        result: dict[str, float | None] = {}
        for name, measured in value.items():
            if not isinstance(name, str) or not name:
                raise IncompatibilityError(
                    "Measurement output names must be nonempty strings."
                )
            result[name] = (
                None
                if measured is None
                else _finite_output(measured, node="measurement")
            )
        return MappingProxyType(result)
    return _finite_output(value, node="measurement")


def _get_output_values(
    node: MultiStepNode, measurement: NodeMeasurement
) -> dict[str, float | None]:
    """Return a measurement's outputs mapped to the node's declared names."""
    if isinstance(measurement.output, Mapping):
        unexpected = set(measurement.output) - set(node.output_names)
        if unexpected:
            raise IncompatibilityError(
                f"Measurement of node '{node.name}' contains undeclared output keys: "
                f"{sorted(unexpected)}."
            )
        return {name: measurement.output.get(name) for name in node.output_names}
    if len(node.output_names) != 1:
        raise IncompatibilityError(
            f"A scalar measurement for multi-output node '{node.name}' is ambiguous."
        )
    return {node.output_names[0]: measurement.output}


def _output_values_match_inputs(
    node: MultiStepNode,
    measurement: NodeMeasurement,
    inputs: Mapping[str, Any],
) -> bool:
    """Check every output of one successful evaluation against local inputs."""
    return all(
        value is not None and inputs.get(name) == value
        for name, value in _get_output_values(node, measurement).items()
    )


@define(frozen=True)
class _EvaluationRef:
    """Stable internal reference to one actual evaluation of one node."""

    index: int = field(validator=ge(0))
    """The position inside the append-only table."""

    node: str = field()
    """The node whose evaluation is referenced."""


def _structure_evaluation_ref(
    specs: dict[str, Any], _: type[_EvaluationRef]
) -> _EvaluationRef:
    """Reject obsolete reference formats rather than silently migrating them."""
    if set(specs) != {"index", "node"}:
        raise ValueError(
            "An evaluation reference must contain only 'index' and 'node'."
        )
    return _EvaluationRef(
        index=converter.structure(specs["index"], int),
        node=converter.structure(specs["node"], str),
    )


converter.register_structure_hook(_EvaluationRef, _structure_evaluation_ref)


@define(frozen=True)
class _PendingAction:
    """A recommended but not yet reported local node execution."""

    node: str = field()
    """The recommended node."""

    inputs: Mapping[str, Any] = field(converter=_freeze_mapping)
    """The complete local input, including real parent outputs."""

    experiment_id: str = field()
    """The experiment the action was evaluated for, retained until registration."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """The concrete parent evaluations supplying the local input."""


@define(frozen=True)
class _ActionContext:
    """Concrete process and lineage context for graph-aware gain evaluation."""

    experiment_id: str = field()
    """The experiment being continued or created."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """The real parent evaluations supplying this action."""


@define(frozen=True)
class _ExecutablePlan:
    """One executable next node with its concrete experiment context."""

    experiment_id: str = field()
    """The experiment to continue or the prepared identifier of a fresh one."""

    node: str = field()
    """The node that the plan makes executable."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """The concrete real parent evaluations supplying the local inputs."""

    inputs: Mapping[str, Any] = field(factory=dict, converter=_freeze_mapping)
    """The fixed parent outputs of the local input, keyed by column name."""

    remaining_cost: float = field(default=1.0)
    """The total cost still required to complete the objective from this plan."""

    is_new: bool = field(default=False)
    """Whether the plan prepares a new experiment instead of continuing one."""


@define(frozen=True)
class _Experiment:
    """Immutable snapshot of one evaluation context."""

    experiment_id: str = field()
    """The automatically generated context identifier."""

    bindings: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """Actual evaluations bound to the context, keyed by node."""

    pending: Mapping[str, _PendingAction] = field(
        factory=dict, converter=_freeze_mapping
    )
    """Recommended local executions awaiting their result, keyed by node."""

    closed: bool = field(default=False)
    """Whether the experiment was closed without a stored failed observation."""

    def _adopt_parent_bindings(
        self, parent_refs: Mapping[str, _EvaluationRef]
    ) -> dict[str, _EvaluationRef]:
        """Return bindings extended with the action's consistent parent refs."""
        bindings = dict(self.bindings)
        for parent, ref in parent_refs.items():
            existing = bindings.get(parent)
            if existing is not None and existing != ref:
                raise IncompatibilityError(
                    f"Parent node '{parent}' is already bound to a different "
                    f"evaluation in experiment '{self.experiment_id}'."
                )
            bindings[parent] = ref
        return bindings

    def with_binding(  # noqa: DOC502
        self,
        node: str,
        ref: _EvaluationRef,
        parent_refs: Mapping[str, _EvaluationRef],
    ) -> _Experiment:
        """Return a snapshot with one additional evaluation binding.

        Args:
            node: The node whose evaluation becomes available.
            ref: The actual evaluation.
            parent_refs: The concrete parent evaluations used by the action.

        Returns:
            The updated snapshot.
        """
        bindings = self._adopt_parent_bindings(parent_refs)
        bindings[node] = ref
        return evolve(self, bindings=bindings)

    def without_pending(self, node: str) -> _Experiment:
        """Return a snapshot with one pending action removed.

        Args:
            node: The node whose pending action is removed.

        Returns:
            The updated snapshot.
        """
        return evolve(
            self, pending={k: v for k, v in self.pending.items() if k != node}
        )

    def with_pending(self, action: _PendingAction) -> _Experiment:
        """Return a snapshot with an additional pending action.

        Args:
            action: The action to reserve.

        Returns:
            A snapshot reserving the action and referencing its parents.

        Raises:
            IncompatibilityError: If the node already has a pending action, or a
                parent is already bound to a different evaluation.
        """
        if action.node in self.pending:
            raise IncompatibilityError(
                f"Node '{action.node}' already has a pending action in experiment "
                f"'{self.experiment_id}'."
            )

        bindings = self._adopt_parent_bindings(action.parent_refs)
        return evolve(
            self,
            bindings=bindings,
            pending={**self.pending, action.node: action},
        )

    def closed_copy(self) -> _Experiment:
        """Return a snapshot that is closed for further automatic continuation.

        Returns:
            The closed snapshot.
        """
        return evolve(self, closed=True)


@define(frozen=True)
class NodeMeasurement(SerialMixin):
    """Record one node execution and all of its observed outputs."""

    node: str = field(validator=and_(instance_of(str), min_len(1)))
    """The observed node."""

    inputs: Mapping[str, Any] = field(
        converter=lambda value: MappingProxyType(dict(value)),
        validator=instance_of(Mapping),
    )
    """The node's own parameter values and configured historical features."""

    output: float | Mapping[str, float | None] | None = field(
        default=None, converter=_normalize_measurement_output
    )
    """The real scalar output or immutable mapping of jointly observed values."""

    experiment_id: str | None = field(
        default=None, validator=optional(instance_of(str)), eq=False
    )
    """The optional experiment this measurement belongs to."""

    condition_values: Mapping[str, Any] = field(
        factory=dict,
        converter=lambda value: MappingProxyType(dict(value)),
        validator=instance_of(Mapping),
        kw_only=True,
    )
    """Observed auxiliary columns used by this node's feasibility conditions."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict,
        converter=_freeze_mapping,
        validator=instance_of(Mapping),
        kw_only=True,
        eq=False,
    )
    """Concrete parent evaluations used to produce this observation."""

    @output.validator
    def _validate_output(  # noqa: DOC101, DOC103
        self, _: Any, value: float | Mapping[str, float | None] | None
    ) -> None:
        """Validate that every present output value is finite and numeric."""
        if value is None:
            return
        if isinstance(value, Mapping):
            for item in value.values():
                if item is not None:
                    _finite_output(item, node=self.node)
        else:
            _finite_output(value, node=self.node)


def _structure_measurement_output(value: Any, _: Any) -> Any:
    """Pass output values to the defensive converter on ``NodeMeasurement``."""
    return value


_measurement_unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    NodeMeasurement,
    converter,
    output=cattrs.gen.override(
        unstruct_hook=lambda value: dict(value) if isinstance(value, Mapping) else value
    ),
)
_measurement_structure_hook = cattrs.gen.make_dict_structure_fn(
    NodeMeasurement,
    converter,
    output=cattrs.gen.override(struct_hook=_structure_measurement_output),
)


def _structure_node_measurement(
    specs: Mapping[str, Any], cls: type[NodeMeasurement]
) -> NodeMeasurement:
    """Restore a measurement while preserving scalar and bundle output forms."""
    values = dict(specs)
    type_ = values.pop("type", None)
    if type_ is not None and type_ != cls.__name__:
        raise TypeError(
            f"The type field '{type_}' does not match the target class "
            f"'{cls.__name__}'."
        )
    return _measurement_structure_hook(values, cls)


converter.register_unstructure_hook(NodeMeasurement, _measurement_unstructure_hook)
converter.register_structure_hook(NodeMeasurement, _structure_node_measurement)


@define(frozen=True, eq=False)
class MultiStepData(SerialMixin):
    """Immutable store of real node measurements.

    Snapshots containing the obsolete source-tagged evaluation references are
    incompatible and must not be loaded as the current format.
    """

    measurements: tuple[NodeMeasurement, ...] = field(
        factory=tuple,
        converter=to_tuple,
        validator=deep_iterable(instance_of(NodeMeasurement)),
    )
    """The individual node measurements, in insertion order."""

    _contexts: Mapping[str, _Experiment] = field(
        factory=dict,
        converter=_freeze_mapping,
        validator=instance_of(Mapping),
        eq=False,
    )
    """The canonical process contexts and their pending actions."""

    _experiment_counter: int = field(default=0, validator=ge(0), eq=False)
    """The last allocated numeric experiment identifier suffix."""

    @classmethod
    def empty(cls) -> MultiStepData:
        """Create an empty store.

        Returns:
            An empty store.
        """
        return cls()

    def add(self, *measurements: NodeMeasurement) -> MultiStepData:
        """Return a new store with the given measurements appended.

        Args:
            *measurements: The measurements to append.

        Returns:
            The updated immutable store.
        """
        return evolve(self, measurements=self.measurements + measurements)

    def with_process_state(
        self, contexts: Mapping[str, _Experiment], experiment_counter: int
    ) -> MultiStepData:
        """Return this data store with an atomically replaced process state."""
        return evolve(
            self,
            contexts=contexts,
            experiment_counter=experiment_counter,
        )

    def get_evaluation(self, dag: MultiStepDAG, ref: _EvaluationRef) -> NodeMeasurement:
        """Resolve a stable internal evaluation reference.

        Args:
            dag: The DAG topology.
            ref: The evaluation reference.

        Returns:
            The referenced node observation.

        Raises:
            IncompatibilityError: If the reference is out of range, refers to an
                unknown node, or does not match the stored node.
        """
        try:
            measurement = self.measurements[ref.index]
        except IndexError as ex:
            raise IncompatibilityError(
                f"Measurement reference {ref.index} is out of range."
            ) from ex
        if measurement.node != ref.node:
            raise IncompatibilityError(
                f"Measurement reference {ref.index} belongs to node "
                f"'{measurement.node}', not '{ref.node}'."
            )
        return measurement

    def evaluation_refs(  # noqa: DOC502
        self, dag: MultiStepDAG, node: str, *, successful_only: bool = True
    ) -> tuple[_EvaluationRef, ...]:
        """Collect references to actual evaluations of one node.

        Args:
            dag: The DAG topology.
            node: The node whose evaluations are requested.
            successful_only: Whether failed observations should be omitted.

        Returns:
            The references in insertion order.

        Raises:
            IncompatibilityError: If the node is not part of the DAG.
        """
        node_object = _get_node(dag, node)
        return tuple(
            _EvaluationRef(index=index, node=node)
            for index, measurement in enumerate(self.measurements)
            if measurement.node == node
            and (
                not successful_only
                or _get_feasibility_label(node_object, measurement) is True
            )
        )

    def measurements_of_experiment(
        self, experiment_id: str
    ) -> tuple[NodeMeasurement, ...]:
        """Return all measurements belonging to one experiment.

        Args:
            experiment_id: The experiment identifier.

        Returns:
            The measurements in insertion order.
        """
        return tuple(
            measurement
            for measurement in self.measurements
            if measurement.experiment_id == experiment_id
        )

    def experiment_bindings(self, experiment_id: str) -> dict[str, _EvaluationRef]:
        """Return the stored observations of one experiment as references.

        Args:
            experiment_id: The experiment identifier.

        Returns:
            One measurement reference per observed node.

        Raises:
            IncompatibilityError: If a node was observed more than once in the
                experiment.
        """
        bindings: dict[str, _EvaluationRef] = {}
        for index, measurement in enumerate(self.measurements):
            if measurement.experiment_id != experiment_id:
                continue
            if measurement.node in bindings:
                raise IncompatibilityError(
                    f"Experiment '{experiment_id}' contains more than one observation "
                    f"of node '{measurement.node}'."
                )
            bindings[measurement.node] = _EvaluationRef(
                index=index, node=measurement.node
            )
        return bindings

    def complete_measurement_frame(
        self,
        dag: MultiStepDAG,
        *,
        execution_mode: Literal["process", "functional"] = "process",
    ) -> pd.DataFrame:
        """Assemble complete, consistent real-reference rows as a temporary view.

        The returned frame is derived on demand. It is never stored and never
        becomes a second observation source. Process mode requires all rows to
        belong to one experiment context, including any real observations explicitly
        bound to it. Functional mode composes observations by their real references.

        Args:
            dag: The DAG topology.
            execution_mode: Whether parent relationships must be from one process
                experiment or may be composed from real references across experiments.

        Returns:
            One row per complete coherent set of successful real references,
            including an ``experiment_id`` column. Functional-mode identifiers
            encode the concrete measurement references used for each row. The
            frame is empty when no complete consistent reference exists.

        Raises:
            IncompatibilityError: If the execution mode is unknown or a reference
                cannot be resolved.
        """
        rows: list[pd.DataFrame] = []
        if execution_mode == "functional":
            references = {
                node.name: self.evaluation_refs(dag, node.name, successful_only=True)
                for node in dag.nodes
            }
            bindings: dict[str, _EvaluationRef] = {}

            def append_composable_rows(index: int) -> None:
                """Enumerate real successful measurements with matching parents."""
                if index == len(dag.topological_order):
                    row = self.make_complete_row(dag, bindings, _skip_invalid=True)
                    if row is None:
                        return
                    reference_id = "functional:" + ",".join(
                        f"{name}={bindings[name].index}"
                        for name in dag.topological_order
                    )
                    row[EXPERIMENT_ID] = reference_id
                    rows.append(row)
                    return

                node_name = dag.topological_order[index]
                node = dag.node_by_name[node_name]
                for ref in references[node_name]:
                    measurement = self.get_evaluation(dag, ref)
                    if any(
                        measurement.parent_refs.get(parent.name)
                        != bindings[parent.name]
                        for parent in node.parents
                    ):
                        continue
                    bindings[node_name] = ref
                    append_composable_rows(index + 1)
                    del bindings[node_name]

            append_composable_rows(0)
        elif execution_mode == "process":
            experiment_ids = dict.fromkeys(
                measurement.experiment_id
                for measurement in self.measurements
                if measurement.experiment_id is not None
            )
            experiment_ids.update(dict.fromkeys(self._contexts))
            seen: set[tuple[_EvaluationRef, ...]] = set()
            for experiment_id in experiment_ids:
                context = self._contexts.get(experiment_id)
                context_bindings = (
                    context.bindings
                    if context is not None
                    else self.experiment_bindings(experiment_id)
                )
                row = self.make_complete_row(dag, context_bindings, _skip_invalid=True)
                if row is None:
                    continue
                row_refs = tuple(
                    context_bindings[name] for name in dag.topological_order
                )
                if row_refs in seen:
                    continue
                seen.add(row_refs)
                row[EXPERIMENT_ID] = experiment_id
                rows.append(row)
        else:
            raise IncompatibilityError(f"Unknown execution mode: {execution_mode!r}.")
        if not rows:
            return pd.DataFrame(columns=dag._complete_measurement_columns)
        return pd.concat(rows, ignore_index=True)

    def matching_evaluation(
        self, dag: MultiStepDAG, node: str, inputs: Mapping[str, Any]
    ) -> tuple[_EvaluationRef, NodeMeasurement] | None:
        """Find a matching observation together with its concrete reference."""
        node_object = _get_node(dag, node)
        for index, measurement in enumerate(self.measurements):
            if measurement.node == node and _inputs_match(
                node_object, measurement.inputs, inputs
            ):
                return (
                    _EvaluationRef(index=index, node=node),
                    measurement,
                )
        return None

    def is_experiment_complete(self, dag: MultiStepDAG, experiment_id: str) -> bool:
        """Check whether every DAG node has a successful measurement.

        Args:
            dag: The DAG topology.
            experiment_id: The experiment identifier.

        Returns:
            ``True`` if every node has a successful measurement with the given
            identifier and none of that experiment's observations failed.
        """
        measured = self.measurements_of_experiment(experiment_id)
        return {measurement.node for measurement in measured} == {
            node.name for node in dag.nodes
        } and all(
            _get_feasibility_label(dag.node_by_name[measurement.node], measurement)
            is True
            for measurement in measured
        )

    def assemble_complete_row(
        self, dag: MultiStepDAG, experiment_id: str
    ) -> pd.DataFrame:
        """Assemble a complete DAG view from one experiment's measurements.

        This is a derived view only; it does not add or store observations.

        Args:
            dag: The DAG topology.
            experiment_id: The experiment identifier.

        Returns:
            A single-row dataframe with one column per DAG parameter and one per
            declared output from each node.

        Raises:
            IncompatibilityError: If the experiment is incomplete or failed.
        """
        if not self.is_experiment_complete(dag, experiment_id):
            raise IncompatibilityError(
                f"Experiment '{experiment_id}' is incomplete or failed."
            )
        bindings = {
            measurement.node: _EvaluationRef(index=index, node=measurement.node)
            for index, measurement in enumerate(self.measurements)
            if measurement.experiment_id == experiment_id
        }
        row = self.make_complete_row(dag, bindings)
        if row is None:
            raise IncompatibilityError(
                f"Experiment '{experiment_id}' does not describe one global DAG row."
            )
        return row

    def has_consistent_bindings(
        self,
        dag: MultiStepDAG,
        bindings: Mapping[str, _EvaluationRef],
    ) -> bool:
        """Check whether local observations form one complete global DAG row.

        Independent reused branches can disagree on shared ancestor outputs.
        Their local measurements remain valid, but have no global row projection.
        Invalid references still raise rather than being hidden as inconsistency.

        Args:
            dag: The DAG topology.
            bindings: The concrete observations bound to the context.

        Returns:
            Whether every node is bound and all parent outputs agree locally.
        """
        if set(bindings) != set(dag.topological_order):
            return False
        resolved = {
            name: self.get_evaluation(dag, ref) for name, ref in bindings.items()
        }
        return all(
            resolved[node.name].parent_refs.get(parent.name) == bindings[parent.name]
            and _output_values_match_inputs(
                parent, resolved[parent.name], resolved[node.name].inputs
            )
            for node in dag.nodes
            for parent in node.parents
        )

    def make_complete_row(
        self,
        dag: MultiStepDAG,
        bindings: Mapping[str, _EvaluationRef],
        *,
        _skip_invalid: bool = False,
    ) -> pd.DataFrame | None:
        """Assemble a temporary complete DAG row from concrete evaluation bindings.

        The row is a reference view. It is never stored as a second measurement.

        Args:
            dag: The DAG topology.
            bindings: Actual evaluations for all nodes, keyed by node name.
            _skip_invalid: Return ``None`` instead of raising when a context is
                incomplete, failed, or cannot describe one global DAG row.

        Returns:
            A single-row dataframe with one column per DAG parameter and one per
            declared output from each node, or ``None`` when ``_skip_invalid`` is
            enabled and the context cannot be projected.

        Raises:
            IncompatibilityError: If a node is unbound, an evaluation failed, or
                the bound parent outputs disagree with a child's local input.
        """
        expected_nodes = set(dag.topological_order)
        if _skip_invalid and set(bindings) != expected_nodes:
            return None
        missing = sorted(expected_nodes - set(bindings))
        if missing:
            raise IncompatibilityError(
                f"Cannot assemble a complete row because nodes are unbound: {missing}."
            )

        resolved = {
            name: self.get_evaluation(dag, ref) for name, ref in bindings.items()
        }
        row: dict[str, Any] = {}
        for node in dag.nodes:
            measurement = resolved[node.name]
            if not _is_successful(node, measurement):
                if _skip_invalid:
                    return None
                raise IncompatibilityError(
                    f"Cannot assemble a complete row because node '{node.name}' failed."
                )
            for parent in node.parents:
                if measurement.parent_refs.get(parent.name) != bindings[
                    parent.name
                ] or not _output_values_match_inputs(
                    parent, resolved[parent.name], measurement.inputs
                ):
                    if _skip_invalid:
                        return None
                    raise IncompatibilityError(
                        f"The local input of node '{node.name}' does not match the "
                        f"bound output of parent '{parent.name}'."
                    )
            for name in dag.owned_parameter_names(node.name):
                row[name] = measurement.inputs[name]
            row.update(_get_output_values(node, measurement))
            row.update(measurement.condition_values)
        return pd.DataFrame([row], columns=dag._complete_measurement_columns)

    def validate(
        self,
        dag: MultiStepDAG,
        numerical_measurements_must_be_within_tolerance: bool = True,
    ) -> None:
        """Validate the stored measurements against the DAG.

        Args:
            dag: The DAG defining the expected data contract.
            numerical_measurements_must_be_within_tolerance: If ``True``, numerical
                parameter values must lie within the parameter range or tolerance.

        Raises:
            IncompatibilityError: If a measurement does not match the node's
                model input, contains invalid values, or declares an ambiguous
                failure.
            IncompleteMeasurementsError: If configured condition values are missing.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        failed: set[tuple[str, str]] = set()
        identities: set[tuple[str, str]] = set()
        inputs_by_node: dict[str, list[dict[str, Any]]] = {}
        for index, measurement in enumerate(self.measurements):
            node = _get_node(dag, measurement.node)
            if measurement.experiment_id is not None:
                identity = (measurement.experiment_id, measurement.node)
                if identity in identities:
                    raise IncompatibilityError(
                        f"Experiment '{measurement.experiment_id}' contains more than "
                        f"one observation of node '{measurement.node}'."
                    )
                identities.add(identity)
            condition_columns = {
                condition.column
                for condition in node.feasibility_conditions
                if condition.column not in node.output_names
            }
            unexpected = set(measurement.condition_values) - condition_columns
            if unexpected:
                raise IncompatibilityError(
                    f"Node '{node.name}' received values for unconfigured feasibility "
                    f"columns: {sorted(unexpected)}."
                )
            is_successful = _get_feasibility_label(node, measurement)
            if is_successful is False and measurement.experiment_id is not None:
                failed.add((measurement.experiment_id, measurement.node))
            inputs_by_node.setdefault(node.name, []).append(dict(measurement.inputs))

            if measurement.parent_refs:
                if set(measurement.parent_refs) != set(dag.parents_of(node.name)):
                    raise IncompatibilityError(
                        f"The parent references of node '{node.name}' are incomplete."
                    )
                for parent_name, ref in measurement.parent_refs.items():
                    parent = dag.node_by_name[parent_name]
                    if ref.node != parent_name or ref.index >= index:
                        raise IncompatibilityError(
                            f"The parent reference for node '{node.name}' is invalid."
                        )
                    parent_measurement = self.get_evaluation(dag, ref)
                    if not _is_successful(parent, parent_measurement) or not (
                        _output_values_match_inputs(
                            parent, parent_measurement, measurement.inputs
                        )
                    ):
                        raise IncompatibilityError(
                            f"The parent input of node '{node.name}' does not match "
                            f"a successful observation of '{parent_name}'."
                        )
                expected = _history_inputs(
                    dag, self, node.name, measurement.parent_refs
                )
                if any(
                    measurement.inputs.get(name) != value
                    for name, value in expected.items()
                ):
                    raise IncompatibilityError(
                        f"The history of node '{node.name}' does not match its "
                        "referenced observations."
                    )

        for node_name, rows in inputs_by_node.items():
            _validate_node_inputs(
                dag.node_by_name[node_name],
                pd.DataFrame(rows),
                numerical_measurements_must_be_within_tolerance,
                context=dag.context,
            )

        for measurement in self.measurements:
            if any(
                (measurement.experiment_id, ancestor) in failed
                for ancestor in dag.ancestors_of(measurement.node)
            ):
                raise IncompatibilityError(
                    f"Node '{measurement.node}' has a failed ancestor in experiment "
                    f"'{measurement.experiment_id}'."
                )

        self._validate_process_state(dag)

    def _validate_process_state(self, dag: MultiStepDAG) -> None:
        """Reject inconsistent references and pending actions in stored contexts."""
        seen_pending: list[_PendingAction] = []
        for experiment_id, experiment in self._contexts.items():
            if not isinstance(experiment, _Experiment) or (
                experiment.experiment_id != experiment_id
            ):
                raise IncompatibilityError(
                    f"The stored context for experiment '{experiment_id}' is invalid."
                )
            if _identifier_number(experiment_id) > self._experiment_counter:
                raise IncompatibilityError(
                    f"Experiment '{experiment_id}' exceeds the stored identifier "
                    "counter."
                )
            for node_name, ref in experiment.bindings.items():
                if node_name not in dag.node_by_name or ref.node != node_name:
                    raise IncompatibilityError(
                        f"Experiment '{experiment_id}' has an invalid binding for "
                        f"node '{node_name}'."
                    )
                self.get_evaluation(dag, ref)
            for node_name, action in experiment.pending.items():
                node = dag.node_by_name.get(node_name)
                if (
                    node is None
                    or action.node != node_name
                    or action.experiment_id != experiment_id
                    or node_name in experiment.bindings
                    or set(action.parent_refs) != set(dag.parents_of(node_name))
                ):
                    raise IncompatibilityError(
                        f"Experiment '{experiment_id}' has an invalid pending action "
                        f"for node '{node_name}'."
                    )
                _validate_node_inputs(
                    node, pd.DataFrame([dict(action.inputs)]), context=dag.context
                )
                for parent_name, ref in action.parent_refs.items():
                    if experiment.bindings.get(parent_name) != ref:
                        raise IncompatibilityError(
                            f"The pending action for node '{node_name}' in experiment "
                            f"'{experiment_id}' has an inconsistent parent binding."
                        )
                    parent = dag.node_by_name[parent_name]
                    measurement = self.get_evaluation(dag, ref)
                    if not _is_successful(parent, measurement) or not (
                        _output_values_match_inputs(parent, measurement, action.inputs)
                    ):
                        raise IncompatibilityError(
                            f"The pending action for node '{node_name}' in experiment "
                            f"'{experiment_id}' has an invalid parent input."
                        )
                expected = _history_inputs(dag, self, node_name, action.parent_refs)
                if any(
                    action.inputs.get(name) != value for name, value in expected.items()
                ):
                    raise IncompatibilityError(
                        f"The pending action for node '{node_name}' has "
                        "invalid history."
                    )
                if contains_action(
                    seen_pending, node=node_name, inputs=action.inputs, dag=dag
                ):
                    raise IncompatibilityError(
                        f"The pending action for node '{node_name}' in experiment "
                        f"'{experiment_id}' duplicates another pending action."
                    )
                seen_pending.append(action)

    def node_training_data(self, dag: MultiStepDAG, node: str) -> pd.DataFrame:
        """Return the output-regression training view for one node.

        The stored observations are never modified. Failed observations enter the
        view according to the node's configured
        :class:`~baybe.recommenders.pure.experimental_feasibility.InfeasibleMeasurementMode`.

        Args:
            dag: The DAG topology.
            node: The node name.

        Returns:
            One row per accepted training observation with columns for the node's
            own parameters, configured history, and the observed or imputed output.

        Raises:
            IncompatibilityError: If ``DUMMY`` lacks a value or ``PREDICTED_MEAN``
                cannot be computed from the successful observations.
        """
        node_object = _get_node(dag, node)
        columns = [*dag.context.input_names(node_object), *node_object.output_names]
        records: list[tuple[dict[str, Any], bool | Any, dict[str, float | None]]] = []
        for measurement in self.measurements:
            if measurement.node != node:
                continue
            values = {
                **_get_output_values(node_object, measurement),
                **measurement.condition_values,
            }
            try:
                successful: bool | Any = _condition_success(node_object, values)
            except IncompleteMeasurementsError:
                successful = pd.NA
            records.append(
                (
                    dict(measurement.inputs),
                    successful,
                    _get_output_values(node_object, measurement),
                )
            )
        if not records:
            return pd.DataFrame(columns=columns)

        mode = node_object.infeasible_measurement_mode
        successes = [
            (inputs, outputs)
            for inputs, successful, outputs in records
            if successful is True
        ]
        failures = [
            (inputs, outputs)
            for inputs, successful, outputs in records
            if successful is False
        ]
        unknowns = [
            (inputs, outputs)
            for inputs, successful, outputs in records
            if pd.isna(successful)
        ]

        if mode is InfeasibleMeasurementMode.OBSERVED:
            rows = [
                {**inputs, **outputs}
                for inputs, outputs in (*successes, *unknowns, *failures)
            ]
        elif mode is InfeasibleMeasurementMode.EXCLUDE:
            rows = [
                {**inputs, **outputs} for inputs, outputs in (*successes, *unknowns)
            ]
        elif mode is InfeasibleMeasurementMode.DUMMY:
            dummy = node_object.infeasible_dummy_value
            if dummy is None:
                raise IncompatibilityError(
                    f"Node '{node_object.name}' uses "
                    f"'{InfeasibleMeasurementMode.DUMMY.name}' but has no dummy "
                    "value."
                )
            dummy_values = (
                dict(dummy)
                if isinstance(dummy, Mapping)
                else {node_object.output_names[0]: dummy}
            )
            rows = [
                {**inputs, **outputs} for inputs, outputs in (*successes, *unknowns)
            ] + [{**inputs, **dummy_values} for inputs, _ in failures]
        else:
            assert mode is InfeasibleMeasurementMode.PREDICTED_MEAN
            rows = [
                {**inputs, **outputs} for inputs, outputs in (*successes, *unknowns)
            ]
            if failures:
                rows.extend(
                    _predicted_mean_rows(
                        dag,
                        node_object,
                        successes,
                        [inputs for inputs, _ in failures],
                    )
                )

        return pd.DataFrame(rows, columns=columns)

    def node_feasibility_training_data(
        self, dag: MultiStepDAG, node: str
    ) -> pd.DataFrame:
        """Return raw local data, identities, and joint labels for a node.

        Args:
            dag: The DAG topology.
            node: The node name.

        Returns:
            A frame with unmodified local inputs, outputs and condition values, a
            nullable label derived from every condition, and stable measurement and
            experiment-group identities. The frame is empty for unconfigured nodes.
        """
        node_object = _get_node(dag, node)
        columns = [
            *dag.context.input_names(node_object),
            *node_object.output_names,
            *dict.fromkeys(
                condition.column
                for condition in node_object.feasibility_conditions
                if condition.column not in node_object.output_names
            ),
            FEASIBILITY_LABEL,
            MEASUREMENT_ID,
            EXPERIMENT_ID,
            EXPERIMENT_GROUP,
        ]
        if not node_object.feasibility_conditions:
            return pd.DataFrame(columns=columns)
        rows = []
        for measurement_id, measurement in enumerate(self.measurements):
            if measurement.node != node:
                continue
            values = {
                **_get_output_values(node_object, measurement),
                **measurement.condition_values,
            }
            try:
                label = _condition_success(node_object, values)
            except IncompleteMeasurementsError:
                label = pd.NA
            experiment_group = (
                ("experiment", measurement.experiment_id)
                if measurement.experiment_id is not None
                else ("measurement", measurement_id)
            )
            rows.append(
                {
                    **measurement.inputs,
                    **_get_output_values(node_object, measurement),
                    **measurement.condition_values,
                    FEASIBILITY_LABEL: label,
                    MEASUREMENT_ID: measurement_id,
                    EXPERIMENT_ID: measurement.experiment_id,
                    EXPERIMENT_GROUP: experiment_group,
                }
            )
        return pd.DataFrame(rows, columns=columns)


def experiment_actions(experiment: _Experiment) -> tuple[_PendingAction, ...]:
    """Return the pending actions of a context in stable order.

    Args:
        experiment: The context to inspect.

    Returns:
        The pending actions ordered by node name.
    """
    return tuple(experiment.pending[node] for node in sorted(experiment.pending))


def pending_actions(
    experiments: Mapping[str, _Experiment],
) -> tuple[_PendingAction, ...]:
    """Collect the pending actions of all contexts.

    Args:
        experiments: All known contexts in creation order.

    Returns:
        The pending actions in context and node order.
    """
    return tuple(
        action
        for experiment in experiments.values()
        for action in experiment_actions(experiment)
    )


def contains_action(
    actions: Iterable[_PendingAction],
    *,
    node: str,
    inputs: Mapping[str, Any],
    dag: MultiStepDAG | None = None,
) -> bool:
    """Return whether an identical local action is already pending.

    When the DAG is supplied, numerical own parameter values are compared within
    their configured tolerance and parent outputs are compared exactly. Without
    it, inputs are compared exactly.

    Args:
        actions: The pending actions to search.
        node: The node of the candidate action.
        inputs: The complete local input of the candidate action.
        dag: Optional DAG supplying the repeat comparison contract.

    Returns:
        ``True`` if an identical node-and-input action is pending.
    """
    if dag is None:
        return any(
            action.node == node and dict(action.inputs) == dict(inputs)
            for action in actions
        )

    node_object = dag.node_by_name.get(node)
    if node_object is None:
        return False
    return any(
        action.node == node and _inputs_match(node_object, action.inputs, inputs)
        for action in actions
    )


def failed_nodes(
    experiment: _Experiment, dag: MultiStepDAG, data: MultiStepData
) -> frozenset[str]:
    """Return the nodes with a failed evaluation in a context.

    Args:
        experiment: The snapshot to inspect.
        dag: The DAG topology.
        data: The store resolving evaluation references.

    Returns:
        The failed node names.
    """
    return frozenset(
        node
        for node, ref in experiment.bindings.items()
        if _get_feasibility_label(
            dag.node_by_name[node],
            data.get_evaluation(dag, ref),
        )
        is False
    )


def experiment_is_closed(
    experiment: _Experiment, dag: MultiStepDAG, data: MultiStepData
) -> bool:
    """Return whether a context is closed for further automatic continuation.

    A failed node or a discarded repeated execution closes its own context.
    Independent contexts remain unaffected.

    Args:
        experiment: The snapshot to inspect.
        dag: The DAG topology.
        data: The store resolving evaluation references.

    Returns:
        ``True`` if the context is explicitly closed or contains a failed evaluation.
    """
    return experiment.closed or bool(failed_nodes(experiment, dag, data))


def executable_nodes(
    experiment: _Experiment, dag: MultiStepDAG, data: MultiStepData
) -> tuple[tuple[str, dict[str, _EvaluationRef]], ...]:
    """Return all currently executable nodes of one context.

    A node is executable when it is neither measured nor pending in the context
    and all of its parents have successful real evaluations bound to the context.
    Independent fork branches are therefore returned together.

    Args:
        experiment: The context to continue.
        dag: The DAG topology.
        data: The store resolving evaluation references.

    Returns:
        The executable node names and their concrete parent bindings, in
        topological order.
    """
    if experiment_is_closed(experiment, dag, data):
        return ()

    result: list[tuple[str, dict[str, _EvaluationRef]]] = []
    for name in dag.topological_order:
        if name in experiment.bindings or name in experiment.pending:
            continue
        parents = dag.parents_of(name)
        if not all(
            parent in experiment.bindings
            and (
                _get_feasibility_label(
                    dag.node_by_name[parent],
                    data.get_evaluation(dag, experiment.bindings[parent]),
                )
                is True
            )
            for parent in parents
        ):
            continue
        result.append(
            (
                name,
                {parent: experiment.bindings[parent] for parent in parents},
            )
        )
    return tuple(result)


def make_executable_plans(
    dag: MultiStepDAG,
    data: MultiStepData,
    experiments: Mapping[str, _Experiment],
    *,
    mode: ExecutionMode,
    first_new_id: int = 1,
) -> tuple[_ExecutablePlan, ...]:
    """Collect every concretely executable action from the current state.

    Process mode continues each open experiment with all currently executable
    fork branches and additionally offers fresh root executions. Functional mode
    combines successful real parent outputs across experiments and prepares a
    fresh experiment for every resulting action.

    Args:
        dag: The DAG topology.
        data: The measurement store.
        experiments: All known experiment snapshots.
        mode: The configured execution mode.
        first_new_id: The first numeric suffix available for a new experiment.

    Returns:
        The executable plans in stable order, without actions that are already
        pending or duplicated inside the returned set.

    Raises:
        IncompatibilityError: If the execution mode is unknown.
    """
    if mode not in ("process", "functional"):
        raise IncompatibilityError(f"Unknown execution mode: {mode!r}.")

    pending = pending_actions(experiments)
    plans: list[_ExecutablePlan] = []
    next_identifier = first_new_id

    def add(
        node: str,
        inputs: Mapping[str, Any],
        *,
        experiment_id: str | None,
        parent_refs: Mapping[str, _EvaluationRef] | None = None,
        is_new: bool,
    ) -> None:
        """Add a non-duplicate plan and allocate a fresh identifier if needed."""
        nonlocal next_identifier
        references = {} if parent_refs is None else parent_refs
        duplicate = any(
            existing.node == node
            and dict(existing.inputs) == dict(inputs)
            and (
                (
                    not is_new
                    and not existing.is_new
                    and existing.experiment_id == experiment_id
                )
                or (
                    is_new
                    and existing.is_new
                    and dict(existing.parent_refs) == dict(references)
                )
            )
            for existing in plans
        )
        if duplicate or contains_action(pending, node=node, inputs=inputs, dag=dag):
            return
        plans.append(
            _ExecutablePlan(
                experiment_id=(
                    f"exp-{next_identifier}" if experiment_id is None else experiment_id
                ),
                node=node,
                parent_refs=references,
                inputs=inputs,
                is_new=is_new,
            )
        )
        if is_new:
            next_identifier += 1

    if mode == "process":
        for experiment in experiments.values():
            for node, parent_refs in executable_nodes(experiment, dag, data):
                inputs = _history_inputs(dag, data, node, parent_refs)
                add(
                    node,
                    inputs,
                    experiment_id=experiment.experiment_id,
                    parent_refs=parent_refs,
                    is_new=False,
                )
        for node in dag.topological_order:
            if dag.parents_of(node):
                continue
            add(node, {}, experiment_id=None, is_new=True)
        return tuple(
            evolve(
                plan,
                remaining_cost=remaining_cost(dag, experiments, plan, mode=mode),
            )
            for plan in plans
        )

    for node_name in dag.topological_order:
        node_object = dag.node_by_name[node_name]
        parent_choices: list[list[tuple[Any, _EvaluationRef]]] = []
        for parent in node_object.parents:
            choices: list[tuple[Any, _EvaluationRef]] = []
            for ref in data.evaluation_refs(dag, parent.name, successful_only=True):
                value = data.get_evaluation(dag, ref).output
                choices.append((value, ref))
            if not choices:
                break
            parent_choices.append(choices)
        if len(parent_choices) != len(node_object.parents):
            continue

        for combination in product(*parent_choices):
            parent_refs = {
                parent.name: ref
                for parent, (_, ref) in zip(node_object.parents, combination)
            }
            inputs = _history_inputs(dag, data, node_name, parent_refs)
            add(
                node_name,
                inputs,
                experiment_id=None,
                parent_refs=parent_refs,
                is_new=True,
            )
    return tuple(
        evolve(
            plan,
            remaining_cost=remaining_cost(dag, experiments, plan, mode=mode),
        )
        for plan in plans
    )


def _history_inputs(
    dag: MultiStepDAG,
    data: MultiStepData,
    node_name: str,
    parent_refs: Mapping[str, _EvaluationRef],
) -> dict[str, Any]:
    """Resolve each configured history feature through its real lineage."""
    result: dict[str, Any] = {}
    for feature in dag.context._features(dag.node_by_name[node_name]):
        if not feature.path:
            raise IncompatibilityError(
                "A history feature must have a predecessor path."
            )
        references = parent_refs
        measurement: NodeMeasurement | None = None
        for ancestor in feature.path:
            try:
                ref = references[ancestor]
            except KeyError as ex:
                raise IncompatibilityError(
                    f"Cannot resolve history for node '{node_name}': the real "
                    f"reference to predecessor '{ancestor}' is missing."
                ) from ex
            measurement = data.get_evaluation(dag, ref)
            references = measurement.parent_refs
        assert measurement is not None
        result[feature.name] = (
            _get_output_values(dag.node_by_name[feature.path[-1]], measurement)[
                feature.source
            ]
            if feature.parameter is None
            else measurement.inputs[feature.source]
        )
    return result


def remaining_cost(
    dag: MultiStepDAG,
    experiments: Mapping[str, _Experiment],
    plan: _ExecutablePlan,
    *,
    mode: ExecutionMode,
) -> float:
    """Compute the node costs still required to complete the objective.

    Already observed nodes and already pending actions of the plan's experiment
    are not charged again. A fresh functional plan only excludes its known
    ancestors, because every other graph output still needs an execution.

    Args:
        dag: The DAG topology.
        experiments: All known experiment snapshots.
        plan: The action whose remaining costs are requested.
        mode: The configured execution mode.

    Returns:
        The sum of the remaining node costs.
    """
    if plan.is_new:
        if mode == "functional":
            omitted = set(dag.ancestors_of(plan.node))
        else:
            omitted = set()
    else:
        experiment = experiments[plan.experiment_id]
        omitted = set(experiment.bindings) | set(experiment.pending)
    return sum(
        dag.node_by_name[name].cost
        for name in dag.topological_order
        if name not in omitted
    )


def register_action(
    experiments: Mapping[str, _Experiment],
    counter: int,
    *,
    experiment_id: str,
    node: str,
    inputs: Mapping[str, Any],
    parent_refs: Mapping[str, _EvaluationRef],
    data: MultiStepData,
    dag: MultiStepDAG,
) -> tuple[dict[str, _Experiment], int]:
    """Reserve one actual action in its explicitly assigned context.

    Args:
        experiments: All contexts in creation order.
        counter: The last allocated numeric experiment counter.
        experiment_id: The concrete context identifier. Must name an existing open
            context or a newly reserved unused identifier.
        node: The selected node.
        inputs: The complete local input of the action.
        parent_refs: The concrete parent evaluations of the action.
        data: The store resolving evaluation references.
        dag: The DAG topology.

    Returns:
        The updated contexts and the updated counter.

    Raises:
        IncompatibilityError: If the context is missing, closed, already measured
            the node, duplicates a pending action, names an unknown node, omits or
            contradicts a parent evaluation, or carries an invalid local input.
    """
    if experiment_id is None:
        raise IncompatibilityError(
            "Every evaluated action must carry its concrete experiment identifier."
        )
    node_object = dag.node_by_name.get(node)
    if node_object is None:
        raise IncompatibilityError(
            f"Node '{node}' is not part of the DAG. Known nodes: "
            f"{sorted(dag.node_by_name)}."
        )
    expected_parents = set(dag.parents_of(node))
    if set(parent_refs) != expected_parents:
        raise IncompatibilityError(
            f"The action for node '{node}' must reference exactly its parent "
            f"evaluations {sorted(expected_parents)}, but references "
            f"{sorted(parent_refs)}."
        )

    context = experiments.get(experiment_id)
    if context is not None:
        if experiment_is_closed(context, dag, data):
            raise IncompatibilityError(
                f"Experiment '{experiment_id}' is closed and cannot start another "
                f"action."
            )
        if node in context.bindings:
            raise IncompatibilityError(
                f"Node '{node}' has already been measured in experiment "
                f"'{experiment_id}'."
            )
        if node in context.pending:
            raise IncompatibilityError(
                f"Node '{node}' already has a pending action in experiment "
                f"'{experiment_id}'."
            )

    for parent, ref in parent_refs.items():
        if ref.node != parent:
            raise IncompatibilityError(
                f"The parent reference for '{parent}' points to node '{ref.node}'."
            )
        evaluation = data.get_evaluation(dag, ref)
        if not _is_successful(dag.node_by_name[parent], evaluation):
            raise IncompatibilityError(
                f"The parent evaluation referenced for '{parent}' is not successful."
            )
        parent_node = dag.node_by_name[parent]
        if not _output_values_match_inputs(parent_node, evaluation, inputs):
            expected = _get_output_values(parent_node, evaluation)
            actual = {name: inputs.get(name) for name in parent_node.output_names}
            raise IncompatibilityError(
                f"The local input of node '{node}' uses parent outputs {actual!r} "
                f"for '{parent}', but the referenced evaluation observed "
                f"{expected!r}."
            )

    expected_history = _history_inputs(dag, data, node, parent_refs)
    if any(inputs.get(name) != value for name, value in expected_history.items()):
        raise IncompatibilityError(f"The action for node '{node}' has invalid history.")
    if set(inputs) != set(dag.context.input_names(node_object)):
        raise IncompatibilityError(
            f"The action for node '{node}' must provide exactly the local input "
            f"columns {list(dag.context.input_names(node_object))}, but provides "
            f"{sorted(inputs)}."
        )
    import pandas as pd

    _validate_node_inputs(node_object, pd.DataFrame([inputs]), context=dag.context)

    if contains_action(pending_actions(experiments), node=node, inputs=inputs, dag=dag):
        raise IncompatibilityError(
            f"An identical action for node '{node}' is already pending. The same "
            "node and local input cannot be reserved twice, even across experiments."
        )

    action = _PendingAction(
        node=node,
        inputs=inputs,
        parent_refs=parent_refs,
        experiment_id=experiment_id,
    )
    updated = dict(experiments)
    if context is None:
        known = _known_experiment_ids(data, experiments)
        if experiment_id in known:
            raise IncompatibilityError(
                f"Experiment identifier '{experiment_id}' is already in use."
            )
        context = _Experiment(experiment_id=experiment_id)
        counter = max(counter, _identifier_number(experiment_id))
    else:
        counter = max(counter, _identifier_number(experiment_id))
    updated[experiment_id] = context.with_pending(action)
    return updated, counter


def _identifier_number(experiment_id: str) -> int:
    """Extract the numeric suffix of an automatically generated identifier."""
    prefix, separator, suffix = experiment_id.partition("-")
    if separator and prefix == "exp" and suffix.isdigit():
        return int(suffix)
    return 0


def _known_experiment_ids(
    data: MultiStepData, experiments: Mapping[str, _Experiment]
) -> set[str]:
    """Collect all identifiers that a new context must not reuse."""
    known = set(experiments)
    known.update(
        measurement.experiment_id
        for measurement in data.measurements
        if measurement.experiment_id is not None
    )
    return known


def _inputs_match(
    node: MultiStepNode,
    stored: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Compare two complete local inputs using the repeat contract.

    Numeric own parameters use their configured tolerance. History features and
    non-numerical inputs must match exactly.

    Args:
        node: The node owning the inputs.
        stored: The already stored local input.
        candidate: The candidate local input.

    Returns:
        Whether both inputs describe the same local execution.
    """
    if set(stored) != set(candidate):
        return False
    return _own_inputs_match(node, stored, candidate) and _parent_inputs_match(
        node, stored, candidate
    )


def _own_inputs_match(
    node: MultiStepNode,
    stored: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Compare the node's own parameter values using the repeat contract.

    Args:
        node: The node owning the inputs.
        stored: The already stored local input.
        candidate: The candidate local input.

    Returns:
        Whether all own parameter values describe the same execution.
    """
    for parameter in node.inputs:
        name = parameter.name
        if getattr(parameter, "is_numerical", False):
            try:
                if not np.isclose(
                    float(stored[name]),
                    float(candidate[name]),
                    rtol=0.0,
                    atol=float(getattr(parameter, "tolerance", 0.0)) + 1e-12,
                ):
                    return False
            except (TypeError, ValueError):
                return False
        elif stored[name] != candidate[name]:
            return False
    return True


def _parent_inputs_match(
    node: MultiStepNode,
    stored: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> bool:
    """Compare incoming historical features exactly.

    Args:
        node: The node owning the inputs.
        stored: The already stored local input.
        candidate: The candidate local input.

    Returns:
        Whether all history features match exactly.
    """
    own = {parameter.name for parameter in node.inputs}
    for name in set(stored) - own:
        try:
            if not bool(stored[name] == candidate[name]):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _make_local_searchspace(
    context: MultiStepContext, node: MultiStepNode, frame: pd.DataFrame
) -> SearchSpace:
    """Create the synthetic local fit search space of a node.

    Args:
        context: The configured history policy.
        node: The node whose model is fitted.
        frame: Local training data containing the configured history.

    Returns:
        The local search space covering own parameters and historical features.
    """
    from baybe.parameters.numerical import NumericalContinuousParameter
    from baybe.searchspace.core import SearchSpace

    parent_parameters: list[NumericalContinuousParameter] = []
    for name in (
        feature.name for feature in context._features(node) if feature.parameter is None
    ):
        if frame.empty:
            # This interval defines only the local input transform. Parent outputs
            # remain on their raw posterior scale and are not clipped to these bounds.
            lower, upper = 0.0, 1.0
        else:
            values = frame[name].to_numpy(dtype=float)
            lower = float(np.min(values))
            upper = float(np.max(values))
        if lower == upper:
            delta = max(abs(lower) * 0.5, 1.0)
            lower -= delta
            upper += delta
        parent_parameters.append(
            NumericalContinuousParameter(name, bounds=(lower, upper))
        )
    return SearchSpace.from_product([*context._parameters(node), *parent_parameters])


def _predicted_mean_rows(
    dag: MultiStepDAG,
    node: MultiStepNode,
    successes: list[tuple[dict[str, Any], dict[str, float | None]]],
    failure_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Impute failure outputs with means predicted from successful observations.

    Args:
        dag: The DAG defining the configured history and target transformations.
        node: The node whose training view is assembled.
        successes: Successful local inputs and real output bundles.
        failure_inputs: Local inputs of failed observations.

    Returns:
        One row per failed observation with a predicted value for every output.

    Raises:
        IncompatibilityError: If there are too few successful observations or the
            temporary successful-only model cannot be fitted or evaluated.
    """
    import torch

    from baybe.multistep.model import _make_local_surrogate
    from baybe.targets.numerical import NumericalTarget

    context = dag.context
    predictions_by_output: dict[str, list[float]] = {}
    failure_frame = pd.DataFrame(
        failure_inputs, columns=list(context.input_names(node))
    )
    for output_name in node.output_names:
        output_successes = [
            (inputs, values[output_name])
            for inputs, values in successes
            if values[output_name] is not None
        ]
        if not output_successes:
            raise IncompatibilityError(
                f"Node '{node.name}' cannot compute "
                f"'{InfeasibleMeasurementMode.PREDICTED_MEAN.name}' training values "
                f"for output '{output_name}' without a successful observed value."
            )
        success_frame = pd.DataFrame(
            [{**inputs, output_name: output} for inputs, output in output_successes],
            columns=[*context.input_names(node), output_name],
        )
        try:
            searchspace = _make_local_searchspace(context, node, success_frame)
            target = dag.target_by_output.get(output_name, NumericalTarget(output_name))
            surrogate = _make_local_surrogate(
                node, searchspace, target, success_frame, output_name
            )
            with torch.no_grad():
                predictions = (
                    cast(Any, surrogate).posterior(failure_frame).mean.squeeze(-1).cpu()
                )
            predictions_by_output[output_name] = [
                float(value) for value in predictions.reshape(-1).tolist()
            ]
        except IncompatibilityError:
            raise
        except Exception as ex:
            raise IncompatibilityError(
                f"Node '{node.name}' cannot compute "
                f"'{InfeasibleMeasurementMode.PREDICTED_MEAN.name}' training values "
                f"for output '{output_name}': {ex}"
            ) from ex

    return [
        {
            **inputs,
            **{
                output_name: predictions_by_output[output_name][index]
                for output_name in node.output_names
            },
        }
        for index, inputs in enumerate(failure_inputs)
    ]


def _is_successful(node: MultiStepNode, measurement: NodeMeasurement) -> bool:
    """Derive success from output and condition values without duplicating state."""
    label = _get_feasibility_label(node, measurement)
    if label is None:
        raise IncompleteMeasurementsError(
            f"Feasibility values for node '{node.name}' are incomplete."
        )
    return label


def _get_feasibility_label(
    node: MultiStepNode, measurement: NodeMeasurement
) -> bool | None:
    """Derive the nullable joint label from one raw node measurement."""
    output_values = _get_output_values(node, measurement)
    try:
        successful = _condition_success(
            node, {**output_values, **measurement.condition_values}
        )
    except IncompleteMeasurementsError:
        return None
    if successful is not False and any(
        value is None for value in output_values.values()
    ):
        raise IncompatibilityError(
            f"An observation of node '{node.name}' without all outputs requires "
            "a known failed feasibility condition."
        )
    return successful


def _validate_node_inputs(
    node: MultiStepNode,
    frame: pd.DataFrame,
    numerical_measurements_must_be_within_tolerance: bool = True,
    *,
    context: MultiStepContext | None = None,
) -> None:
    """Validate a local input frame using the existing observation contract.

    Args:
        node: The node whose model input is validated.
        frame: The local input frame.
        numerical_measurements_must_be_within_tolerance: If ``True``, numerical
            parameter values must lie within the parameter range or tolerance.
        context: The history policy, defaulting to direct parent outputs.

    Raises:
        IncompatibilityError: If the columns or values violate the observation
            contract.
    """
    if not frame.columns.is_unique:
        raise IncompatibilityError(
            f"Input columns for node '{node.name}' must be unique."
        )
    context = MultiStepContext() if context is None else context
    _validate_input_columns(context, node, frame.columns)
    if frame.empty:
        return

    _validate_parameter_frame(
        context._parameters(node),
        frame,
        f"node '{node.name}'",
        numerical_measurements_must_be_within_tolerance,
    )

    for name in (
        feature.name for feature in context._features(node) if feature.parameter is None
    ):
        for value in frame[name]:
            _validate_finite_input(name, value, node.name)


def _validate_parameter_frame(
    parameters: Iterable[Parameter],
    frame: pd.DataFrame,
    context: str,
    numerical_measurements_must_be_within_tolerance: bool,
) -> None:
    """Validate parameter columns through BayBE's shared validator."""
    parameters = tuple(parameters)
    if frame.empty:
        return

    try:
        validate_parameter_input(
            frame,
            parameters,
            numerical_measurements_must_be_within_tolerance,
        )
    except (TypeError, ValueError) as ex:
        raise IncompatibilityError(
            f"Invalid parameter inputs for {context}: {ex}"
        ) from ex
    if not numerical_measurements_must_be_within_tolerance:
        for parameter in parameters:
            if not parameter.is_numerical:
                continue
            for value in frame[parameter.name]:
                if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
                    raise IncompatibilityError(
                        f"The numerical parameter '{parameter.name}' used by "
                        f"{context} must be finite and non-Boolean, got {value!r}."
                    )


def _get_node(dag: MultiStepDAG, node: str) -> MultiStepNode:
    """Return a node or raise a descriptive error."""
    node_object = dag.node_by_name.get(node)
    if node_object is None:
        raise IncompatibilityError(
            f"Node '{node}' is not part of the DAG. Known nodes: "
            f"{sorted(dag.node_by_name)}."
        )
    return node_object


def _validate_input_columns(
    context: MultiStepContext, node: MultiStepNode, columns: Iterable[str]
) -> None:
    """Validate that the input columns match the node's model input."""
    expected = set(context.input_names(node))
    actual = set(columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise IncompatibilityError(
            f"The inputs of node '{node.name}' do not match its model input. "
            f"Missing columns: {missing}, unexpected columns: {unexpected}. "
            f"Expected columns: {list(context.input_names(node))}."
        )


def _validate_finite_input(name: str, value: Any, node: str) -> None:
    """Validate a parent-output input value."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise IncompatibilityError(
            f"The parent output '{name}' used by node '{node}' must be numeric, "
            f"got {value!r}."
        )
    if not np.isfinite(value):
        raise IncompatibilityError(
            f"The parent output '{name}' used by node '{node}' must be finite, "
            f"got {value!r}."
        )


__all__ = [
    "DuplicateMeasurementWarning",
    "MultiStepData",
    "NodeMeasurement",
    "RepeatedMeasurementWarning",
]


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
