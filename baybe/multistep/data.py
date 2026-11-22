"""Measurement store for multi-step campaigns.

A :class:`NodeMeasurement` is one real observation of one node. Its inputs are the
full input of the node model: the values of the node's own parameters together
with the observed outputs of its parents. Node training views are derived from
the stored measurements on demand. Complete DAG rows are never stored as a second
data source; they are temporary views assembled from concrete observations.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from copy import deepcopy
from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

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

from ._columns import FEASIBILITY_LABEL
from .dag import MultiStepDAG
from .node import MultiStepNode

if TYPE_CHECKING:
    from baybe.searchspace.core import SearchSpace

ExecutionMode = Literal["process", "functional"]
"""How executable actions are derived from recorded experiments."""


class DuplicateMeasurementWarning(UserWarning):
    """An identical observation was reported again for the same identity."""


class RepeatedMeasurementWarning(UserWarning):
    """A repeated execution was discarded because repeats are disabled."""


def _condition_success(node: MultiStepNode, values: Mapping[str, Any]) -> bool:
    """Evaluate all node conditions, allowing missing data only after a failure.

    Args:
        node: The node whose execution conditions are evaluated.
        values: Observed output and auxiliary values keyed by measurement column.

    Returns:
        Whether all configured conditions pass.

    Raises:
        IncompleteMeasurementsError: If a condition value is missing and no other
            observed condition establishes failure.
        InvalidTargetValueError: If a measured value has an incompatible type or
            is non-finite.
    """
    passed = True
    missing: list[str] = []
    for condition in node.feasibility_conditions:
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
    node: MultiStepNode, measurements: pd.DataFrame
) -> pd.Series:
    """Derive joint Boolean success labels from a node's configured conditions.

    Args:
        node: The node whose conditions are evaluated.
        measurements: Rows containing the node output and auxiliary measurements.

    Returns:
        A Boolean label for each row, preserving index and using the internal label
        column name.
    """
    labels = [
        _condition_success(node, cast(Mapping[str, Any], row))
        for row in measurements.to_dict(orient="records")
    ]
    return pd.Series(labels, index=measurements.index, name=FEASIBILITY_LABEL)


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only copy of a mapping."""
    return MappingProxyType(dict(value))


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
    """One observation of one node."""

    node: str = field(validator=and_(instance_of(str), min_len(1)))
    """The observed node."""

    inputs: Mapping[str, Any] = field(
        converter=lambda value: MappingProxyType(dict(value)),
        validator=instance_of(Mapping),
    )
    """The node's own parameter values and observed parent outputs."""

    output: float | None = field(default=None)
    """The real observed output, absent only for explicitly failed observations."""

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
        self, _: Any, value: float | None
    ) -> None:
        """Validate that the output is a finite number."""
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise IncompatibilityError(
                f"The output of node '{self.node}' must be numeric, got {value!r}."
            )
        if not np.isfinite(value):
            raise IncompatibilityError(
                f"The output of node '{self.node}' must be finite, got {value!r}."
            )

    def __attrs_post_init__(self) -> None:
        """Store the output as a plain float."""
        if self.output is not None:
            object.__setattr__(self, "output", float(self.output))


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
            and (not successful_only or _is_successful(node_object, measurement))
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
        from ._columns import EXPERIMENT_ID

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
            _is_successful(dag.node_by_name[measurement.node], measurement)
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
            node output.

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
            and resolved[node.name].inputs[parent.output]
            == resolved[parent.name].output
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
            node output, or ``None`` when ``_skip_invalid`` is enabled and the
            context cannot be projected.

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
                if (
                    measurement.parent_refs.get(parent.name) != bindings[parent.name]
                    or measurement.inputs[parent.output] != resolved[parent.name].output
                ):
                    if _skip_invalid:
                        return None
                    raise IncompatibilityError(
                        f"The local input of node '{node.name}' does not match the "
                        f"bound output of parent '{parent.name}'."
                    )
            for name in dag.owned_parameter_names(node.name):
                row[name] = measurement.inputs[name]
            row[node.output] = measurement.output
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
                if condition.column != node.output
            }
            unexpected = set(measurement.condition_values) - condition_columns
            if unexpected:
                raise IncompatibilityError(
                    f"Node '{node.name}' received values for unconfigured feasibility "
                    f"columns: {sorted(unexpected)}."
                )
            is_successful = _is_successful(node, measurement)
            if not is_successful and measurement.experiment_id is not None:
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
                    if not _is_successful(parent, parent_measurement) or (
                        measurement.inputs.get(parent.output)
                        != parent_measurement.output
                    ):
                        raise IncompatibilityError(
                            f"The parent input of node '{node.name}' does not match "
                            f"a successful observation of '{parent_name}'."
                        )

        for node_name, rows in inputs_by_node.items():
            _validate_node_inputs(
                dag.node_by_name[node_name],
                pd.DataFrame(rows),
                numerical_measurements_must_be_within_tolerance,
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
                _validate_node_inputs(node, pd.DataFrame([dict(action.inputs)]))
                for parent_name, ref in action.parent_refs.items():
                    if experiment.bindings.get(parent_name) != ref:
                        raise IncompatibilityError(
                            f"The pending action for node '{node_name}' in experiment "
                            f"'{experiment_id}' has an inconsistent parent binding."
                        )
                    parent = dag.node_by_name[parent_name]
                    measurement = self.get_evaluation(dag, ref)
                    if not _is_successful(parent, measurement) or (
                        action.inputs[parent.output] != measurement.output
                    ):
                        raise IncompatibilityError(
                            f"The pending action for node '{node_name}' in experiment "
                            f"'{experiment_id}' has an invalid parent input."
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
            own parameters, its parent outputs, and the observed or imputed output.

        Raises:
            IncompatibilityError: If ``DUMMY`` lacks a value or ``PREDICTED_MEAN``
                cannot be computed from the successful observations.
        """
        node_object = _get_node(dag, node)
        columns = [*node_object.model_input_names, node_object.output]
        records: list[tuple[dict[str, Any], bool, float | None]] = [
            (
                dict(measurement.inputs),
                _is_successful(node_object, measurement),
                measurement.output,
            )
            for measurement in self.measurements
            if measurement.node == node
        ]
        if not records:
            return pd.DataFrame(columns=columns)

        mode = node_object.infeasible_measurement_mode
        successes = [
            (inputs, output)
            for inputs, successful, output in records
            if successful and output is not None
        ]
        failures = [
            (inputs, output) for inputs, successful, output in records if not successful
        ]

        if mode is InfeasibleMeasurementMode.OBSERVED:
            rows = [
                {**inputs, node_object.output: output} for inputs, output in successes
            ] + [
                {**inputs, node_object.output: output}
                for inputs, output in failures
                if output is not None
            ]
        elif mode is InfeasibleMeasurementMode.EXCLUDE:
            rows = [
                {**inputs, node_object.output: output} for inputs, output in successes
            ]
        elif mode is InfeasibleMeasurementMode.DUMMY:
            dummy = node_object.infeasible_dummy_value
            if dummy is None:
                raise IncompatibilityError(
                    f"Node '{node_object.name}' uses "
                    f"'{InfeasibleMeasurementMode.DUMMY.name}' but has no dummy "
                    "value."
                )
            rows = [
                {**inputs, node_object.output: output} for inputs, output in successes
            ] + [{**inputs, node_object.output: dummy} for inputs, _ in failures]
        else:
            assert mode is InfeasibleMeasurementMode.PREDICTED_MEAN
            rows = [
                {**inputs, node_object.output: output} for inputs, output in successes
            ]
            if failures:
                rows.extend(
                    _predicted_mean_rows(
                        node_object,
                        successes,
                        [inputs for inputs, _ in failures],
                    )
                )

        return pd.DataFrame(rows, columns=columns)

    def node_feasibility_training_data(
        self, dag: MultiStepDAG, node: str
    ) -> pd.DataFrame:
        """Return all raw feasibility observations with their full local inputs.

        Args:
            dag: The DAG topology.
            node: The node name.

        Returns:
            A raw-label training frame, empty for unconfigured nodes.
        """
        node_object = _get_node(dag, node)
        columns = [*node_object.model_input_names, FEASIBILITY_LABEL]
        if not node_object.feasibility_conditions:
            return pd.DataFrame(columns=columns)
        rows = [
            {
                **measurement.inputs,
                FEASIBILITY_LABEL: _is_successful(node_object, measurement),
            }
            for measurement in self.measurements
            if measurement.node == node
        ]
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
        if not _is_successful(
            dag.node_by_name[node],
            data.get_evaluation(dag, ref),
        )
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
            and _is_successful(
                dag.node_by_name[parent],
                data.get_evaluation(dag, experiment.bindings[parent]),
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
                inputs = _parent_outputs(dag, data, parent_refs)
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
            inputs = {
                parent.output: value
                for parent, (value, _) in zip(node_object.parents, combination)
            }
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


def _parent_outputs(
    dag: MultiStepDAG,
    data: MultiStepData,
    parent_refs: Mapping[str, _EvaluationRef],
) -> dict[str, Any]:
    """Resolve the real parent outputs supplying a local action input."""
    return {
        dag.node_by_name[parent].output: data.get_evaluation(dag, ref).output
        for parent, ref in parent_refs.items()
    }


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
        expected = evaluation.output
        actual = inputs.get(dag.node_by_name[parent].output)
        if actual != expected:
            raise IncompatibilityError(
                f"The local input of node '{node}' uses parent output {actual!r} "
                f"for '{parent}', but the referenced evaluation observed "
                f"{expected!r}."
            )

    if set(inputs) != set(node_object.model_input_names):
        raise IncompatibilityError(
            f"The action for node '{node}' must provide exactly the local input "
            f"columns {list(node_object.model_input_names)}, but provides "
            f"{sorted(inputs)}."
        )
    import pandas as pd

    _validate_node_inputs(node_object, pd.DataFrame([inputs]))

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

    Numeric own parameters use their configured tolerance. Parent outputs and
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
    """Compare incoming parent outputs exactly.

    Args:
        node: The node owning the inputs.
        stored: The already stored local input.
        candidate: The candidate local input.

    Returns:
        Whether all parent outputs match exactly.
    """
    for name in node.parent_output_names:
        try:
            if not bool(stored[name] == candidate[name]):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _make_local_searchspace(node: MultiStepNode, frame: pd.DataFrame) -> SearchSpace:
    """Build the synthetic local fit search space of a node.

    Args:
        node: The node whose model is fitted.
        frame: Local training data containing the parent outputs.

    Returns:
        The local search space covering own parameters and observed parent outputs.
    """
    from baybe.parameters.numerical import NumericalContinuousParameter
    from baybe.searchspace.core import SearchSpace

    parent_parameters: list[NumericalContinuousParameter] = []
    for name in node.parent_output_names:
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
    return SearchSpace.from_product([*node.inputs, *parent_parameters])


def _predicted_mean_rows(
    node: MultiStepNode,
    successes: list[tuple[dict[str, Any], float]],
    failure_inputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Impute failure outputs with means predicted from successful observations.

    Args:
        node: The node whose training view is assembled.
        successes: Successful local inputs and real outputs.
        failure_inputs: Local inputs of failed observations.

    Returns:
        One row per failed observation with the predicted output.

    Raises:
        IncompatibilityError: If there are too few successful observations or the
            temporary successful-only model cannot be fitted or evaluated.
    """
    if len(successes) < 2:
        raise IncompatibilityError(
            f"Node '{node.name}' uses "
            f"'{InfeasibleMeasurementMode.PREDICTED_MEAN.name}' but provides fewer "
            f"than two successful observations ({len(successes)})."
        )
    import torch

    from baybe.targets.numerical import NumericalTarget

    success_frame = pd.DataFrame(
        [{**inputs, node.output: output} for inputs, output in successes],
        columns=[*node.model_input_names, node.output],
    )
    try:
        searchspace = _make_local_searchspace(node, success_frame)
        surrogate = deepcopy(node.surrogate_model)
        target = NumericalTarget(node.output)
        surrogate.fit(searchspace, target.to_objective(), success_frame)
        failure_frame = pd.DataFrame(
            failure_inputs, columns=list(node.model_input_names)
        )
        with torch.no_grad():
            predictions = (
                cast(Any, surrogate).posterior(failure_frame).mean.squeeze(-1).cpu()
            )
        values = predictions.reshape(-1).tolist()
    except IncompatibilityError:
        raise
    except Exception as ex:
        raise IncompatibilityError(
            f"Node '{node.name}' cannot compute "
            f"'{InfeasibleMeasurementMode.PREDICTED_MEAN.name}' training values: {ex}"
        ) from ex

    return [
        {**inputs, node.output: float(value)}
        for inputs, value in zip(failure_inputs, values)
    ]


def _is_successful(node: MultiStepNode, measurement: NodeMeasurement) -> bool:
    """Derive success from output and condition values without duplicating state."""
    successful = _condition_success(
        node, {node.output: measurement.output, **measurement.condition_values}
    )
    if successful and measurement.output is None:
        raise IncompatibilityError(
            f"An observation of node '{node.name}' without an output requires "
            "a failed feasibility condition."
        )
    return successful


def _validate_node_inputs(
    node: MultiStepNode,
    frame: pd.DataFrame,
    numerical_measurements_must_be_within_tolerance: bool = True,
) -> None:
    """Validate a local input frame using the existing observation contract.

    Args:
        node: The node whose model input is validated.
        frame: The local input frame.
        numerical_measurements_must_be_within_tolerance: If ``True``, numerical
            parameter values must lie within the parameter range or tolerance.

    Raises:
        IncompatibilityError: If the columns or values violate the observation
            contract.
    """
    if not frame.columns.is_unique:
        raise IncompatibilityError(
            f"Input columns for node '{node.name}' must be unique."
        )
    _validate_input_columns(node, frame.columns)
    if frame.empty:
        return

    _validate_parameter_frame(
        node.inputs,
        frame,
        f"node '{node.name}'",
        numerical_measurements_must_be_within_tolerance,
    )

    for name in node.parent_output_names:
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


def _validate_input_columns(node: MultiStepNode, columns: Iterable[str]) -> None:
    """Validate that the input columns match the node's model input."""
    expected = set(node.model_input_names)
    actual = set(columns)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise IncompatibilityError(
            f"The inputs of node '{node.name}' do not match its model input. "
            f"Missing columns: {missing}, unexpected columns: {unexpected}. "
            f"Expected columns: {list(node.model_input_names)}."
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
