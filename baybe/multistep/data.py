"""Measurement store for multi-step campaigns.

A :class:`NodeMeasurement` is one observation of one node. Its inputs are the
full input of the node model: the values of the node's own parameters together
with the observed outputs of its parents. An optional ``experiment_id`` groups
measurements that belong to the same DAG evaluation; once every node of an
experiment has been measured, the campaign assembles a complete DAG row from it.
Node training data is derived from the stored measurements; nothing is stored
twice.

Complete DAG rows are stored separately because the stock acquisition-function
machinery needs joint parameter configurations for its reference computations.
The campaign assembles such a row from the node measurements once an experiment
is complete; rows of fully successful experiments can also be imported directly.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
from attrs import define, evolve, field
from attrs.validators import instance_of, min_len

from baybe.exceptions import (
    IncompatibilityError,
    IncompleteMeasurementsError,
    InvalidTargetValueError,
)
from baybe.serialization import SerialMixin
from baybe.utils.basic import to_tuple

from ._columns import (
    COMPLETE_SOURCE,
    COMPLETE_SOURCE_DERIVED,
    COMPLETE_SOURCE_IMPORTED,
    EXPERIMENT_ID,
    FEASIBILITY_LABEL,
)
from ._experiments import _EvaluationRef
from .dag import MultiStepDAG
from .node import MultiStepNode


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


def _normalize_complete_frame(
    value: pd.DataFrame,
    *,
    default_source: str = COMPLETE_SOURCE_IMPORTED,
) -> pd.DataFrame:
    """Copy complete rows and establish their observation provenance.

    Args:
        value: Complete DAG rows, optionally carrying a provenance marker.
        default_source: Provenance assigned to rows without a marker.

    Returns:
        A defensive copy with a valid provenance column.

    Raises:
        IncompatibilityError: If the provenance column contains unknown values.
    """
    frame = value.copy().reset_index(drop=True)
    if COMPLETE_SOURCE not in frame.columns:
        frame[COMPLETE_SOURCE] = default_source
        return frame

    source = frame[COMPLETE_SOURCE]
    invalid = sorted(
        {
            value
            for value in source.dropna().unique()
            if value not in (COMPLETE_SOURCE_IMPORTED, COMPLETE_SOURCE_DERIVED)
        }
    )
    if invalid:
        raise IncompatibilityError(
            f"Unknown complete-measurement provenance values: {invalid}. "
            f"Expected '{COMPLETE_SOURCE_IMPORTED}' or '{COMPLETE_SOURCE_DERIVED}'."
        )
    frame[COMPLETE_SOURCE] = source.fillna(default_source)
    return frame


@define(frozen=True)
class NodeMeasurement(SerialMixin):
    """One observation of one node."""

    node: str = field(validator=min_len(1))
    """The observed node."""

    inputs: Mapping[str, Any] = field(
        converter=lambda value: MappingProxyType(dict(value)),
        validator=instance_of(Mapping),
    )
    """The node's own parameter values and observed parent outputs."""

    output: float | None = field(default=None)
    """The real observed output, absent only for explicitly failed observations."""

    experiment_id: str | None = field(default=None, eq=False)
    """The optional experiment this measurement belongs to."""

    condition_values: Mapping[str, Any] = field(
        factory=dict,
        converter=lambda value: MappingProxyType(dict(value)),
        validator=instance_of(Mapping),
        kw_only=True,
    )
    """Observed auxiliary columns used by this node's feasibility conditions."""

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
    """Immutable store of node measurements and complete DAG rows."""

    measurements: tuple[NodeMeasurement, ...] = field(
        factory=tuple,
        converter=to_tuple,
        validator=instance_of(tuple),
    )
    """The individual node measurements, in insertion order."""

    complete_measurements: pd.DataFrame = field(
        factory=pd.DataFrame,
        converter=_normalize_complete_frame,
        validator=instance_of(pd.DataFrame),
    )
    """Complete successful rows with parameters, outputs, and condition values.

    Rows assembled from node measurements are marked as derived reference views.
    Imported rows are marked as independent observations. Derived rows never
    enter training views; their bound node measurements already do.
    """

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

    def add_complete(
        self, measurements: pd.DataFrame, *, derived: bool = False
    ) -> MultiStepData:
        """Return a new store with the given complete rows appended.

        Args:
            measurements: Complete DAG rows in experimental representation.
            derived: Whether the rows are reference views assembled from stored
                node measurements. Imported rows are independent observations.

        Returns:
            The updated immutable store.

        Raises:
            IncompatibilityError: If a derived row carries an incompatible
                provenance marker.
        """
        source = COMPLETE_SOURCE_DERIVED if derived else COMPLETE_SOURCE_IMPORTED
        frame = _normalize_complete_frame(measurements, default_source=source)
        if derived and (frame[COMPLETE_SOURCE] != COMPLETE_SOURCE_DERIVED).any():
            raise IncompatibilityError(
                "Rows added as derived complete measurements must not claim "
                "independent provenance."
            )
        if not derived and (frame[COMPLETE_SOURCE] == COMPLETE_SOURCE_DERIVED).any():
            raise IncompatibilityError(
                "Rows added as imported complete measurements must not claim "
                "derived provenance."
            )
        combined = pd.concat([self.complete_measurements, frame], ignore_index=True)
        return evolve(self, complete_measurements=combined)

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
        if ref.source == "measurement":
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

        if ref.source == "complete":
            node = dag.node_by_name.get(ref.node)
            if node is None:
                raise IncompatibilityError(f"Node '{ref.node}' is not part of the DAG.")
            if ref.index < 0 or ref.index >= len(self.complete_measurements):
                raise IncompatibilityError(
                    f"Complete-measurement reference {ref.index} is out of range."
                )
            row = self.complete_measurements.iloc[ref.index]
            missing = [
                name
                for name in node.model_input_names
                if name not in self.complete_measurements.columns
            ]
            if missing:
                raise IncompatibilityError(
                    f"The complete measurement for node '{node.name}' is missing "
                    f"required columns: {missing}."
                )
            experiment_id = (
                str(row[EXPERIMENT_ID])
                if EXPERIMENT_ID in self.complete_measurements
                and not pd.isna(row[EXPERIMENT_ID])
                else None
            )
            return NodeMeasurement(
                node=node.name,
                inputs={name: row[name] for name in node.model_input_names},
                output=row[node.output],
                experiment_id=experiment_id,
                condition_values={
                    condition.column: row[condition.column]
                    for condition in node.feasibility_conditions
                    if condition.column != node.output
                    and condition.column in self.complete_measurements
                },
            )

        raise IncompatibilityError(f"Unsupported evaluation source '{ref.source}'.")

    def evaluation_refs(  # noqa: DOC502
        self, dag: MultiStepDAG, node: str, *, successful_only: bool = True
    ) -> tuple[_EvaluationRef, ...]:
        """Collect references to actual evaluations of one node.

        Derived complete rows are reference views of stored measurements and are
        deliberately excluded. Imported complete rows are genuine independent
        observations and remain available.

        Args:
            dag: The DAG topology.
            node: The node whose evaluations are requested.
            successful_only: Whether failed observations should be omitted.

        Returns:
            The references in measurement-then-import order.

        Raises:
            IncompatibilityError: If the node is not part of the DAG.
        """
        node_object = _get_node(dag, node)
        refs = [
            _EvaluationRef(source="measurement", index=index, node=node)
            for index, measurement in enumerate(self.measurements)
            if measurement.node == node
            and (not successful_only or _is_successful(node_object, measurement))
        ]
        if COMPLETE_SOURCE in self.complete_measurements:
            imported = (
                self.complete_measurements[COMPLETE_SOURCE] == COMPLETE_SOURCE_IMPORTED
            )
            refs.extend(
                _EvaluationRef(source="complete", index=int(index), node=node)
                for index in self.complete_measurements.index[imported]
            )
        return tuple(refs)

    def is_experiment_complete(self, dag: MultiStepDAG, experiment_id: str) -> bool:
        """Check whether every DAG node has a measurement for the given experiment.

        Args:
            dag: The DAG topology.
            experiment_id: The experiment identifier.

        Returns:
            ``True`` if every node has a successful measurement with the given
            identifier and none of that experiment's observations failed.
        """
        if EXPERIMENT_ID in self.complete_measurements:
            derived = self.complete_measurements[
                (self.complete_measurements[EXPERIMENT_ID] == experiment_id)
                & (
                    self.complete_measurements[COMPLETE_SOURCE]
                    == COMPLETE_SOURCE_DERIVED
                )
            ]
            if not derived.empty:
                return True

        measured_nodes = {
            measurement.node
            for measurement in self.measurements
            if measurement.experiment_id == experiment_id
        }
        return measured_nodes == {node.name for node in dag.nodes} and all(
            _is_successful(dag.node_by_name[m.node], m)
            for m in self.measurements
            if m.experiment_id == experiment_id
        )

    def assemble_complete_row(
        self, dag: MultiStepDAG, experiment_id: str
    ) -> pd.DataFrame:
        """Assemble a complete DAG row from the measurements of an experiment.

        This legacy helper only considers measurements carrying the experiment
        identifier. The campaign completes contexts from their explicit bindings.

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
            m.node: _EvaluationRef(source="measurement", index=index, node=m.node)
            for index, m in enumerate(self.measurements)
            if m.experiment_id == experiment_id
        }
        return self.make_complete_row(dag, bindings)

    def has_consistent_bindings(
        self, dag: MultiStepDAG, bindings: Mapping[str, _EvaluationRef]
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
            resolved[node.name].inputs[parent.output] == resolved[parent.name].output
            for node in dag.nodes
            for parent in node.parents
        )

    def make_complete_row(
        self, dag: MultiStepDAG, bindings: Mapping[str, _EvaluationRef]
    ) -> pd.DataFrame:
        """Assemble a complete DAG row from concrete evaluation bindings.

        The row is a reference view. It is not stored as a new measurement unless
        the campaign explicitly appends it with derived provenance.

        Args:
            dag: The DAG topology.
            bindings: Actual evaluations for all nodes, keyed by node name.

        Returns:
            A single-row dataframe with one column per DAG parameter and one per
            node output.

        Raises:
            IncompatibilityError: If a node is unbound, an evaluation failed, or
                the bound parent outputs disagree with a child's local input.
        """
        missing = sorted({node.name for node in dag.nodes} - set(bindings))
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
                raise IncompatibilityError(
                    f"Cannot assemble a complete row because node '{node.name}' failed."
                )
            for parent in node.parents:
                if measurement.inputs[parent.output] != resolved[parent.name].output:
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
                model input, contains invalid values, or the complete rows do
                not cover the DAG columns.
            IncompleteMeasurementsError: If configured condition values are missing.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        failed: set[tuple[str, str]] = set()
        for measurement in self.measurements:
            node = _get_node(dag, measurement.node)
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
            if is_successful and measurement.output is None:
                raise IncompatibilityError(
                    f"Successful observation of node '{node.name}' requires an output."
                )
            if not is_successful and measurement.experiment_id is not None:
                failed.add((measurement.experiment_id, measurement.node))
            _validate_input_columns(node, measurement.inputs.keys())
            for name in node.model_input_names:
                value = measurement.inputs[name]
                if name in dag.parameter_by_name:
                    _validate_parameter_value(
                        dag,
                        name,
                        value,
                        node.name,
                        numerical_measurements_must_be_within_tolerance,
                    )
                else:
                    _validate_finite_input(name, value, node.name)

        for measurement in self.measurements:
            if any(
                (measurement.experiment_id, ancestor) in failed
                for ancestor in dag.ancestors_of(measurement.node)
            ):
                raise IncompatibilityError(
                    f"Node '{measurement.node}' has a failed ancestor in experiment "
                    f"'{measurement.experiment_id}'."
                )

        if self.complete_measurements.empty:
            return

        frame = self.complete_measurements
        if frame.columns.duplicated().any():
            duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
            raise IncompatibilityError(
                f"Complete measurement columns must be unique, but found "
                f"duplicates: {duplicates}."
            )

        missing = [
            column
            for column in dag._complete_measurement_columns
            if column not in frame.columns
        ]
        if missing:
            raise IncompatibilityError(
                f"Complete measurements are missing required columns: {missing}."
            )

        for node in dag.nodes:
            if node.feasibility_conditions:
                success = _extract_feasibility_values(node, frame)
                if not success.all():
                    raise IncompatibilityError(
                        f"Complete measurements require all feasibility conditions "
                        f"to pass for node '{node.name}'."
                    )

        for _, row in frame.iterrows():
            for name in dag.parameter_by_name:
                _validate_parameter_value(
                    dag,
                    name,
                    row[name],
                    name,
                    numerical_measurements_must_be_within_tolerance,
                )
            for node in dag.nodes:
                value = row[node.output]
                if isinstance(value, bool) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise IncompatibilityError(
                        f"The complete-measurement output of node '{node.name}' "
                        f"must be numeric, got {value!r}."
                    )
                if not np.isfinite(value):
                    raise IncompatibilityError(
                        f"The complete-measurement output of node '{node.name}' "
                        f"must be finite, got {value!r}."
                    )

    def node_training_data(self, dag: MultiStepDAG, node: str) -> pd.DataFrame:
        """Return supervised training rows for one node model.

        Only successful observations enter this view, even when a failed observation
        includes a real output. Genuine repetitions are retained; complete-row caches
        backed by node observations are excluded using experiment provenance.

        Args:
            dag: The DAG topology.
            node: The node name.

        Returns:
            One row per measurement with columns for the node's own parameters,
            its parent outputs, and the observed output.
        """
        return self._node_training_data(dag, node, feasibility=False)

    def node_feasibility_training_data(
        self, dag: MultiStepDAG, node: str
    ) -> pd.DataFrame:
        """Return all raw feasibility observations with their full local inputs.

        Genuine repetitions retain their multiplicity; only complete-row caches
        backed by node observations are excluded using experiment provenance.

        Args:
            dag: The DAG topology.
            node: The node name.

        Returns:
            A raw-label training frame, empty for unconfigured nodes.
        """
        return self._node_training_data(dag, node, feasibility=True)

    def _node_training_data(
        self, dag: MultiStepDAG, node: str, *, feasibility: bool
    ) -> pd.DataFrame:
        """Project observations and independent imports without erasing repeats."""
        node_object = _get_node(dag, node)
        if feasibility and not node_object.feasibility_conditions:
            return pd.DataFrame(
                columns=[*node_object.model_input_names, FEASIBILITY_LABEL]
            )
        label = FEASIBILITY_LABEL if feasibility else node_object.output
        columns = [*node_object.model_input_names, label]

        frames: list[pd.DataFrame] = []
        if not self.complete_measurements.empty:
            complete = self.complete_measurements.loc[
                self.complete_measurements[COMPLETE_SOURCE] == COMPLETE_SOURCE_IMPORTED
            ]
            if not complete.empty:
                if feasibility:
                    frame = complete[list(node_object.model_input_names)].copy()
                    frame[label] = _extract_feasibility_values(node_object, complete)
                    frames.append(frame[columns])
                else:
                    frames.append(complete[columns].copy())

        rows = [
            {**measurement.inputs, label: _is_successful(node_object, measurement)}
            if feasibility
            else {**measurement.inputs, label: measurement.output}
            for measurement in self.measurements
            if measurement.node == node
            and (feasibility or _is_successful(node_object, measurement))
        ]
        if rows:
            frames.append(pd.DataFrame(rows, columns=columns))

        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True)


def _is_successful(node: MultiStepNode, measurement: NodeMeasurement) -> bool:
    """Derive success from output and condition values without duplicating state."""
    return _condition_success(
        node, {node.output: measurement.output, **measurement.condition_values}
    )


def _validate_node_inputs(
    dag: MultiStepDAG,
    node: MultiStepNode,
    frame: pd.DataFrame,
    numerical_measurements_must_be_within_tolerance: bool = True,
) -> None:
    """Validate a local input frame using the existing observation contract.

    Args:
        dag: The DAG defining the parameters.
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
    for row in frame.to_dict("records"):
        for name in node.model_input_names:
            value = row[name]
            if name in dag.parameter_by_name:
                _validate_parameter_value(
                    dag,
                    name,
                    value,
                    node.name,
                    numerical_measurements_must_be_within_tolerance,
                )
            else:
                _validate_finite_input(name, value, node.name)


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


def _validate_parameter_value(
    dag: MultiStepDAG,
    name: str,
    value: Any,
    node: str,
    numerical_measurements_must_be_within_tolerance: bool = True,
) -> None:
    """Validate one parameter value using the parameter encoding.

    Args:
        dag: The DAG owning the parameter.
        name: The parameter name.
        value: The value to validate.
        node: The node the value is used by.
        numerical_measurements_must_be_within_tolerance: If ``True``, numerical
            values must lie within the parameter range or tolerance. If ``False``,
            numerical values bypass the range check but must still be finite numbers.

    Raises:
        IncompatibilityError: If the value violates the parameter contract.
    """
    parameter = dag.parameter_by_name[name]
    if parameter.is_numerical and not numerical_measurements_must_be_within_tolerance:
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise IncompatibilityError(
                f"The numerical parameter '{name}' used by node '{node}' must be "
                f"numeric, got {value!r}."
            )
        if not np.isfinite(value):
            raise IncompatibilityError(
                f"The numerical parameter '{name}' used by node '{node}' must be "
                f"finite, got {value!r}."
            )
        return
    try:
        in_range = parameter.is_in_range(value)
    except Exception as ex:  # noqa: BLE001 - translated into a domain error
        raise IncompatibilityError(
            f"Invalid value {value!r} for parameter '{name}' used by node '{node}'."
        ) from ex
    if not in_range:
        raise IncompatibilityError(
            f"Invalid value {value!r} for parameter '{name}' used by node '{node}'."
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
    "MultiStepData",
    "NodeMeasurement",
]


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
