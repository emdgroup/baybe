"""Measurement store for multi-step campaigns.

A :class:`NodeMeasurement` is one observation of one node. Its inputs are the
full input of the node model: the values of the node's own parameters together
with the observed outputs of its parents. Node training data is derived from the
stored measurements; nothing is stored twice.

Complete DAG rows are stored separately because the stock acquisition-function
machinery needs joint parameter configurations for its reference computations.
The campaign assembles such a row from the node measurements once a plan is
complete; rows of finished plans can also be imported directly.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import instance_of, min_len

from baybe.exceptions import IncompatibilityError

from .dag import MultiStepDAG
from .node import MultiStepNode


def _to_frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Copy a mapping into an immutable proxy."""
    return MappingProxyType(dict(value))


def _to_measurement_tuple(
    value: Iterable[NodeMeasurement],
) -> tuple[NodeMeasurement, ...]:
    """Copy node measurements into a tuple."""
    return tuple(value)


def _copy_frame(value: pd.DataFrame) -> pd.DataFrame:
    """Copy a dataframe so that the stored state cannot be mutated from outside."""
    return value.copy()


def _python_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


@define(frozen=True)
class NodeMeasurement:
    """One observation of one node.

    Args:
        node: The name of the observed node.
        inputs: The full node-model input, that is, the values of the node's own
            parameters and the observed outputs of its parents.
        output: The observed node output.
    """

    node: str = field(validator=min_len(1))
    """The observed node."""

    inputs: Mapping[str, Any] = field(
        converter=_to_frozen_mapping, validator=instance_of(Mapping)
    )
    """The full node-model input."""

    output: float = field()
    """The observed node output."""

    @output.validator
    def _validate_output(self, _: Any, value: float) -> None:  # noqa: DOC101, DOC103
        """Validate that the output is a finite number."""
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
        object.__setattr__(self, "output", float(self.output))


@define(frozen=True, eq=False)
class MultiStepData:
    """Immutable store of node measurements and complete DAG rows.

    Args:
        measurements: The individual node measurements, in insertion order.
        complete_measurements: Complete DAG rows in experimental
            representation. The frame contains one column per DAG parameter and
            one column per node output.
    """

    measurements: tuple[NodeMeasurement, ...] = field(
        factory=tuple,
        converter=_to_measurement_tuple,
        validator=instance_of(tuple),
    )
    """The individual node measurements."""

    complete_measurements: pd.DataFrame = field(
        factory=pd.DataFrame, converter=_copy_frame, validator=instance_of(pd.DataFrame)
    )
    """Complete DAG rows used by the acquisition-function machinery."""

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
        return MultiStepData(
            self.measurements + tuple(measurements), self.complete_measurements
        )

    def add_complete(self, measurements: pd.DataFrame) -> MultiStepData:
        """Return a new store with the given complete rows appended.

        Args:
            measurements: Complete DAG rows in experimental representation.

        Returns:
            The updated immutable store.
        """
        frame = pd.concat([self.complete_measurements, measurements], ignore_index=True)
        return MultiStepData(self.measurements, frame)

    def validate(self, dag: MultiStepDAG) -> None:
        """Validate the stored measurements against the DAG.

        Args:
            dag: The DAG defining the expected data contract.

        Raises:
            IncompatibilityError: If a measurement does not match the node's
                model input, contains invalid values, or the complete rows do
                not cover the DAG columns.
        """
        for measurement in self.measurements:
            node = _get_node(dag, measurement.node)
            _validate_input_columns(node, measurement.inputs.keys())
            for name in node.model_input_names:
                value = measurement.inputs[name]
                if name in dag.parameter_by_name:
                    _validate_parameter_value(dag, name, value, node.name)
                else:
                    _validate_finite_input(name, value, node.name)

        if self.complete_measurements.empty:
            return

        frame = self.complete_measurements
        if frame.columns.duplicated().any():
            duplicates = sorted(set(frame.columns[frame.columns.duplicated()]))
            raise IncompatibilityError(
                f"Complete measurement columns must be unique, but found "
                f"duplicates: {duplicates}."
            )

        required = (*dag.parameter_by_name, *(node.output for node in dag.nodes))
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise IncompatibilityError(
                f"Complete measurements are missing required columns: {missing}."
            )

        for _, row in frame.iterrows():
            for name in dag.parameter_by_name:
                _validate_parameter_value(dag, name, row[name], name)
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

        The rows are collected from both the individual node measurements and
        the complete DAG rows, without duplicating stored information.

        Args:
            dag: The DAG topology.
            node: The node name.

        Returns:
            One row per measurement with columns for the node's own parameters,
            its parent outputs, and the observed output.
        """
        node_object = _get_node(dag, node)
        columns = [*node_object.model_input_names, node_object.output]

        frames: list[pd.DataFrame] = []
        if not self.complete_measurements.empty:
            frames.append(self.complete_measurements[columns].copy())

        rows = [
            {**measurement.inputs, node_object.output: measurement.output}
            for measurement in self.measurements
            if measurement.node == node
        ]
        if rows:
            frames.append(pd.DataFrame(rows, columns=columns))

        if not frames:
            return pd.DataFrame(columns=columns)
        return pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)


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
    dag: MultiStepDAG, name: str, value: Any, node: str
) -> None:
    """Validate one parameter value using the parameter encoding."""
    parameter = dag.parameter_by_name[name]
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
