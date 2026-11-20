"""Private experiment snapshots and stateless bookkeeping helpers.

A :class:`~baybe.multistep.data.NodeMeasurement` is the only stored output of an
execution. Experiments reference those observations by an internal
:class:`_EvaluationRef`; they never copy outputs or training rows. A snapshot also
stores the concrete local inputs of pending actions so that a recommendation can
be validated when its result is reported.

The types in this module are private. The campaign remains the single mutable
owner of process state.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from attrs import define, evolve, field

from baybe.exceptions import IncompatibilityError

if TYPE_CHECKING:
    from .dag import MultiStepDAG
    from .data import MultiStepData


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only copy of a mapping."""
    return MappingProxyType(dict(value))


@define(frozen=True)
class _EvaluationRef:
    """Stable internal reference to one actual evaluation of one node."""

    source: Literal["measurement", "complete"] = field()
    """The table holding the evaluation."""

    index: int = field()
    """The position inside the append-only table."""

    node: str = field()
    """The node whose evaluation is referenced."""


@define(frozen=True)
class _PendingAction:
    """A recommended but not yet reported local node execution."""

    node: str = field()
    """The recommended node."""

    inputs: Mapping[str, Any] = field(converter=_freeze_mapping)
    """The complete local input, including real parent outputs."""

    parent_refs: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """The concrete parent evaluations supplying the local input."""


@define(frozen=True)
class _Experiment:
    """Immutable snapshot of one evaluation context."""

    experiment_id: str = field()
    """The automatically generated context identifier."""

    bindings: Mapping[str, _EvaluationRef] = field(
        factory=dict, converter=_freeze_mapping
    )
    """Actual evaluations bound to the context, keyed by node.

    The mapping contains both evaluations measured for this context and
    evaluations adopted from earlier contexts.
    """

    parent_bindings: Mapping[str, Mapping[str, _EvaluationRef]] = field(
        factory=dict, converter=_freeze_mapping
    )
    """Parent evaluations used by each node binding, keyed by node."""

    pending: Mapping[str, _PendingAction] = field(
        factory=dict, converter=_freeze_mapping
    )
    """Recommended local executions awaiting their result, keyed by node."""

    def with_binding(
        self,
        node: str,
        ref: _EvaluationRef,
        parent_refs: Mapping[str, _EvaluationRef],
    ) -> _Experiment:
        """Return a snapshot with one additional evaluation binding.

        The concrete parent evaluations are adopted into the context as well.
        A parent that is already bound to a different evaluation is rejected
        because the context would otherwise describe conflicting local inputs.

        Args:
            node: The node whose evaluation becomes available.
            ref: The actual evaluation.
            parent_refs: The concrete parent evaluations used by the action.

        Returns:
            The updated snapshot.

        Raises:
            IncompatibilityError: If a parent is already bound to a different
                evaluation.
        """
        bindings = dict(self.bindings)
        for parent, parent_ref in parent_refs.items():
            existing = bindings.get(parent)
            if existing is not None and existing != parent_ref:
                raise IncompatibilityError(
                    f"Parent node '{parent}' is already bound to a different "
                    f"evaluation in experiment '{self.experiment_id}'."
                )
            bindings[parent] = parent_ref
        bindings[node] = ref
        parent_bindings = {**self.parent_bindings, node: dict(parent_refs)}
        return evolve(self, bindings=bindings, parent_bindings=parent_bindings)

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

        The concrete parent evaluations of the action are adopted into the
        context immediately. They are real observations even though the
        selected action itself has not been reported yet.

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

        bindings = dict(self.bindings)
        for parent, ref in action.parent_refs.items():
            existing = bindings.get(parent)
            if existing is not None and existing != ref:
                raise IncompatibilityError(
                    f"Parent node '{parent}' is already bound to a different "
                    f"evaluation in experiment '{self.experiment_id}'."
                )
            bindings[parent] = ref
        return evolve(
            self,
            bindings=bindings,
            pending={**self.pending, action.node: action},
        )


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
) -> bool:
    """Return whether an identical local action is already pending.

    Inputs are compared exactly. Numerically equal but separately reported
    executions therefore remain distinct actions and training observations.

    Args:
        actions: The pending actions to search.
        node: The node of the candidate action.
        inputs: The complete local input of the candidate action.

    Returns:
        ``True`` if an identical node-and-input action is pending.
    """
    return any(
        action.node == node and dict(action.inputs) == dict(inputs)
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
    from .data import _is_successful

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
    """Return whether a context contains a failed evaluation.

    A failed node closes its own context for further automatic continuation.
    Independent contexts remain unaffected.

    Args:
        experiment: The snapshot to inspect.
        dag: The DAG topology.
        data: The store resolving evaluation references.

    Returns:
        ``True`` if any bound evaluation failed.
    """
    return bool(failed_nodes(experiment, dag, data))


def next_executable_node(
    experiment: _Experiment, dag: MultiStepDAG, data: MultiStepData
) -> tuple[str, dict[str, _EvaluationRef]] | None:
    """Return the next executable node of a context.

    Args:
        experiment: The context to continue.
        dag: The DAG topology.
        data: The store resolving evaluation references.

    Returns:
        The node and its concrete parent bindings, or ``None`` if the context has
        no executable node left.
    """
    if experiment_is_closed(experiment, dag, data):
        return None
    for name in dag.topological_order:
        if name in experiment.bindings or name in experiment.pending:
            continue
        parents = dag.parents_of(name)
        if all(parent in experiment.bindings for parent in parents):
            return name, {parent: experiment.bindings[parent] for parent in parents}
    return None


def choose_action_context(
    experiments: Mapping[str, _Experiment],
    *,
    node: str,
    parent_refs: Mapping[str, _EvaluationRef],
    data: MultiStepData,
    dag: MultiStepDAG,
) -> _Experiment | None:
    """Find a deterministic existing context compatible with an action.

    A context is compatible when the node is neither measured nor pending, no
    failed evaluation is present, no parent of the action is still pending in the
    context, and every parent binding already present agrees with the action's
    concrete parent evaluation. Missing parent bindings are adopted from the
    action.

    Args:
        experiments: All known contexts in creation order.
        node: The selected node.
        parent_refs: The concrete parent evaluations of the action.
        data: The store resolving evaluation references.
        dag: The DAG topology.

    Returns:
        The first compatible context, or ``None`` if a new context is required.
    """
    for experiment in experiments.values():
        if node in experiment.bindings or node in experiment.pending:
            continue
        if experiment_is_closed(experiment, dag, data):
            continue
        if any(parent in experiment.pending for parent in parent_refs):
            continue
        if any(
            parent in experiment.bindings and experiment.bindings[parent] != ref
            for parent, ref in parent_refs.items()
        ):
            continue
        return experiment
    return None


def register_action(
    experiments: Mapping[str, _Experiment],
    counter: int,
    *,
    node: str,
    inputs: Mapping[str, Any],
    parent_refs: Mapping[str, _EvaluationRef],
    data: MultiStepData,
    dag: MultiStepDAG,
) -> tuple[dict[str, _Experiment], int, str]:
    """Reserve one actual action in an existing or new context.

    Args:
        experiments: All contexts in creation order.
        counter: The last allocated numeric experiment counter.
        node: The selected node.
        inputs: The complete local input of the action.
        parent_refs: The concrete parent evaluations of the action.
        data: The store resolving evaluation references.
        dag: The DAG topology.

    Returns:
        The updated contexts, the updated counter, and the assigned identifier.

    Raises:
        IncompatibilityError: If the action duplicates an existing pending action,
            names an unknown node, omits or contradicts a parent evaluation, or
            carries an invalid local input.
    """
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

    from .data import _is_successful, _validate_node_inputs

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

    _validate_node_inputs(dag, node_object, pd.DataFrame([inputs]))

    action = _PendingAction(node=node, inputs=inputs, parent_refs=parent_refs)
    context = choose_action_context(
        experiments,
        node=node,
        parent_refs=parent_refs,
        data=data,
        dag=dag,
    )
    updated = dict(experiments)
    if context is None:
        known = _known_experiment_ids(data, experiments)
        while True:
            counter += 1
            experiment_id = f"exp-{counter}"
            if experiment_id not in known:
                break
        context = _Experiment(experiment_id=experiment_id)
    else:
        experiment_id = context.experiment_id

    updated[experiment_id] = context.with_pending(action)
    return updated, counter, experiment_id


def _known_experiment_ids(
    data: MultiStepData, experiments: Mapping[str, _Experiment]
) -> set[str]:
    """Collect all identifiers that a new context must not reuse."""
    from ._columns import EXPERIMENT_ID

    known = set(experiments)
    known.update(
        measurement.experiment_id
        for measurement in data.measurements
        if measurement.experiment_id is not None
    )
    if EXPERIMENT_ID in data.complete_measurements:
        known.update(
            str(value)
            for value in data.complete_measurements[EXPERIMENT_ID].dropna().unique()
        )
    return known


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
