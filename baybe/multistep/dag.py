"""DAG topology for multi-step campaigns."""

from __future__ import annotations

import copy
import gc
from collections import Counter
from collections.abc import Mapping
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from math import prod
from typing import TYPE_CHECKING, Any

from attrs import define, evolve, field
from attrs.validators import and_, deep_iterable, instance_of, min_len

from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.serialization import SerialMixin, converter
from baybe.targets.base import Target
from baybe.utils.basic import refers_to, to_tuple

from ._columns import RESERVED_COLUMNS
from .node import MultiStepNode

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter

    from .model import MultiStepModel


@define(frozen=True, slots=False)
class MultiStepDAG(SerialMixin):
    """A directed acyclic graph of node definitions.

    The DAG is the single source of truth for the topology, the per-node model
    configurations, and the optimization objective. It holds no experiment state,
    no inventory, and no fitted models.
    """

    nodes: tuple[MultiStepNode, ...] = field(
        converter=to_tuple,
        validator=and_(min_len(1), deep_iterable(instance_of(MultiStepNode))),
    )
    """All nodes in the DAG. Their order is irrelevant."""

    objective: Objective = field(validator=instance_of(Objective))
    """The optimization objective, whose targets cover all graph outputs."""

    @nodes.validator
    def _validate_nodes(self, _attribute, _value):  # noqa: DOC101, DOC103
        """Validate topology, names, and encodings."""
        # Also rejects duplicate node names, unknown parents, and cycles.
        _ = self.topological_order

        dag_nodes = {id(node) for node in self.nodes}
        for node in self.nodes:
            for parent in node.parents:
                if id(parent) not in dag_nodes:
                    raise IncompatibilityError(
                        f"Node '{node.name}' references parent '{parent.name}' "
                        f"which is not part of the DAG."
                    )

        for node in self.nodes:
            if not node.inputs and not node.parents:
                raise IncompatibilityError(
                    f"Node '{node.name}' has neither own inputs nor parents. A root "
                    f"node must define at least one input parameter."
                )

        outputs = [node.output for node in self.nodes]
        duplicates = sorted(
            name for name, count in Counter(outputs).items() if count > 1
        )
        if duplicates:
            raise IncompatibilityError(
                f"Node output names must be unique, but found duplicates: {duplicates}."
            )

        parameter_names = [
            parameter.name for node in self.nodes for parameter in node.inputs
        ]
        duplicates = sorted(
            name for name, count in Counter(parameter_names).items() if count > 1
        )
        if duplicates:
            raise IncompatibilityError(
                f"Parameter names must be unique across nodes, but found "
                f"duplicates: {duplicates}."
            )

        comp_rep_columns = [
            column for node in self.nodes for column in node._own_comp_rep_columns
        ]

        auxiliary_columns: dict[str, str] = {}
        for node in self.nodes:
            condition_types: dict[str, bool] = {}
            for condition in node.feasibility_conditions:
                column = condition.column
                if column == node.output:
                    if condition.operator == "is_true":
                        raise IncompatibilityError(
                            f"Node '{node.name}' has a numeric output; Boolean "
                            "feasibility conditions must use an auxiliary column."
                        )
                    continue
                is_boolean = condition.operator == "is_true"
                previous_type = condition_types.setdefault(column, is_boolean)
                if previous_type != is_boolean:
                    raise IncompatibilityError(
                        f"Feasibility column '{column}' of node '{node.name}' cannot "
                        "use both Boolean and numerical conditions."
                    )
                if column in outputs or column in parameter_names:
                    raise IncompatibilityError(
                        f"Auxiliary feasibility column '{column}' "
                        f"of node '{node.name}' "
                        "must not collide with another node output or parameter."
                    )
                if column in comp_rep_columns or column in RESERVED_COLUMNS:
                    raise IncompatibilityError(
                        f"Auxiliary feasibility column '{column}' "
                        f"of node '{node.name}' "
                        "must not collide with computational or reserved columns."
                    )
                owner = auxiliary_columns.setdefault(column, node.name)
                if owner != node.name:
                    raise IncompatibilityError(
                        f"Auxiliary feasibility columns must be unique per node; "
                        f"'{column}' is used by both '{owner}' and '{node.name}'."
                    )

        collisions = sorted(
            set(outputs) & (set(parameter_names) | set(comp_rep_columns))
        )
        if collisions:
            raise IncompatibilityError(
                f"Node outputs must not collide with parameter names or "
                f"computational-representation columns: {collisions}."
            )

        collisions = sorted(
            set(RESERVED_COLUMNS)
            & (set(parameter_names) | set(outputs) | set(comp_rep_columns))
        )
        if collisions:
            raise IncompatibilityError(
                f"Parameter and output names must not use reserved columns: "
                f"{collisions}."
            )

        for node in self.nodes:
            self._validate_node_searchspace(node)

    @objective.validator
    def _validate_objective(self, _attribute, value):  # noqa: DOC101, DOC103
        """Validate the objective against the graph outputs and target names."""
        target_names = [target.name for target in value.targets]
        duplicates = sorted(
            name for name, count in Counter(target_names).items() if count > 1
        )
        if duplicates:
            raise IncompatibilityError(
                f"Objective target names must be unique, but found duplicates: "
                f"{duplicates}."
            )

        leaf_outputs = set(self.graph_output_names)
        unknown = sorted(set(target_names) - leaf_outputs)
        if unknown:
            raise IncompatibilityError(
                f"Objective targets {unknown} do not match any graph output. Only "
                f"the outputs of topological leaf nodes can be optimized, but found "
                f"targets that are not graph outputs. Graph outputs: "
                f"{sorted(leaf_outputs)}."
            )

        missing = sorted(leaf_outputs - set(target_names))
        if missing:
            raise IncompatibilityError(
                f"The objective must include every graph output, but targets for "
                f"{missing} are missing."
            )

    @staticmethod
    def _validate_node_searchspace(node: MultiStepNode) -> None:
        """Verify that a node's search space can be rebuilt from its parameters."""
        if not node.inputs:
            return
        searchspace = node.searchspace
        if searchspace.is_constrained:
            raise IncompatibilityError(
                f"Node '{node.name}' has a constrained search space, which is "
                f"not supported by multi-step planning."
            )
        if searchspace.discrete.is_empty:
            return
        n_expected = prod(len(p.values) for p in searchspace.discrete.parameters)
        if len(searchspace.discrete.exp_rep) != n_expected:
            raise IncompatibilityError(
                f"Node '{node.name}' has a restricted discrete candidate set, "
                f"which is not supported by multi-step planning."
            )

    @property
    def _complete_measurement_columns(self) -> tuple[str, ...]:
        """The parameter, output, and condition columns of a complete DAG row."""
        auxiliary_columns = dict.fromkeys(
            condition.column
            for node in self.nodes
            for condition in node.feasibility_conditions
            if condition.column != node.output
        )
        return tuple(
            dict.fromkeys(
                (
                    *self.parameter_by_name,
                    *(node.output for node in self.nodes),
                    *auxiliary_columns,
                )
            )
        )

    @property
    def final_nodes(self) -> tuple[str, ...]:
        """The topological leaf node names, in topological order.

        The set is defined by the graph topology alone and is independent of the
        order in which the objective lists its targets.
        """
        return self.leaf_nodes

    @property
    def final_output_dim(self) -> int:
        """Total number of outputs produced by all graph leaves."""
        return len(self.graph_output_names)

    @property
    def final_target_names(self) -> tuple[str, ...]:
        """Target names of the objective, in objective order."""
        return tuple(target.name for target in self.objective.targets)

    @property
    def graph_output_names(self) -> tuple[str, ...]:
        """The graph output names in network-output order.

        The order is the topological leaf order. It is independent of the objective
        target order; use :attr:`objective_output_indices` to align the two.
        """
        return tuple(self.node_by_name[name].output for name in self.graph_output_nodes)

    @property
    def graph_output_nodes(self) -> tuple[str, ...]:
        """The topological leaf node names, in topological order."""
        return self.leaf_nodes

    @property
    def leaf_nodes(self) -> tuple[str, ...]:
        """The node names without children, in topological order."""
        return tuple(
            name for name in self.topological_order if not self.children_of(name)
        )

    @property
    def objective_output_indices(self) -> tuple[int, ...]:
        """Indices selecting objective targets from the graph output vector.

        The network posterior returns graph outputs in :attr:`graph_output_names`
        order. Applying an objective whose targets are declared in a different order
        therefore requires this index mapping.
        """
        positions = {name: index for index, name in enumerate(self.graph_output_names)}
        return tuple(positions[target.name] for target in self.objective.targets)

    @property
    def node_by_name(self) -> dict[str, MultiStepNode]:
        """Lookup from node name to node object."""
        return {node.name: node for node in self.nodes}

    @property
    def parameter_by_name(self) -> dict[str, Parameter]:
        """Lookup from parameter name to parameter object."""
        return {
            parameter.name: parameter
            for node in self.nodes
            for parameter in node.inputs
        }

    @cached_property
    def _parent_map(self) -> dict[str, tuple[str, ...]]:
        """Mapping from child node name to parent node names."""
        return {
            node.name: tuple(parent.name for parent in node.parents)
            for node in self.nodes
            if node.parents
        }

    @cached_property
    def _child_map(self) -> dict[str, tuple[str, ...]]:
        """Mapping from parent node name to child node names."""
        children: dict[str, list[str]] = {}
        for node in self.nodes:
            for parent in node.parents:
                children.setdefault(parent.name, []).append(node.name)
        return {name: tuple(names) for name, names in children.items()}

    @cached_property
    def target_by_output(self) -> dict[str, Target]:
        """Mapping from node output name to the objective target modeling it.

        Only the final nodes are covered. All other node outputs are modeled with
        a plain numerical target of the same name.
        """
        return {target.name: target for target in self.objective.targets}

    @cached_property
    def topological_order(self) -> tuple[str, ...]:
        """All node names in topological order.

        Returns:
            The node names, with every node listed after its parents.

        Raises:
            IncompatibilityError: If node names are duplicated, a parent is
                unknown, or the graph contains a cycle.
        """
        nodes_by_name = self.node_by_name
        if len(nodes_by_name) != len(self.nodes):
            raise IncompatibilityError("The DAG contains duplicate node names.")

        # Register all nodes first to preserve the declared order among roots.
        sorter = TopologicalSorter(dict.fromkeys(nodes_by_name, ()))
        for node in self.nodes:
            for parent in node.parents:
                if parent.name not in nodes_by_name:
                    raise IncompatibilityError(
                        f"Node '{node.name}' references unknown parent '{parent.name}'."
                    )
            sorter.add(node.name, *(parent.name for parent in node.parents))

        try:
            return tuple(sorter.static_order())
        except CycleError as ex:
            raise IncompatibilityError("The DAG contains a cycle.") from ex

    @cached_property
    def full_searchspace(self) -> SearchSpace:
        """Combined search space across all node-owned parameters."""
        return SearchSpace.from_product(
            [parameter for node in self.nodes for parameter in node.inputs]
        )

    def make_model(self) -> MultiStepModel:
        """Build an unfitted multi-step model for this DAG.

        Returns:
            A new model bound to this DAG.
        """
        from .model import MultiStepModel

        return MultiStepModel(dag=self)

    def owned_parameter_names(self, node: str) -> tuple[str, ...]:
        """Return the names of the parameters owned by a node.

        Args:
            node: The node name.

        Returns:
            The node's own parameter names in search-space order.

        Raises:
            IncompatibilityError: If the node is unknown.
        """
        node_object = self.node_by_name.get(node)
        if node_object is None:
            raise IncompatibilityError(f"Unknown node '{node}'.")
        return tuple(parameter.name for parameter in node_object.inputs)

    def parents_of(self, node: str) -> tuple[str, ...]:
        """Return the parent node names of the given node."""
        return self._parent_map.get(node, ())

    def children_of(self, node: str) -> tuple[str, ...]:
        """Return the child node names of the given node."""
        return self._child_map.get(node, ())

    def ancestors_of(self, node: str) -> tuple[str, ...]:
        """All ancestors of a node in topological order, excluding itself."""
        if node not in self.node_by_name:
            raise IncompatibilityError(f"Unknown node '{node}'.")
        result = {node}
        for name in reversed(self.topological_order):
            if name in result:
                result.update(self.parents_of(name))
        result.remove(node)
        return tuple(name for name in self.topological_order if name in result)

    def descendants_of(self, node: str) -> tuple[str, ...]:
        """All descendants of a node in topological order, excluding itself."""
        if node not in self.node_by_name:
            raise IncompatibilityError(f"Unknown node '{node}'.")
        result = {node}
        for name in self.topological_order:
            if name in result:
                result.update(self.children_of(name))
        result.remove(node)
        return tuple(name for name in self.topological_order if name in result)


def _node_name(spec: Mapping[str, Any], *, context: str) -> str:
    """Extract and validate the name of a serialized node definition."""
    name = spec.get("name")
    if not isinstance(name, str) or not name:
        raise IncompatibilityError(
            f"Every {context} must provide a nonempty string 'name'."
        )
    return name


def _check_nested_node_definitions(
    prototypes: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate that nested parent definitions agree with their top-level peers.

    Serialized DAGs repeat the complete definition of a parent inside every child.
    The structure hook only uses the top-level entries to construct canonical
    instances, but it rejects contradictory duplicates instead of silently picking
    one of them.

    Args:
        prototypes: The top-level node specifications, keyed by node name.

    Raises:
        IncompatibilityError: If a nested parent has no top-level definition or
            differs from it.
    """
    for child_name, spec in prototypes.items():
        for parent_spec in spec.get("parents", ()):
            if not isinstance(parent_spec, Mapping):
                raise IncompatibilityError(
                    f"Parents of node '{child_name}' must be node definitions."
                )
            parent_name = _node_name(parent_spec, context="parent definition")
            if parent_name not in prototypes:
                raise IncompatibilityError(
                    f"Node '{child_name}' references parent '{parent_name}' which "
                    f"is not part of the DAG."
                )
            if parent_spec != prototypes[parent_name]:
                raise IncompatibilityError(
                    f"The nested definition of node '{parent_name}' referenced by "
                    f"'{child_name}' contradicts its top-level definition."
                )


def _structure_multistep_dag(
    specs: Mapping[str, Any], cls: type[MultiStepDAG]
) -> MultiStepDAG:
    """Restore a DAG while resolving parent references to canonical instances.

    Args:
        specs: The dictionary representation of the DAG.
        cls: The DAG class to instantiate.

    Returns:
        The restored DAG.

    Raises:
        TypeError: If the serialized type marker names a different class.
        IncompatibilityError: If node names are duplicated, a parent is missing,
            nested definitions contradict their top-level peers, or the graph
            contains a cycle.
    """
    if not isinstance(specs, Mapping):
        raise TypeError(f"Expected a mapping to restore '{cls.__name__}'.")

    # The input is caller-owned and must remain untouched.
    raw_specs: dict[str, Any] = copy.deepcopy(dict(specs))

    type_ = raw_specs.pop("type", None)
    if type_ is not None and not refers_to(cls, type_):
        raise TypeError(
            f"The type field '{type_}' does not match the target class "
            f"'{cls.__name__}'."
        )

    raw_nodes = raw_specs.pop("nodes", None)
    if not isinstance(raw_nodes, (list, tuple)):
        raise IncompatibilityError(
            "A serialized DAG must provide a sequence of node definitions."
        )

    prototypes: dict[str, Mapping[str, Any]] = {}
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise IncompatibilityError("Node definitions must be mappings.")
        name = _node_name(raw, context="node definition")
        if name in prototypes:
            raise IncompatibilityError(
                f"The DAG contains duplicate node names: {name!r}."
            )
        prototypes[name] = raw

    if not prototypes:
        raise IncompatibilityError("A serialized DAG must contain at least one node.")

    # Separate unknown-parent validation from the nested-definition agreement check
    # so that canonical resolution always sees a closed parent graph.
    adjacency: dict[str, tuple[str, ...]] = {}
    for name, raw in prototypes.items():
        parents = raw.get("parents", ())
        if not isinstance(parents, (list, tuple)):
            raise IncompatibilityError(
                f"Parents of node '{name}' must be a sequence of node definitions."
            )
        parent_names: list[str] = []
        for parent in parents:
            if not isinstance(parent, Mapping):
                raise IncompatibilityError(
                    f"Parents of node '{name}' must be node definitions."
                )
            parent_name = _node_name(parent, context="parent definition")
            if parent_name not in prototypes:
                raise IncompatibilityError(
                    f"Node '{name}' references parent '{parent_name}' which is not "
                    f"part of the DAG."
                )
            parent_names.append(parent_name)
        adjacency[name] = tuple(parent_names)

    _check_nested_node_definitions(prototypes)

    sorter = TopologicalSorter(dict.fromkeys(adjacency, ()))
    for name, parents in adjacency.items():
        sorter.add(name, *parents)
    try:
        order = tuple(sorter.static_order())
    except CycleError as ex:
        raise IncompatibilityError("The DAG contains a cycle.") from ex
    base_nodes = {
        name: converter.structure(
            {key: value for key, value in raw.items() if key != "parents"},
            MultiStepNode,
        )
        for name, raw in prototypes.items()
    }
    canonical: dict[str, MultiStepNode] = {}
    for name in order:
        canonical[name] = evolve(
            base_nodes[name],
            parents=tuple(canonical[parent] for parent in adjacency[name]),
        )

    objective = raw_specs.pop("objective", None)
    if objective is None:
        raise IncompatibilityError("A serialized DAG must provide an objective.")

    nodes = tuple(canonical[name] for name in prototypes)
    return cls(
        nodes=nodes,
        objective=converter.structure(objective, Objective),
        **raw_specs,
    )


converter.register_structure_hook(MultiStepDAG, _structure_multistep_dag)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
