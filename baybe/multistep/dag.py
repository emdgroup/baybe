"""DAG topology for multi-step campaigns."""

from __future__ import annotations

import gc
from collections import Counter
from functools import cached_property
from graphlib import CycleError, TopologicalSorter
from math import prod
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.validators import and_, deep_iterable, instance_of, min_len

from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.searchspace.core import SearchSpace
from baybe.targets.base import Target
from baybe.utils.basic import to_tuple

from ._columns import RESERVED_COLUMNS
from .node import MultiStepNode

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter

    from .model import MultiStepModel


@define(frozen=True, slots=False)
class MultiStepDAG:
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
    """The optimization objective, whose targets name the final nodes."""

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
        """Validate the objective against the node outputs and targets."""
        by_output = {node.output: node for node in self.nodes}

        target_names = [target.name for target in value.targets]
        duplicates = sorted(
            name for name, count in Counter(target_names).items() if count > 1
        )
        if duplicates:
            raise IncompatibilityError(
                f"Objective target names must be unique, but found duplicates: "
                f"{duplicates}."
            )

        unknown = sorted(set(target_names) - set(by_output))
        if unknown:
            raise IncompatibilityError(
                f"Objective targets {unknown} do not match any node output. Known "
                f"outputs: {sorted(by_output)}."
            )

        final_names = {by_output[target.name].name for target in value.targets}
        relevant = set(final_names)
        for name in final_names:
            relevant.update(self.ancestors_of(name))
        irrelevant = sorted({node.name for node in self.nodes} - relevant)
        if irrelevant:
            raise IncompatibilityError(
                f"Nodes {irrelevant} are not final nodes or ancestors of final "
                f"nodes and would not influence the objective."
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

    @cached_property
    def final_nodes(self) -> tuple[str, ...]:
        """The nodes referenced by the objective targets, in objective order."""
        by_output = {node.output: node.name for node in self.nodes}
        return tuple(by_output[target.name] for target in self.objective.targets)

    @property
    def final_output_dim(self) -> int:
        """Total number of outputs produced by all final nodes."""
        return len(self.final_nodes)

    @property
    def final_target_names(self) -> tuple[str, ...]:
        """Target names of all final nodes, in objective order."""
        return tuple(target.name for target in self.objective.targets)

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


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
