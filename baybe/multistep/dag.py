"""DAG topology for multi-step campaigns."""

from __future__ import annotations

import gc
from collections import deque
from functools import cached_property
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


@define(frozen=True)
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

        outputs = [node.output for node in self.nodes]
        duplicates = sorted({name for name in outputs if outputs.count(name) > 1})
        if duplicates:
            raise IncompatibilityError(
                f"Node output names must be unique, but found duplicates: {duplicates}."
            )

        parameter_names = [
            parameter.name
            for node in self.nodes
            for parameter in node.searchspace.parameters
        ]
        duplicates = sorted(
            {name for name in parameter_names if parameter_names.count(name) > 1}
        )
        if duplicates:
            raise IncompatibilityError(
                f"Parameter names must be unique across nodes, but found "
                f"duplicates: {duplicates}."
            )

        comp_rep_columns = [
            column
            for node in self.nodes
            for column in node.searchspace.comp_rep_columns
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
            {name for name in target_names if target_names.count(name) > 1}
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
        searchspace = node.searchspace
        if searchspace.is_constrained:
            raise IncompatibilityError(
                f"Node '{node.name}' has a constrained search space, which is "
                f"not supported by multi-step planning."
            )
        if searchspace.discrete.is_empty:
            return
        n_expected = 1
        for parameter in searchspace.discrete.parameters:
            n_expected *= len(parameter.values)
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
            for parameter in node.searchspace.parameters
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
        """All node names in topological order."""
        names = [node.name for node in self.nodes]
        if len(set(names)) != len(names):
            raise IncompatibilityError("The DAG contains duplicate node names.")

        in_degree = {node.name: len(node.parents) for node in self.nodes}
        children: dict[str, list[str]] = {name: [] for name in names}
        for node in self.nodes:
            for parent in node.parents:
                if parent.name not in children:
                    raise IncompatibilityError(
                        f"Node '{node.name}' references unknown parent '{parent.name}'."
                    )
                children[parent.name].append(node.name)

        queue = deque(name for name in names if in_degree[name] == 0)
        order: list[str] = []
        while queue:
            name = queue.popleft()
            order.append(name)
            for child in children[name]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise IncompatibilityError(
                f"The DAG contains a cycle: only {len(order)} of {len(self.nodes)} "
                f"nodes could be sorted topologically."
            )
        return tuple(order)

    @cached_property
    def full_searchspace(self) -> SearchSpace:
        """Combined search space across all node-owned parameters."""
        parameters: list[Parameter] = []
        for node in self.nodes:
            parameters.extend(node.searchspace.parameters)
        return SearchSpace.from_product(parameters)

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
        return tuple(parameter.name for parameter in node_object.searchspace.parameters)

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
        result: set[str] = set()

        def visit(current: str) -> None:
            for parent in self.parents_of(current):
                if parent not in result:
                    result.add(parent)
                    visit(parent)

        visit(node)
        return tuple(name for name in self.topological_order if name in result)

    def descendants_of(self, node: str) -> tuple[str, ...]:
        """All descendants of a node in topological order, excluding itself."""
        if node not in self.node_by_name:
            raise IncompatibilityError(f"Unknown node '{node}'.")
        result: set[str] = set()

        def visit(current: str) -> None:
            for child in self.children_of(current):
                if child not in result:
                    result.add(child)
                    visit(child)

        visit(node)
        return tuple(name for name in self.topological_order if name in result)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
