"""Configuration of the historical features supplied to node models."""

from __future__ import annotations

import gc
from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING

from attrs import define, evolve, field

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.serialization import SerialMixin

if TYPE_CHECKING:
    from .node import MultiStepNode


class ContextMode(Enum):
    """Select which predecessor features accompany a node's own inputs."""

    PARENT_OUTPUTS = "parent_outputs"
    PARENT_INPUTS_OUTPUTS = "parent_inputs_outputs"
    ANCESTOR_OUTPUTS = "ancestor_outputs"
    ANCESTOR_INPUTS_OUTPUTS = "ancestor_inputs_outputs"


@define(frozen=True)
class _HistoryFeature:
    """A feature sourced from one specific predecessor path."""

    path: tuple[str, ...]
    """Predecessor names from the direct parent to the feature's owner."""

    source: str
    """Original parameter or output column."""

    name: str
    """Column name in the receiving node's local model."""

    parameter: Parameter | None = None
    """Original parameter, or ``None`` for a node output."""

    def make_parameter(self) -> Parameter:
        """Create a locally named parameter preserving its original encoding.

        Returns:
            The original parameter or a renamed copy with identical encodings.

        Raises:
            IncompatibilityError: If this feature is not a parameter or its renamed
                encoding is incompatible with the original.
        """
        if self.parameter is None:
            raise IncompatibilityError("An output feature is not a parameter.")
        if self.name == self.parameter.name:
            return self.parameter
        renamed = evolve(self.parameter, name=self.name)
        if isinstance(self.parameter, DiscreteParameter) and isinstance(
            renamed, DiscreteParameter
        ):
            original = self.parameter.comp_df
            columns: list[str] = []
            for column in original.columns:
                if column == self.parameter.name:
                    columns.append(self.name)
                elif isinstance(column, str) and column.startswith(
                    f"{self.parameter.name}_"
                ):
                    columns.append(f"{self.name}{column[len(self.parameter.name) :]}")
                else:
                    raise IncompatibilityError(
                        f"The encoding column {column!r} of parameter "
                        f"'{self.parameter.name}' cannot be renamed unambiguously."
                    )
            # Some encodings perturb degenerate rows randomly. Reusing the original
            # values ensures sampling and the locally fitted model agree exactly.
            object.__setattr__(renamed, "comp_df", original.set_axis(columns, axis=1))
        return renamed


@define(frozen=True)
class MultiStepContext(SerialMixin):
    """Define a DAG-wide policy for assembling node model inputs.

    Every node always receives its own inputs. If a predecessor is reachable by
    multiple paths, its features are represented separately for each path, since
    functional joins may combine different real executions. Direct parent outputs
    retain their original names. Other repeated features use a path-qualified name.
    For a path ``B -> A`` the output ``a`` is named ``__history__1_B__1_A__a``;
    the numbers are the lengths of the node names. Included predecessor inputs
    are their own tunable parameters, not their entire local model inputs.

    For example, configure a DAG with
    ``context=MultiStepContext(ContextMode.ANCESTOR_INPUTS_OUTPUTS)``.
    Existing DAGs default to :attr:`ContextMode.PARENT_OUTPUTS`.
    """

    mode: ContextMode = field(default=ContextMode.PARENT_OUTPUTS, converter=ContextMode)
    """The predecessor features included in every node's model."""

    def _features(self, node: MultiStepNode) -> tuple[_HistoryFeature, ...]:
        """Describe historical features in deterministic parent-path order."""
        include_inputs = self.mode in (
            ContextMode.PARENT_INPUTS_OUTPUTS,
            ContextMode.ANCESTOR_INPUTS_OUTPUTS,
        )
        include_ancestors = self.mode in (
            ContextMode.ANCESTOR_OUTPUTS,
            ContextMode.ANCESTOR_INPUTS_OUTPUTS,
        )
        paths: list[tuple[tuple[str, ...], MultiStepNode]] = []

        def visit(predecessor: MultiStepNode, path: tuple[str, ...]) -> None:
            """Collect paths, not just nodes, to preserve independent lineages."""
            paths.append((path, predecessor))
            if include_ancestors:
                for parent in predecessor.parents:
                    visit(parent, (*path, parent.name))

        for parent in node.parents:
            visit(parent, (parent.name,))
        counts = Counter(predecessor.name for _, predecessor in paths)
        result: list[_HistoryFeature] = []
        for path, predecessor in paths:
            items: list[tuple[str, Parameter | None]] = []
            if include_inputs:
                items.extend(
                    (parameter.name, parameter) for parameter in predecessor.inputs
                )
            items.extend((source, None) for source in predecessor.output_names)
            for source, parameter in items:
                direct_output = len(path) == 1 and parameter is None
                if counts[predecessor.name] > 1 and not direct_output:
                    prefix = "__".join(f"{len(part)}_{part}" for part in path)
                    name = f"__history__{prefix}__{source}"
                else:
                    name = source
                result.append(_HistoryFeature(path, source, name, parameter))
        return tuple(result)

    def input_names(self, node: MultiStepNode) -> tuple[str, ...]:
        """Return the experimental input columns for a node model.

        Args:
            node: The receiving node.

        Returns:
            Own parameter names followed by path-specific history columns.

        Raises:
            IncompatibilityError: If generated history names collide.
        """
        names = tuple(parameter.name for parameter in node.inputs) + tuple(
            feature.name for feature in self._features(node)
        )
        duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicates:
            raise IncompatibilityError(
                f"History feature names for node '{node.name}' collide: {duplicates}."
            )
        return names

    def _parameters(self, node: MultiStepNode) -> tuple[Parameter, ...]:
        """Return own and historical parameter definitions with local names."""
        return (
            *node.inputs,
            *(
                feature.make_parameter()
                for feature in self._features(node)
                if feature.parameter is not None
            ),
        )


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
