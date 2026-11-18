"""Node definitions for multi-step campaigns."""

from __future__ import annotations

import gc
from collections import Counter
from functools import cached_property

from attrs import define, field
from attrs.validators import deep_iterable, gt, instance_of, min_len

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import Parameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float


@define(frozen=True, slots=False)
class MultiStepNode:
    """A single step in a multi-step experiment DAG.

    A node owns its input parameters, the name of its output, its surrogate
    configuration, its execution cost, and references to parent nodes whose
    outputs it consumes. The output itself is an observed quantity produced by
    executing the node, not a search-space dimension. It does not own experiment
    state, fitted models, or status.
    """

    name: str = field(validator=min_len(1))
    """The unique name of this node within the DAG."""

    output: str = field(validator=min_len(1))
    """The name of the single output column this node produces."""

    inputs: tuple[Parameter, ...] = field(
        factory=tuple,
        converter=to_tuple,
        validator=deep_iterable(instance_of(Parameter)),
    )
    """Input parameters not produced by parent nodes."""

    parents: tuple[MultiStepNode, ...] = field(factory=tuple, converter=to_tuple)
    """The parent nodes whose outputs this node consumes as inputs."""

    surrogate_model: SurrogateProtocol = field(
        factory=GaussianProcessSurrogate,
        hash=False,
    )
    """The unfitted surrogate configuration used to model this node.

    The template is never fitted in place; each fit operates on a deep copy.
    Defaults to
    :class:`~baybe.surrogates.gaussian_process.core.GaussianProcessSurrogate`.
    """

    cost: float = field(default=1.0, converter=float, validator=[finite_float, gt(0.0)])
    """The cost of executing this node once.

    The campaign compares candidate actions by their acquisition utility per
    unit cost, so that an expensive node is only executed when its expected
    improvement justifies the expense.
    """

    @inputs.validator
    def _validate_inputs(self, _attribute, value):  # noqa: DOC101, DOC103
        """Validate that input names are unique and differ from the output name."""
        names = Counter(parameter.name for parameter in value)
        duplicates = sorted(name for name, count in names.items() if count > 1)
        if duplicates:
            raise IncompatibilityError(
                f"Node '{self.name}' lists duplicate input parameters: {duplicates}."
            )
        if self.output in names:
            raise IncompatibilityError(
                f"Node '{self.name}' uses output name '{self.output}' also as "
                f"an input parameter."
            )

    @parents.validator
    def _validate_parents(self, _attribute, value):  # noqa: DOC101, DOC103
        """Validate parent types and that references are unique."""
        if not all(isinstance(parent, MultiStepNode) for parent in value):
            raise TypeError(
                f"Node '{self.name}' parents must be {MultiStepNode.__name__} "
                f"instances."
            )
        names = Counter(parent.name for parent in value)
        duplicates = sorted(name for name, count in names.items() if count > 1)
        if duplicates:
            raise IncompatibilityError(
                f"Node '{self.name}' lists duplicate parents: {duplicates}."
            )
        if self.name in names:
            raise IncompatibilityError(f"Node '{self.name}' cannot be its own parent.")

    @property
    def _own_comp_rep_columns(self) -> tuple[str, ...]:
        """The computational-representation columns of the node's own inputs.

        Parent-only nodes have no own search-space dimensions, in which case the
        empty tuple is returned without constructing an empty search space.
        """
        if not self.inputs:
            return ()
        return self.searchspace.comp_rep_columns

    @property
    def parent_output_names(self) -> tuple[str, ...]:
        """The output names of the node's parent nodes, in parent order."""
        return tuple(parent.output for parent in self.parents)

    @property
    def model_input_names(self) -> tuple[str, ...]:
        """The experimental-representation columns used to fit the node model."""
        return (
            *(parameter.name for parameter in self.inputs),
            *self.parent_output_names,
        )

    @cached_property
    def searchspace(self) -> SearchSpace:
        """The search space of the node's own tunable input parameters."""
        return SearchSpace.from_product(self.inputs)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
