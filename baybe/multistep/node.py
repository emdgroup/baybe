"""Node definitions for multi-step campaigns."""

from __future__ import annotations

import gc
from collections import Counter
from functools import cached_property
from typing import Literal

from attrs import define, field
from attrs.validators import (
    and_,
    deep_iterable,
    gt,
    in_,
    instance_of,
    min_len,
    optional,
)

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import Parameter
from baybe.recommenders.pure.experimental_feasibility import (
    InfeasibleMeasurementMode,
)
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float


@define(frozen=True)
class FeasibilityCondition:
    """Require one measured value to pass a numerical or Boolean condition."""

    column: str = field(validator=and_(instance_of(str), min_len(1)))
    """The output or auxiliary measurement column to check."""

    operator: Literal["<=", ">=", "is_true"] = field(
        validator=[instance_of(str), in_(("<=", ">=", "is_true"))]
    )
    """The comparison to apply, or ``is_true`` for a Boolean success flag."""

    threshold: float | None = field(
        default=None,
        converter=lambda value: None if value is None else float(value),
        validator=optional(finite_float),
    )
    """The numerical bound, required for ``<=`` and ``>=`` comparisons."""

    @threshold.validator
    def _validate_threshold(self, attribute, value) -> None:
        if (self.operator == "is_true") != (value is None):
            raise ValueError(
                "A feasibility condition requires a threshold for '<=' or '>=', "
                "and no threshold for 'is_true'."
            )


@define(frozen=True, slots=False)
class MultiStepNode:
    """A single step in a multi-step experiment DAG.

    A node owns its input parameters, the name of its output, its surrogate
    configuration, its execution cost, and references to parent nodes whose
    outputs it consumes. The output itself is an observed quantity produced by
    executing the node, not a search-space dimension. It does not own experiment
    state, fitted models, or status.
    """

    name: str = field(validator=and_(instance_of(str), min_len(1)))
    """The unique name of this node within the DAG."""

    output: str = field(validator=and_(instance_of(str), min_len(1)))
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

    The campaign accounts for this cost when computing the total remaining cost
    of an action's continuation. A failed measurement still incurs the cost.
    """

    feasibility_conditions: tuple[FeasibilityCondition, ...] = field(
        factory=tuple,
        converter=to_tuple,
        validator=deep_iterable(instance_of(FeasibilityCondition)),
        kw_only=True,
    )
    """Conditions that define successful execution at this node.

    All conditions must pass. They may check the node's output or auxiliary measured
    columns, and are modeled jointly as the probability of successful execution.
    """

    infeasible_measurement_mode: InfeasibleMeasurementMode = field(
        default=InfeasibleMeasurementMode.OBSERVED,
        converter=InfeasibleMeasurementMode,
        kw_only=True,
    )
    """How failed observations enter the output-regression training view.

    The stored real observations and the independent feasibility labels are never
    changed by this setting. ``OBSERVED`` keeps a real failure output when one was
    reported, ``EXCLUDE`` removes failures, ``DUMMY`` replaces every failure output
    with :attr:`infeasible_dummy_value`, and ``PREDICTED_MEAN`` replaces failures
    with predictions of a model trained on successful observations only.
    """

    infeasible_dummy_value: float | None = field(
        default=None,
        converter=lambda value: None if value is None else float(value),
        validator=optional(finite_float),
        kw_only=True,
    )
    """The replacement output for failed observations in ``DUMMY`` mode."""

    @infeasible_dummy_value.validator
    def _validate_infeasible_dummy_value(self, _attribute, value):  # noqa: DOC101, DOC103
        """Require a finite dummy value whenever ``DUMMY`` mode is selected."""
        if (
            self.infeasible_measurement_mode is InfeasibleMeasurementMode.DUMMY
            and value is None
        ):
            raise IncompatibilityError(
                f"Node '{self.name}' uses "
                f"'{InfeasibleMeasurementMode.DUMMY.name}' but does not provide "
                f"a finite 'infeasible_dummy_value'."
            )

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
        """Local inputs under the default direct-parent-output context.

        For a configured DAG, use ``dag.context.input_names(node)`` instead.
        """
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
