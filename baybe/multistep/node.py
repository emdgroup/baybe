"""Node definitions for multi-step campaigns."""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Mapping
from functools import cached_property
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, Literal

import cattrs
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
from baybe.serialization import converter
from baybe.surrogates.base import SurrogateProtocol
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float

from ._configuration_space import _ConfigurationSpace


def _normalize_output(
    value: str | tuple[str, ...] | list[str],
) -> str | tuple[str, ...]:
    """Normalize an output declaration while preserving the legacy string form."""
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(value)
    raise TypeError("A node output must be a string or a sequence of strings.")


def _validate_output(
    _instance: Any, _attribute: Any, value: str | tuple[str, ...]
) -> None:
    """Validate output names and reject ambiguous or duplicate declarations."""
    names = (value,) if isinstance(value, str) else value
    if not names:
        raise ValueError("A node must declare at least one output name.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Node output names must be nonempty strings.")
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise IncompatibilityError(f"Node output names must be unique: {duplicates}.")


def _normalize_dummy(
    value: float | Mapping[str, float] | None,
) -> float | Mapping[str, float] | None:
    """Copy and normalize dummy values without accepting non-finite numbers."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for name, number in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Dummy output names must be nonempty strings.")
            result[name] = _finite_dummy(number)
        return MappingProxyType(result)
    return _finite_dummy(value)


def _finite_dummy(value: Any) -> float:
    """Return a finite real number, rejecting Booleans and nonnumeric values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Dummy output values must be finite real numbers.")
    number = float(value)
    if not isfinite(number):
        raise ValueError("Dummy output values must be finite real numbers.")
    return number


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

    A node owns its input parameters, one or more named outputs, its surrogate
    configuration, its execution cost, and references to parent nodes whose
    outputs it consumes. Its outputs are observed together as one execution, not
    search-space dimensions. It does not own experiment state, fitted models, or
    status.
    """

    name: str = field(validator=and_(instance_of(str), min_len(1)))
    """The unique name of this node within the DAG."""

    output: str | tuple[str, ...] = field(
        converter=_normalize_output, validator=_validate_output
    )
    """The name of the output column, or a tuple of jointly produced columns."""

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

    All conditions must pass. Every configured node uses one shared classifier for
    this joint success event; numeric-output posterior probabilities are features of
    that classifier, while auxiliary values contribute to its training labels.
    """

    infeasible_measurement_mode: InfeasibleMeasurementMode = field(
        default=InfeasibleMeasurementMode.OBSERVED,
        converter=InfeasibleMeasurementMode,
        kw_only=True,
    )
    """How failed observations enter the output-regression training view.

    The stored real observations and raw joint feasibility labels are never changed by
    this setting. ``OBSERVED`` keeps a real failure output when one was
    reported, ``EXCLUDE`` removes failures, ``DUMMY`` replaces every failure output
    with :attr:`infeasible_dummy_value`, and ``PREDICTED_MEAN`` replaces failures
    with predictions of a model trained on successful observations only.
    """

    infeasible_dummy_value: float | Mapping[str, float] | None = field(
        default=None,
        converter=_normalize_dummy,
        kw_only=True,
    )
    """The replacement output value(s) for failed observations in ``DUMMY`` mode."""

    @infeasible_dummy_value.validator
    def _validate_infeasible_dummy_value(self, _attribute, value):  # noqa: DOC101, DOC103
        """Require a finite dummy value whenever ``DUMMY`` mode is selected."""
        if self.infeasible_measurement_mode is InfeasibleMeasurementMode.DUMMY and (
            value is None
            or (isinstance(value, Mapping) and set(value) != set(self.output_names))
            or (len(self.output_names) > 1 and not isinstance(value, Mapping))
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
        collisions = set(self.output_names) & set(names)
        if collisions:
            output_name = sorted(collisions)[0]
            raise IncompatibilityError(
                f"Node '{self.name}' uses output name '{output_name}' also as "
                f"an input parameter."
            )

        if isinstance(self.infeasible_dummy_value, Mapping):
            unknown = set(self.infeasible_dummy_value) - set(self.output_names)
            if unknown:
                raise IncompatibilityError(
                    f"Node '{self.name}' has dummy values for unknown outputs: "
                    f"{sorted(unknown)}."
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

    @cached_property
    def _own_parameter_layout(self) -> _ConfigurationSpace:
        """The product-free layout of the node's own input parameters."""
        return _ConfigurationSpace.from_parameters(self.inputs)

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
        return tuple(name for parent in self.parents for name in parent.output_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Return this node's output names in their canonical order."""
        return (self.output,) if isinstance(self.output, str) else self.output

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


def _structure_output(value: Any, _: Any) -> str | tuple[str, ...]:
    """Structure legacy strings and new output-name sequences."""
    return _normalize_output(value)


def _structure_dummy(value: Any, _: Any) -> Any:
    """Pass copied dummy values through the node field converter."""
    return value


_node_unstructure_hook = cattrs.gen.make_dict_unstructure_fn(
    MultiStepNode,
    converter,
    infeasible_dummy_value=cattrs.gen.override(
        unstruct_hook=lambda value: dict(value) if isinstance(value, Mapping) else value
    ),
)
_node_structure_hook = cattrs.gen.make_dict_structure_fn(
    MultiStepNode,
    converter,
    output=cattrs.gen.override(struct_hook=_structure_output),
    infeasible_dummy_value=cattrs.gen.override(struct_hook=_structure_dummy),
)


def _structure_multistep_node(specs: Mapping[str, Any], cls: type[MultiStepNode]):
    """Restore a node while preserving field-level output normalization."""
    values = dict(specs)
    type_ = values.pop("type", None)
    if type_ is not None and type_ != cls.__name__:
        raise TypeError(
            f"The type field '{type_}' does not match the target class "
            f"'{cls.__name__}'."
        )
    return _node_structure_hook(values, cls)


converter.register_unstructure_hook(MultiStepNode, _node_unstructure_hook)
converter.register_structure_hook(MultiStepNode, _structure_multistep_node)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
