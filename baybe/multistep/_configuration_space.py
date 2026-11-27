"""Factorized parameter layout for multi-step configurations and model fitting."""

from __future__ import annotations

import gc
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, instance_of

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import DiscreteParameter, Parameter
from baybe.parameters.categorical import CategoricalParameter, TaskParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.utils import sort_parameters
from baybe.utils.dataframe import get_transform_objects

if TYPE_CHECKING:
    from baybe.multistep.dag import MultiStepDAG
    from baybe.parameters.selectors import ParameterSelectorProtocol


def _ordered_parameters(parameters: Sequence[Parameter]) -> tuple[Parameter, ...]:
    """Return parameters in search-space order: discrete first, then continuous."""
    return (
        *sort_parameters(p for p in parameters if p.is_discrete),
        *sort_parameters(p for p in parameters if p.is_continuous),
    )


def _validate_supported_parameter(parameter: Parameter) -> None:
    """Reject parameter types and encodings outside the product-free contract.

    Raises:
        IncompatibilityError: If the parameter type or encoding is unsupported.
    """
    if isinstance(parameter, TaskParameter):
        raise IncompatibilityError(
            f"Parameter '{parameter.name}' is a '{TaskParameter.__name__}', which is "
            f"not supported by the product-free input layout."
        )
    if isinstance(
        parameter, (NumericalContinuousParameter, NumericalDiscreteParameter)
    ):
        return
    if isinstance(parameter, CategoricalParameter):
        if parameter.encoding not in (
            CategoricalEncoding.OHE,
            CategoricalEncoding.INT,
        ):
            raise IncompatibilityError(
                f"Parameter '{parameter.name}' uses the unsupported categorical "
                f"encoding '{parameter.encoding}'."
            )
        return
    raise IncompatibilityError(
        f"Parameter '{parameter.name}' of type '{type(parameter).__name__}' is not "
        f"supported by the product-free input layout. Supported types are "
        f"'{NumericalContinuousParameter.__name__}', "
        f"'{NumericalDiscreteParameter.__name__}', and "
        f"'{CategoricalParameter.__name__}' with OHE or INT encoding."
    )


def _validate_active_levels(parameter: Parameter) -> None:
    """Check that every declared active level has a complete encoding row.

    Raises:
        IncompatibilityError: If the parameter declares no active levels or an
            active level is missing from the parameter's computational
            representation.
    """
    if not isinstance(parameter, DiscreteParameter):
        return

    active_values = parameter.active_values
    if not active_values:
        raise IncompatibilityError(
            f"Discrete parameter '{parameter.name}' must declare at least one active "
            f"level."
        )

    encoded_levels = set(parameter.comp_df.index)
    unknown = [value for value in active_values if value not in encoded_levels]
    if unknown:
        raise IncompatibilityError(
            f"Parameter '{parameter.name}' declares active levels without a "
            f"computational-representation row: {unknown}."
        )


def _validate_parameter_definitions(parameters: Sequence[Parameter]) -> None:
    """Validate names, levels, encodings, and generated columns parameter-wise.

    Raises:
        IncompatibilityError: If names or generated comp-rep columns are not
            unique, or a parameter is unsupported.
    """
    duplicate_names = sorted(
        name for name, count in Counter(p.name for p in parameters).items() if count > 1
    )
    if duplicate_names:
        raise IncompatibilityError(
            f"Parameter names must be unique in the product-free input layout, but "
            f"found duplicates: {duplicate_names}."
        )

    for parameter in parameters:
        _validate_supported_parameter(parameter)
        _validate_active_levels(parameter)

    columns = [
        column for parameter in parameters for column in parameter.comp_rep_columns
    ]
    duplicate_columns = sorted(
        column for column, count in Counter(columns).items() if count > 1
    )
    if duplicate_columns:
        raise IncompatibilityError(
            f"Computational-representation columns of the product-free input layout "
            f"collide: {duplicate_columns}."
        )


@define(frozen=True, slots=False)
class _DiscreteParameterView:
    """Read-only parameter view used by factories that inspect discrete inputs.

    This is deliberately not a :class:`~baybe.searchspace.discrete.SubspaceDiscrete`:
    it exposes the parameter definitions already owned by the layout and never a
    candidate table or product.
    """

    parameters: tuple[DiscreteParameter, ...] = field(converter=tuple)
    """The discrete parameter definitions of the underlying layout."""


@define(frozen=True, slots=False)
class _ConfigurationSpace:
    """Describe configuration columns without enumerating their global product.

    The layout serves both complete-configuration workflows and the product-free
    surrogate fit. It uses the parameter definitions and their complete
    computational-representation data frames as the single source for names,
    ordering, columns, scaling bounds, and transformations.
    """

    parameters: tuple[Parameter, ...] = field(
        converter=tuple, validator=deep_iterable(instance_of(Parameter))
    )
    """Parameters in constructor order."""

    parameter_names: tuple[str, ...] = field(init=False)
    """Experimental parameter names in constructor order."""

    comp_rep_columns: tuple[str, ...] = field(init=False)
    """Computational representation column names."""

    _comp_parameters: tuple[Parameter, ...] = field(init=False, repr=False)
    """Parameters in computational representation order."""

    discrete_dims: Mapping[int, tuple[float, ...]] = field(init=False)
    """Numerical discrete values keyed by their computational dimension."""

    _bounds: pd.DataFrame = field(init=False, repr=False, eq=False)
    """The private bounds frame returned defensively by :attr:`bounds`."""

    def __attrs_post_init__(self) -> None:
        """Derive immutable layout metadata from the owned parameters."""
        comp_parameters = _ordered_parameters(self.parameters)
        discrete = tuple(p for p in comp_parameters if p.is_discrete)
        continuous = tuple(p for p in comp_parameters if p.is_continuous)
        object.__setattr__(self, "_comp_parameters", comp_parameters)
        object.__setattr__(
            self, "parameter_names", tuple(p.name for p in self.parameters)
        )

        columns = tuple(
            column for p in comp_parameters for column in p.comp_rep_columns
        )
        object.__setattr__(self, "comp_rep_columns", columns)

        lower: list[float] = []
        upper: list[float] = []
        dims: dict[int, tuple[float, ...]] = {}
        column_position = 0
        for parameter in discrete:
            comp_values = parameter.comp_df
            lower.extend(comp_values.min(axis=0).astype(float).tolist())
            upper.extend(comp_values.max(axis=0).astype(float).tolist())
            if isinstance(parameter, NumericalDiscreteParameter):
                dims[column_position] = tuple(
                    float(value) for value in parameter.values
                )
            column_position += len(parameter.comp_rep_columns)
        for parameter in continuous:
            bounds = getattr(parameter, "bounds", None)
            if bounds is None:
                raise IncompatibilityError(
                    f"Continuous parameter '{parameter.name}' has no numerical bounds."
                )
            lo, hi = bounds.to_tuple()
            lower.append(float(lo))
            upper.append(float(hi))

        object.__setattr__(self, "discrete_dims", MappingProxyType(dims))
        object.__setattr__(
            self,
            "_bounds",
            pd.DataFrame([lower, upper], index=["min", "max"], columns=columns),
        )

    @classmethod
    def from_dag(cls, dag: MultiStepDAG) -> _ConfigurationSpace:
        """Create a factorized layout from every DAG-owned parameter.

        This permissive constructor also covers parameter types that the
        product-free surrogate fit does not support. Use :meth:`from_parameters`
        for fit contexts.

        Args:
            dag: The multi-step DAG supplying node-owned parameters.

        Returns:
            A factorized layout in the established BayBE computational order.
        """
        return cls(parameters=tuple(dag.parameter_by_name.values()))

    @classmethod
    def from_parameters(cls, parameters: Sequence[Parameter]) -> _ConfigurationSpace:
        """Create a strict fit-context layout from parameter definitions.

        Unlike :meth:`from_dag`, this constructor enforces the product-free fit
        contract: only supported parameter types and encodings are accepted, every
        discrete parameter needs an active level, and parameter names and generated
        comp-rep columns must be unique.

        Args:
            parameters: The parameters spanning the layout.

        Returns:
            The immutable fit-context layout.
        """
        parameters = tuple(parameters)
        _validate_parameter_definitions(parameters)
        return cls(parameters=_ordered_parameters(parameters))

    @property
    def _continuous_parameters(self) -> tuple[NumericalContinuousParameter, ...]:
        """The continuous parameters in computational representation order."""
        return tuple(
            cast(NumericalContinuousParameter, parameter)
            for parameter in self._comp_parameters
            if parameter.is_continuous
        )

    @property
    def _discrete_parameters(self) -> tuple[DiscreteParameter, ...]:
        """The discrete parameters in computational representation order."""
        return tuple(
            cast(DiscreteParameter, parameter)
            for parameter in self._comp_parameters
            if parameter.is_discrete
        )

    @property
    def _task_parameter(self) -> TaskParameter | None:
        """The task parameter of the layout, if it has one."""
        parameters = [
            parameter
            for parameter in self.parameters
            if isinstance(parameter, TaskParameter)
        ]
        if not parameters:
            return None
        if len(parameters) > 1:
            raise IncompatibilityError(
                "The input layout must not contain more than one task parameter."
            )
        return parameters[0]

    def active_levels(self, name: str) -> tuple[Any, ...]:
        """Return the active levels of a discrete parameter.

        Args:
            name: The parameter name.

        Returns:
            The active levels in the parameter's stable level order.

        Raises:
            IncompatibilityError: If the parameter is unknown or not discrete.
        """
        try:
            parameter = next(
                parameter for parameter in self.parameters if parameter.name == name
            )
        except StopIteration as ex:
            raise IncompatibilityError(f"Unknown parameter '{name}'.") from ex

        if not isinstance(parameter, DiscreteParameter):
            raise IncompatibilityError(
                f"Parameter '{name}' is not a discrete parameter and has no levels."
            )
        return tuple(parameter.active_values)

    @property
    def bounds(self) -> pd.DataFrame:
        """Return computational-representation bounds as a defensive copy."""
        return self._bounds.copy(deep=True)

    @property
    def discrete(self) -> _DiscreteParameterView:
        """The discrete-parameter view of the layout."""
        return _DiscreteParameterView(self._discrete_parameters)

    @property
    def n_tasks(self) -> int:
        """The number of tasks encoded in the layout."""
        if (task_param := self._task_parameter) is None:
            return 1
        return len(task_param.values)

    @property
    def scaling_bounds(self) -> pd.DataFrame:
        """The bounds used for scaling the surrogate model input."""
        return self.bounds

    @property
    def task_idx(self) -> int | None:
        """The computational column index of the task parameter, if available."""
        if (task_param := self._task_parameter) is None:
            return None
        return self.comp_rep_columns.index(task_param.name)

    def _drop_parameters(self, names: Collection[str], /) -> _ConfigurationSpace:
        """Return a layout without the named parameters.

        Args:
            names: The names of the parameters to remove.

        Raises:
            ValueError: If any name does not match a parameter in the layout.

        Returns:
            A new layout containing only the remaining parameter definitions.
        """
        current_names = set(self.parameter_names)
        names_set = set(names)
        if unknown := names_set - current_names:
            raise ValueError(
                f"Parameter name(s) {sorted(unknown)} not found in the input layout. "
                f"Available: {sorted(current_names)}."
            )
        return _ConfigurationSpace(
            tuple(
                parameter
                for parameter in self.parameters
                if parameter.name not in names_set
            )
        )

    def get_comp_rep_parameter_indices(
        self,
        name_or_selector: str | ParameterSelectorProtocol,
        /,
    ) -> tuple[int, ...]:
        """Find comp-rep column indices for a parameter selection.

        Unknown parameter names and selectors without matches yield an empty tuple,
        matching :class:`~baybe.searchspace.core.SearchSpace`.

        Args:
            name_or_selector: Either the name of a single parameter or a selector
                that filters parameters to be included.

        Returns:
            The integer indices of the selected comp-rep columns.
        """
        if isinstance(name_or_selector, str):
            parameters = [
                parameter
                for parameter in self.parameters
                if parameter.name == name_or_selector
            ]
        else:
            parameters = [
                parameter
                for parameter in self.parameters
                if name_or_selector(parameter)
            ]

        columns = self.comp_rep_columns
        return tuple(
            index
            for parameter in parameters
            for index, column in enumerate(columns)
            if column in parameter.comp_rep_columns
        )

    def _get_n_comp_rep_columns(
        self,
        name_or_selector: str | ParameterSelectorProtocol,
        /,
    ) -> int:
        """Get the number of comp-rep columns for a parameter selection.

        Args:
            name_or_selector: Either the name of a single parameter or a selector
                that filters parameters to be included.

        Returns:
            The number of selected comp-rep columns.
        """
        return len(self.get_comp_rep_parameter_indices(name_or_selector))

    def get_parameters_by_name(self, names: Sequence[str]) -> tuple[Parameter, ...]:
        """Return parameters with the specified names.

        Unknown names are ignored, matching
        :class:`~baybe.searchspace.core.SearchSpace`.

        Args:
            names: Sequence of parameter names.

        Returns:
            The named parameters in computational representation order.
        """
        return tuple(
            parameter for parameter in self._comp_parameters if parameter.name in names
        )

    def sample(self, *, sampling_seed: int) -> pd.DataFrame:
        """Draw one independent uniform sample per parameter.

        Args:
            sampling_seed: Non-negative integer seed for the local random generator.

        Returns:
            One experimental-representation configuration row.

        Raises:
            TypeError: If the seed is not an integer.
            ValueError: If the seed is negative.
        """
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
            raise TypeError("sampling_seed must be a non-negative integer.")
        if sampling_seed < 0:
            raise ValueError("sampling_seed must be a non-negative integer.")

        rng = np.random.default_rng(sampling_seed)
        values: dict[str, Any] = {}
        for parameter in self.parameters:
            if parameter.is_discrete:
                allowed = parameter.values
                values[parameter.name] = allowed[int(rng.integers(len(allowed)))]
            else:
                assert isinstance(parameter, NumericalContinuousParameter)
                lower, upper = parameter.bounds.to_tuple()
                values[parameter.name] = float(rng.uniform(lower, upper))
        return pd.DataFrame([values], columns=self.parameter_names)

    def transform(
        self,
        frame: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        """Transform parameters from experimental to computational representation.

        Args:
            frame: The dataframe to be transformed.
            allow_missing: If ``False``, each parameter of the layout must have
                exactly one corresponding column in the given dataframe. If ``True``,
                the dataframe may contain only a subset of parameter columns.
            allow_extra: If ``False``, each column present in the dataframe must
                correspond to exactly one parameter of the layout. If ``True``, the
                dataframe may contain additional non-parameter-related columns,
                which will be ignored.

        Returns:
            A corresponding dataframe with parameters in computational
            representation.

        Raises:
            IncompatibilityError: If parameter columns or values are invalid.
            TypeError: If ``frame`` is not a dataframe.
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Configurations to transform must be a pandas DataFrame.")
        if not frame.columns.is_unique:
            raise IncompatibilityError(
                "Configuration frame parameter columns must be unique."
            )
        if not (allow_missing or allow_extra):
            return self._transform_configuration(frame)

        discrete_names = [parameter.name for parameter in self._discrete_parameters]
        continuous_names = [parameter.name for parameter in self._continuous_parameters]
        discrete_input = frame.drop(columns=continuous_names, errors="ignore")
        continuous_input = frame.drop(columns=discrete_names, errors="ignore")
        discrete = self._transform_discrete(
            discrete_input, allow_missing=allow_missing, allow_extra=allow_extra
        )
        continuous = self._transform_continuous(
            continuous_input, allow_missing=allow_missing, allow_extra=allow_extra
        )
        return pd.concat([discrete, continuous], axis=1)

    def _transform_configuration(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a complete configuration frame with exact columns."""
        if set(frame.columns) != set(self.parameter_names):
            raise IncompatibilityError(
                "Configuration frame columns must match the owned parameter "
                f"names {list(self.parameter_names)}."
            )
        for _, row in frame.iterrows():
            self.validate(row.to_dict())
        transformed: list[pd.DataFrame] = []
        for parameter in self._comp_parameters:
            values = frame[parameter.name]
            if parameter.is_discrete:
                transformed.append(parameter.transform(values))
            else:
                transformed.append(values.to_frame(name=parameter.name))
        if not transformed:
            return pd.DataFrame(index=frame.index)
        result = pd.concat(transformed, axis=1)
        return result.loc[:, list(self.comp_rep_columns)]

    def _transform_discrete(
        self,
        frame: pd.DataFrame,
        /,
        *,
        allow_missing: bool,
        allow_extra: bool,
    ) -> pd.DataFrame:
        """Transform the discrete part like :class:`SubspaceDiscrete`."""
        parameters = get_transform_objects(
            frame,
            self._discrete_parameters,
            allow_missing=allow_missing,
            allow_extra=allow_extra,
        )
        if len(frame) < 1:
            return pd.DataFrame(index=frame.index)

        frames = [
            parameter.transform(frame[parameter.name]) for parameter in parameters
        ]
        comp_rep = pd.concat(frames, axis=1) if frames else pd.DataFrame()
        columns = [
            column
            for parameter in self._discrete_parameters
            for column in parameter.comp_rep_columns
        ]
        return comp_rep[columns]

    def _transform_continuous(
        self,
        frame: pd.DataFrame,
        /,
        *,
        allow_missing: bool,
        allow_extra: bool,
    ) -> pd.DataFrame:
        """Transform the continuous part like :class:`SubspaceContinuous`."""
        parameters = get_transform_objects(
            frame,
            self._continuous_parameters,
            allow_missing=allow_missing,
            allow_extra=allow_extra,
        )
        return frame[[parameter.name for parameter in parameters]]

    def validate(self, configuration: Mapping[str, Any]) -> None:
        """Validate exact parameter keys and values without mutating the input.

        Args:
            configuration: A mapping from every owned parameter name to a value.

        Raises:
            IncompatibilityError: If keys, types, finiteness, or values are invalid.
        """
        if not isinstance(configuration, Mapping):
            raise TypeError("A configuration must be a mapping of parameter values.")
        expected = set(self.parameter_names)
        received = set(configuration)
        if missing := sorted(expected - received):
            raise IncompatibilityError(
                f"Configuration is missing parameter(s): {missing}."
            )
        if extra := sorted(received - expected):
            raise IncompatibilityError(
                f"Configuration contains unknown parameter(s): {extra}."
            )

        for parameter in self.parameters:
            value = configuration[parameter.name]
            if parameter.is_numerical:
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise IncompatibilityError(
                        f"Numerical parameter '{parameter.name}' requires a real value."
                    )
                if not isfinite(float(value)):
                    raise IncompatibilityError(
                        f"Numerical parameter '{parameter.name}' must be finite."
                    )
            if isinstance(parameter, NumericalDiscreteParameter):
                is_valid = value in parameter.values
            else:
                is_valid = parameter.is_in_range(value)
            if not is_valid:
                raise IncompatibilityError(
                    f"Value {value!r} is outside the allowed values or bounds of "
                    f"parameter '{parameter.name}'."
                )

    def validate_numerical_parameters(self) -> None:
        """Reject parameter kinds unsupported by numeric optimization.

        Raises:
            IncompatibilityError: If an owned parameter is not numeric and continuous
                or discrete.
        """
        unsupported = [
            parameter.name
            for parameter in self.parameters
            if not isinstance(
                parameter, (NumericalContinuousParameter, NumericalDiscreteParameter)
            )
        ]
        if unsupported:
            raise IncompatibilityError(
                "Full-configuration optimization and random sampling support only "
                "numerical continuous and discrete parameters; "
                f"unsupported: {unsupported}."
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
