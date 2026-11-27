"""Factorized parameter layout for complete multi-step configurations."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, instance_of

from baybe.exceptions import IncompatibilityError
from baybe.parameters.base import Parameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.utils import sort_parameters

if TYPE_CHECKING:
    from baybe.multistep.dag import MultiStepDAG


@define(frozen=True, slots=False)
class _ConfigurationSpace:
    """Describe configuration columns without enumerating their global product."""

    parameters: tuple[Parameter, ...] = field(
        converter=tuple, validator=deep_iterable(instance_of(Parameter))
    )
    """Parameters in computational representation order."""

    parameter_names: tuple[str, ...] = field(init=False)
    """Experimental parameter names in computational representation order."""

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
        discrete = sort_parameters(p for p in self.parameters if p.is_discrete)
        continuous = sort_parameters(p for p in self.parameters if p.is_continuous)
        comp_parameters = (*discrete, *continuous)
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
            if isinstance(parameter, NumericalContinuousParameter):
                lo, hi = parameter.bounds.to_tuple()
                lower.append(float(lo))
                upper.append(float(hi))
            else:
                raise IncompatibilityError(
                    f"Continuous parameter '{parameter.name}' has no numerical bounds."
                )

        object.__setattr__(self, "discrete_dims", MappingProxyType(dims))
        object.__setattr__(
            self,
            "_bounds",
            pd.DataFrame([lower, upper], index=["min", "max"], columns=columns),
        )

    @classmethod
    def from_dag(cls, dag: MultiStepDAG) -> _ConfigurationSpace:
        """Create a factorized layout from every DAG-owned parameter.

        Args:
            dag: The multi-step DAG supplying node-owned parameters.

        Returns:
            A factorized layout in the established BayBE computational order.
        """
        return cls(parameters=tuple(dag.parameter_by_name.values()))

    @property
    def bounds(self) -> pd.DataFrame:
        """Return computational-representation bounds as a defensive copy."""
        return self._bounds.copy(deep=True)

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
            if not parameter.is_in_range(value):
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
                "Full-configuration numerical optimization supports only numerical "
                f"continuous and discrete parameters; unsupported: {unsupported}."
            )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform experimental configuration columns using BayBE encodings.

        Args:
            frame: Experimental-representation configurations.

        Returns:
            Computational-representation configurations in layout column order.

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
        if list(frame.columns) != list(self.parameter_names):
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


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
