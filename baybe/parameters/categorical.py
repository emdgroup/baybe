"""Categorical parameters."""

import gc
from functools import cached_property
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len

from baybe.parameters.base import DiscreteParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import validate_unique_values
from baybe.utils.conversion import nonstring_to_tuple
from baybe.utils.numerical import DTypeFloatNumpy


def _convert_values(value, self, field) -> tuple[str, ...]:
    """Sort and convert values for categorical parameters."""
    value = nonstring_to_tuple(value, self, field)
    return tuple(sorted(value))


@define(frozen=True, slots=False)
class CategoricalParameter(DiscreteParameter):
    """Parameter class for categorical parameters."""

    # class variables
    is_numerical: ClassVar[bool] = False
    # See base class.

    # object variables
    _values: tuple[str, ...] = field(
        alias="values",
        converter=Converter(_convert_values, takes_self=True, takes_field=True),  # type: ignore
        validator=(  # type: ignore
            min_len(2),
            validate_unique_values,
            deep_iterable(member_validator=(instance_of(str), min_len(1))),
        ),
    )
    # See base class.

    encoding: CategoricalEncoding = field(
        default=CategoricalEncoding.OHE, converter=CategoricalEncoding
    )
    # See base class.

    @property
    def values(self) -> tuple:
        """The values of the parameter."""
        return self._values

    @cached_property
    def comp_df(self) -> pd.DataFrame:  # noqa: D102
        # See base class.
        if self.encoding is CategoricalEncoding.OHE:
            cols = [f"{self.name}_{val}" for val in self.values]
            comp_df = pd.DataFrame(
                np.eye(len(self.values), dtype=DTypeFloatNumpy), columns=cols
            )
        elif self.encoding is CategoricalEncoding.INT:
            comp_df = pd.DataFrame(
                range(len(self.values)), dtype=DTypeFloatNumpy, columns=[self.name]
            )
        comp_df.index = pd.Index(self.values)

        return comp_df


@define(frozen=True, slots=False)
class TaskParameter(CategoricalParameter):
    """Parameter class for task parameters."""

    # object variables
    active_values: tuple = field(converter=tuple)
    """An optional list of values describing for which tasks recommendations should be
    given. By default, all parameters are considered active."""

    encoding: CategoricalEncoding = field(default=CategoricalEncoding.INT, init=False)
    # See base class.

    @active_values.default
    def _default_active_values(self) -> tuple:
        """Set all parameters active by default."""
        # TODO [16605]: Redesign metadata handling
        return self.values

    @active_values.validator
    def _validate_active_values(  # noqa: DOC101, DOC103
        self, _: Any, values: tuple
    ) -> None:
        """Validate the active parameter values.

        If no such list is provided, no validation is being performed. In particular,
        the errors listed below are only relevant if the ``values`` list is provided.

        Raises:
            ValueError: If an empty active parameters list is provided.
            ValueError: If the active parameter values are not unique.
            ValueError: If not all active values are valid parameter choices.
        """
        # TODO [16605]: Redesign metadata handling
        if len(values) == 0:
            raise ValueError(
                "If an active parameters list is provided, it must not be empty."
            )
        if len(set(values)) != len(values):
            raise ValueError("The active parameter values must be unique.")
        if not all(v in self.values for v in values):
            raise ValueError("All active values must be valid parameter choices.")


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
