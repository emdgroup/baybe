"""Categorical parameters."""

import gc
from enum import Enum
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.parameters.base import _DiscreteLabelLikeParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import validate_unique_values
from baybe.settings import active_settings
from baybe.utils.conversion import nonstring_to_tuple


class TaskCorrelation(Enum):
    """Task correlation modes for TaskParameter."""

    UNKNOWN = "unknown"
    POSITIVE = "positive"


def _convert_values(value, self, field) -> tuple[str, ...]:
    """Sort and convert values for categorical parameters."""
    value = nonstring_to_tuple(value, self, field)
    return tuple(sorted(value, key=lambda x: (str(type(x)), x)))


def _validate_label_min_len(self, attr, value) -> None:
    """An attrs-compatible validator to ensure minimum label length."""  # noqa: D401
    if isinstance(value, str) and len(value) < 1:
        raise ValueError(
            f"Strings used as '{attr.alias}' for '{self.__class__.__name__}' must "
            f"have at least 1 character."
        )


@define(frozen=True, slots=False)
class CategoricalParameter(_DiscreteLabelLikeParameter):
    """Parameter class for categorical parameters."""

    # object variables
    _values: tuple[str | bool, ...] = field(
        alias="values",
        converter=Converter(_convert_values, takes_self=True, takes_field=True),  # type: ignore
        validator=(  # type: ignore
            validate_unique_values,
            deep_iterable(
                member_validator=(instance_of((str, bool)), _validate_label_min_len),
                iterable_validator=min_len(2),
            ),
        ),
    )
    # See base class.

    encoding: CategoricalEncoding = field(
        default=CategoricalEncoding.OHE, converter=CategoricalEncoding
    )
    # See base class.

    @override
    @property
    def values(self) -> tuple:
        """The values of the parameter."""
        return self._values

    @override
    @cached_property
    def comp_df(self) -> pd.DataFrame:
        if self.encoding is CategoricalEncoding.OHE:
            cols = [
                f"{self.name}_{'b' if isinstance(val, bool) else ''}{val}"
                for val in self.values
            ]
            comp_df = pd.DataFrame(
                np.eye(len(self.values), dtype=active_settings.DTypeFloatNumpy),
                columns=cols,
            )
        elif self.encoding is CategoricalEncoding.INT:
            comp_df = pd.DataFrame(
                range(len(self.values)),
                dtype=active_settings.DTypeFloatNumpy,
                columns=[self.name],
            )
        comp_df.index = pd.Index(self.values)

        return comp_df


@define(frozen=True, slots=False)
class TaskParameter(CategoricalParameter):
    """Parameter class for task parameters."""

    encoding: CategoricalEncoding = field(default=CategoricalEncoding.INT, init=False)
    # See base class.

    task_correlation: TaskCorrelation = field(default=TaskCorrelation.POSITIVE)
    """Task correlation. Defaults to positive correlation via PositiveIndexKernel."""

    @task_correlation.validator
    def _validate_task_correlation_active_values(  # noqa: DOC101, DOC103
        self, _: Any, value: TaskCorrelation
    ) -> None:
        """Validate active values compatibility with task correlation mode.

        Raises:
            ValueError: If task_correlation is POSITIVE but active_values contains more
                than one value.
        """
        # Check POSITIVE constraint: must have exactly one active value
        # Note: _active_values is the internal field, could be None
        if value == TaskCorrelation.POSITIVE and self._active_values is not None:
            if len(self._active_values) > 1:
                raise ValueError(
                    f"Task correlation '{TaskCorrelation.POSITIVE.value}' requires "
                    f"one active value, but {len(self._active_values)} were provided: "
                    f"{self._active_values}. The POSITIVE mode uses the "
                    f"PositiveIndexKernel which assumes a single target task."
                )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
