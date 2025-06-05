"""Categorical parameters."""

import gc
from functools import cached_property

import numpy as np
import pandas as pd
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import override

from baybe.parameters.base import _DiscreteLabelLikeParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import validate_unique_values
from baybe.utils.conversion import nonstring_to_tuple
from baybe.utils.numerical import DTypeFloatNumpy


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

    encoding: CategoricalEncoding = field(default=CategoricalEncoding.INT, init=False)
    # See base class.


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
