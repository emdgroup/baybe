"""Categorical parameters."""

import gc

import narwhals.stable.v2 as nw
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import assert_never, override

from baybe.parameters.base import _JOIN_KEY, _EncodedDiscreteParameter
from baybe.parameters.enum import CategoricalEncoding
from baybe.parameters.validation import validate_unique_values
from baybe.utils.conversion import nonstring_to_tuple


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
class CategoricalParameter(_EncodedDiscreteParameter):
    """Parameter class for categorical parameters."""

    # object variables
    _values: tuple[str | bool, ...] = field(
        alias="values",
        converter=Converter(_convert_values, takes_self=True, takes_field=True),  # type: ignore
        validator=(
            validate_unique_values,
            deep_iterable(
                member_validator=(instance_of((str, bool)), _validate_label_min_len),
                iterable_validator=min_len(2),
            ),
        ),
    )
    # See base class.

    encoding: CategoricalEncoding = field(
        default=CategoricalEncoding.OHE, converter=CategoricalEncoding, kw_only=True
    )
    # See base class.

    @override
    @property
    def values(self) -> tuple:
        """The values of the parameter."""
        return self._values

    @override
    def summary(self) -> dict:
        return {**super().summary(), "Encoding": self.encoding}

    @override
    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        if self.encoding is CategoricalEncoding.OHE:
            return tuple(
                f"{self.name}_{'b' if isinstance(val, bool) else ''}{val}"
                for val in self.values
            )
        if self.encoding is CategoricalEncoding.INT:
            return (self.name,)

        assert_never(self.encoding)

    @override
    def _encoding_table(self, values: nw.Series, /) -> nw.DataFrame:
        if self.encoding is CategoricalEncoding.OHE:
            # TODO[narwhalify]: avoid hard-coded float type
            return (
                values.rename(_JOIN_KEY)
                .to_frame()
                .with_columns(
                    (nw.col(_JOIN_KEY) == v).cast(nw.Float64).alias(ohe_col)
                    for v, ohe_col in zip(self.values, self.comp_rep_columns)
                )
            )

        if self.encoding is CategoricalEncoding.INT:
            # TODO[narwhalify]: avoid hard-coded float type
            mapping = {v: float(i) for i, v in enumerate(self.values)}
            return (
                values.rename(_JOIN_KEY)
                .to_frame()
                .with_columns(
                    nw.col(_JOIN_KEY)
                    .replace_strict(mapping, return_dtype=nw.Float64)
                    .alias(self.name)
                )
            )

        assert_never(self.encoding)


@define(frozen=True, slots=False)
class TaskParameter(CategoricalParameter):
    """Parameter class for task parameters."""

    encoding: CategoricalEncoding = field(default=CategoricalEncoding.INT, init=False)
    # See base class.


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
