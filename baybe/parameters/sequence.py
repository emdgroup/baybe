"""Sequence parameters."""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import cached_property
from itertools import chain, product
from typing import TypeAlias

import narwhals.stable.v2 as nw
from attrs import Attribute, Converter, define, field
from attrs.validators import (
    and_,
    deep_iterable,
    ge,
    instance_of,
    is_callable,
    min_len,
    optional,
)
from narwhals.stable.v2.typing import IntoDataFrame, IntoSeries
from typing_extensions import override

from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.base import _JOIN_KEY, _EncodedDiscreteParameter
from baybe.utils.conversion import nonstring_to_tuple

Encoder: TypeAlias = Callable[[IntoSeries], IntoDataFrame]
"""The protocol defining the contract for encoders.

Takes a series of values and returns the corresponding encoded representations as
a dataframe, where the row order matches the order of the input series.
"""


@define(frozen=True, slots=False)
class SequenceParameter(_EncodedDiscreteParameter):
    """Parameter class for sequence parameters."""

    alphabet: tuple[str, ...] = field(
        converter=Converter(  # type: ignore[misc]
            lambda value, self, field: tuple(
                sorted(nonstring_to_tuple(value, self, field))
            ),
            takes_self=True,
            takes_field=True,
        ),
        validator=deep_iterable(
            member_validator=and_(instance_of(str), min_len(1)),
            iterable_validator=min_len(1),
        ),
    )
    """The alphabet defining the tokens used to construct the sequences."""

    encoder: Encoder = field(validator=is_callable())
    """The used sequence encoder."""

    min_length: int = field(
        default=0, validator=and_(instance_of(int), ge(0)), kw_only=True
    )
    """The minimum token length of the constructed sequences."""

    max_length: int | None = field(
        default=None, validator=optional(and_(instance_of(int), ge(1))), kw_only=True
    )
    """Optional maximum token length of the constructed sequences."""

    @max_length.validator
    def _validate_max_length(  # noqa: DOC101, DOC103
        self, _: Attribute, value: int | None
    ) -> None:
        if value is not None and value < self.min_length:
            raise ValueError(
                f"The maximum sequence length ({value}) must be greater than or "
                f"equal to the minimum sequence length ({self.min_length})."
            )

    @override
    @property
    def is_finite(self) -> bool:
        return self.max_length is not None

    @override
    @cached_property
    def values(self) -> tuple[tuple[str, ...], ...]:
        if not self.is_finite:
            raise InfiniteSpaceError(
                f"Cannot enumerate the sequences of a '{self.__class__.__name__}' "
                "that has no explicit maximum length."
            )
        assert self.max_length is not None
        all_values = map(
            tuple,
            chain.from_iterable(
                product(self.alphabet, repeat=length)
                for length in range(self.min_length, self.max_length + 1)
            ),
        )
        return tuple(all_values)

    @property
    @override
    def comp_rep_columns(self) -> tuple[str, ...]:
        return (self.name,)

    @override
    def _encoding_table(self, values: nw.Series, /) -> nw.DataFrame:
        result = nw.from_native(self.encoder(values), eager_only=True)
        return (
            result.lazy()
            .collect(backend=nw.get_native_namespace(values))
            .with_columns(values.rename(_JOIN_KEY))
        )

    @override
    def is_in_range(self, item: tuple[str, ...]) -> bool:
        if not isinstance(item, tuple):
            return False
        length = len(item)
        if length < self.min_length or (
            self.max_length is not None and length > self.max_length
        ):
            return False

        return all(ch in self.alphabet for ch in item)

    @override
    def summary(self) -> dict:
        information: dict[str, object] = dict(
            Name=self.name,
            Type=self.__class__.__name__,
            Alphabet=self.alphabet,
            MinLength=self.min_length,
        )
        if self.max_length is not None:
            information["MaxLength"] = self.max_length
            information["nValues"] = len(self.values)
        return information


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
