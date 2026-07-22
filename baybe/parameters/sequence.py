"""Sequence parameters."""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import cached_property
from itertools import chain, product

import narwhals.stable.v2 as nw
from attrs import Converter, define, field
from attrs.validators import (
    and_,
    deep_iterable,
    ge,
    instance_of,
    is_callable,
    min_len,
    optional,
)
from narwhals.stable.v2.typing import DataFrameT, SeriesT
from typing_extensions import override

from baybe.exceptions import InfiniteSpaceError
from baybe.parameters.base import _EncodedDiscreteParameter
from baybe.utils.conversion import nonstring_to_tuple

SequenceEncoderCallable = Callable[[SeriesT], DataFrameT]


@define(frozen=True, slots=False)
class SequenceParameter(_EncodedDiscreteParameter):
    """Parameter class for sequence parameters."""

    alphabet: tuple[str, ...] = field(
        converter=Converter(  # type: ignore
            lambda value, self, field: tuple(
                sorted(nonstring_to_tuple(value, type(self), field))
            ),
            takes_self=True,
            takes_field=True,
        ),
        validator=deep_iterable(
            member_validator=and_(instance_of(str), min_len(1)),
            iterable_validator=min_len(1),
        ),
    )
    """The alphabet of the sequence parameter."""

    encoder: SequenceEncoderCallable = field(validator=is_callable())
    """The encoder function for the sequence parameter.
    It should take a Series of sequences and return a DataFrame with
    the encoded representation in exactly the same order as the input Series."""

    min_length: int = field(default=0, validator=ge(0), kw_only=True)
    """The minimum number of letters from the alphabet for
    constructing a sequence."""

    max_length: int | None = field(
        default=None, validator=optional(ge(1)), kw_only=True
    )
    """Optional maximum number of letters from the alphabet for constructing
    a sequence. If provided, the parameter becomes finite."""

    @max_length.validator
    def _validate_max_length(  # noqa: DOC101, DOC103
        self, _: object, value: int | None
    ) -> None:
        """Validate the maximum length.

        Raises:
            ValueError: If the maximum length is less than the minimum length.
        """
        if value is not None and value < self.min_length:
            raise ValueError(
                f"Maximum length ({value}) must be greater than or equal to "
                f"minimum length ({self.min_length})."
            )

    @override
    @property
    def is_finite(self) -> bool:
        return self.max_length is not None

    @cached_property
    def _enumerate_values(self) -> tuple[tuple[str, ...], ...]:
        """Enumerate all possible values of the sequence parameter.

        Returns:
            A tuple of all possible values.

        Raises:
            InfiniteSpaceError: If the parameter has no maximum length.
        """
        if not self.is_finite:
            raise InfiniteSpaceError(
                f"Cannot enumerate values for a {self.__class__.__name__} "
                "without an explicit maximum length."
            )
        assert self.max_length is not None, (
            "max_length must be set for finite parameters."
        )
        all_values = map(
            tuple,
            chain.from_iterable(
                product(self.alphabet, repeat=length)
                for length in range(self.min_length, self.max_length + 1)
            ),
        )
        return tuple(all_values)

    @override
    @property
    def values(self) -> tuple[tuple[str, ...], ...]:
        if not self.is_finite:
            raise InfiniteSpaceError(
                f"Cannot enumerate values for a {self.__class__.__name__} "
                "without an explicit maximum length."
            )
        return self._enumerate_values

    @property
    @override
    def comp_rep_columns(self) -> tuple[str, ...]:
        return (self.name,)

    @override
    def _encoding_table(self, values: nw.Series, /) -> nw.DataFrame:
        return self.encoder(values)

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
