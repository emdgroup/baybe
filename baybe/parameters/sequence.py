"""Sequence parameters."""

from __future__ import annotations

import gc
from collections.abc import Callable
from functools import cached_property
from itertools import chain, product

import narwhals.stable.v2 as nw
from attrs import define, field
from attrs.validators import instance_of, is_callable, optional
from narwhals.stable.v2.typing import DataFrameT, SeriesT
from typing_extensions import override

from baybe.exceptions import InfiniteParameterError
from baybe.parameters.base import _EncodedDiscreteParameter

SequenceEncoderCallable = Callable[[SeriesT], DataFrameT]


@define(frozen=True, slots=False)
class SequenceParameter(_EncodedDiscreteParameter):
    """Parameter class for sequence parameters."""

    alphabet: tuple[str, ...] = field(converter=tuple, validator=instance_of(tuple))
    """The alphabet of the sequence parameter."""

    encoder: SequenceEncoderCallable = field(validator=is_callable())
    """The encoder function for the sequence parameter.
    It should take a Series of sequences and return a DataFrame with
    the encoded representation in exactly the same order as the input Series."""

    min_length: int = field(default=0, validator=instance_of(int), kw_only=True)
    """The minimum length of the sequence parameter."""

    max_length: int | None = field(
        default=None, validator=optional(instance_of(int)), kw_only=True
    )
    """Optional maximum length of the sequence parameter."""

    @alphabet.validator
    def _validate_alphabet(  # noqa: DOC101, DOC103
        self, _: object, value: tuple[str, ...]
    ) -> None:
        """Validate the alphabet."""
        if not value:
            raise ValueError("Alphabet cannot be empty.")
        if any(len(ch) != 1 for ch in value):
            raise ValueError(
                "All characters in the alphabet must be single characters."
            )

    @min_length.validator
    def _validate_min_length(  # noqa: DOC101, DOC103
        self, _: object, value: int
    ) -> None:
        """Validate the minimum length."""
        if value < 0:
            raise ValueError("Minimum length cannot be negative.")

    @max_length.validator
    def _validate_max_length(  # noqa: DOC101, DOC103
        self, _: object, value: int | None
    ) -> None:
        """Validate the maximum length."""
        if value is None:
            return
        if value < 1:
            raise ValueError("Maximum length must be a positive integer.")
        if value < self.min_length:
            raise ValueError(
                f"Maximum length ({value}) must be greater than or equal to "
                f"minimum length ({self.min_length})."
            )

    @override
    @property
    def is_finite(self) -> bool:
        """Indicates whether the parameter has a finite number of values."""
        return self.max_length is not None

    def _enumerate_values(self) -> tuple[str, ...]:
        """Enumerate all possible values of the sequence parameter.

        Returns:
            A tuple of all possible values.

        Raises:
            InfiniteParameterError: If the parameter has no maximum length.
        """
        if not self.is_finite:
            raise InfiniteParameterError(
                "Cannot enumerate values for a SequenceParameter "
                "without an explicit maximum length."
            )
        assert self.max_length is not None, (
            "max_length must be set for finite parameters."
        )
        all_values = map(
            "".join,
            chain.from_iterable(
                product(sorted(self.alphabet), repeat=length)
                for length in range(self.min_length, self.max_length + 1)
            ),
        )
        return tuple(all_values)

    @override
    @cached_property
    def values(self) -> tuple:
        """The values the parameter can take.

        Returns:
            A tuple of all possible values.

        Raises:
            InfiniteParameterError: If the parameter has no maximum length.
        """
        if not self.is_finite:
            raise InfiniteParameterError(
                "Cannot enumerate values for a SequenceParameter "
                "without an explicit maximum length."
            )
        return self._enumerate_values()

    @override
    @property
    def comp_rep_columns(self) -> tuple[str, ...]:
        """The columns spanning the computational representation."""
        return (self.name,)

    @override
    def _encoding_table(self, values: nw.Series, /) -> nw.DataFrame:
        return self.encoder(values)

    @override
    def is_in_range(self, item: str) -> bool:
        if not isinstance(item, str):
            return False
        item_length = len(item)
        if (
            not item
            or item_length < self.min_length
            or (self.max_length is not None and item_length > self.max_length)
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
