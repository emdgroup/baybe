"""Sequence parameters."""

from __future__ import annotations

import gc
from functools import cached_property
from itertools import chain, product
from typing import Protocol, runtime_checkable

import narwhals.stable.v2 as nw
from attrs import define, field
from attrs.validators import instance_of, optional
from typing_extensions import override

from baybe.exceptions import InfiniteParameterError
from baybe.parameters.base import _JOIN_KEY, _EncodedDiscreteParameter


@runtime_checkable
class SequenceEncoderProtocol(Protocol):
    """Protocol for sequence encoder callables.

    The callable receives the sequence values and the name of the key column it
    must include in the returned DataFrame. The key column must contain the
    original input values (used as join key in :meth:`DiscreteParameter.transform`).
    """

    __slots__ = ()

    def encode(
        self, values: nw.Series, alphabet: tuple[str, ...], *, key: str, name: str
    ) -> nw.DataFrame:
        """Encode the given sequence values.

        Args:
            values: The unique sequence values to encode.
            alphabet: The alphabet of the sequence parameter.
            key: The column name to use for the original values in the returned
                DataFrame.
            name: The name of the parameter, which should be used for the
                encoded representation column in the returned DataFrame.

        Returns:
            A DataFrame with the key column containing the original values and
            one column named after the parameter containing the encoded
            representation.
        """
        ...


@define(frozen=True, slots=False)
class SequenceParameter(_EncodedDiscreteParameter):
    """Parameter class for sequence parameters."""

    alphabet: tuple[str, ...] = field(converter=tuple, validator=instance_of(tuple))
    """The alphabet of the sequence parameter."""

    encoder: SequenceEncoderProtocol = field(
        validator=instance_of(SequenceEncoderProtocol)
    )
    """The encoder function for the sequence parameter."""

    min_length: int = field(default=0, validator=instance_of(int), kw_only=True)
    """The minimum length of the sequence parameter."""

    max_length: int | None = field(
        default=None, validator=optional(instance_of(int)), kw_only=True
    )
    """Optional maximum length of the sequence parameter."""

    @alphabet.validator
    def _validate_alphabet(self, _: object, value: tuple[str, ...]) -> None:
        """Validate the alphabet."""
        if not value:
            raise ValueError("Alphabet cannot be empty.")
        if any(len(ch) != 1 for ch in value):
            raise ValueError(
                "All characters in the alphabet must be single characters."
            )

    @min_length.validator
    def _validate_min_length(self, _: object, value: int) -> None:
        """Validate the minimum length."""
        if value < 0:
            raise ValueError("Minimum length cannot be negative.")

    @max_length.validator
    def _validate_max_length(self, _: object, value: int | None) -> None:
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
        encoded = self.encoder.encode(
            values, self.alphabet, key=_JOIN_KEY, name=self.name
        )

        if _JOIN_KEY not in encoded.columns:
            raise ValueError(
                f"The encoder for parameter '{self.name}' must return a DataFrame "
                f"with a column named '{_JOIN_KEY}' containing the original values."
            )
        if self.name not in encoded.columns:
            raise ValueError(
                f"The encoder for parameter '{self.name}' must return a DataFrame "
                f"with a column named '{self.name}' containing the encoded "
                f"representation."
            )
        if set(encoded[_JOIN_KEY].to_list()) != set(values.to_list()):
            raise ValueError(
                f"The encoder for parameter '{self.name}' must return a DataFrame "
                f"whose '{_JOIN_KEY}' column contains exactly the input values."
            )

        return encoded

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
