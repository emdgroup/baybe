"""Categorical parameters."""

import gc
from functools import cached_property

import narwhals.stable.v2 as nw
from attr.converters import optional as optional_c
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import assert_never, override

from baybe.kernels.base import Kernel
from baybe.parameters.base import (
    _JOIN_KEY,
    _EncodedDiscreteParameter,
)
from baybe.parameters.enum import CategoricalEncoding, TransferLearningMode
from baybe.utils.conversion import nonstring_to_tuple, sort_tuple
from baybe.utils.validation import validate_unique_values


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
        converter=[  # type: ignore[misc]
            Converter(nonstring_to_tuple, takes_self=True, takes_field=True),  # type: ignore[call-overload]
            sort_tuple,
        ],
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
    """The encoding used the generate the parameters computational representation."""

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
        if invalid := set(values.to_list()) - set(self.values):
            raise ValueError(
                f"Cannot encode the value(s) {invalid}. "
                f"Valid choices are: {set(self.values)}."
            )

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
    """The encoding used the generate the parameters computational representation."""

    _override_kernel: None = field(init=False, default=None)
    """Task parameters derive their kernel override from the transfer learning mode."""

    override_transfer_learning_mode: TransferLearningMode | None = field(
        default=None,
        converter=optional_c(TransferLearningMode),
    )
    """Optional override for how the task dimension is modeled.

    Only applies to :class:`.GaussianProcessSurrogate`. When ``None``, the surrogate's
    kernel factory decides how the task dimension is treated. When set, the surrogate
    attaches the requested task kernel to a task-free base kernel derived from the
    configured factory.
    """

    @override
    @cached_property
    def override_kernel(self) -> Kernel | None:
        """The task kernel defined by the transfer learning mode, if any."""
        from baybe.kernels.basic import IndexKernel, PositiveIndexKernel

        n_tasks, names = len(self.values), (self.name,)
        match mode := self.override_transfer_learning_mode:
            case None:
                return None
            case TransferLearningMode.POSITIVE_INDEX_KERNEL:
                return PositiveIndexKernel(
                    num_tasks=n_tasks, rank=n_tasks, parameter_names=names
                )
            case TransferLearningMode.INDEX_KERNEL:
                return IndexKernel(
                    num_tasks=n_tasks, rank=n_tasks, parameter_names=names
                )
            case _:
                assert_never(mode)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
