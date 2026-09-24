"""Categorical parameters."""

import gc
from functools import cached_property

import numpy as np
import pandas as pd
from attr.converters import optional as optional_c
from attrs import Converter, define, field
from attrs.validators import deep_iterable, instance_of, min_len
from typing_extensions import assert_never, override

from baybe.kernels.base import Kernel
from baybe.parameters.base import _DiscreteLabelLikeParameter
from baybe.parameters.enum import CategoricalEncoding, TransferLearningMode
from baybe.settings import active_settings
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
class CategoricalParameter(_DiscreteLabelLikeParameter):
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
    @property
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
