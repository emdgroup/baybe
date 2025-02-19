"""Functionality for single-target objectives."""

import gc
import warnings

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.targets.base import Target
from baybe.targets.enum import TargetMode
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import get_transform_objects, pretty_print_df
from baybe.utils.plotting import to_string


@define(frozen=True, slots=False)
class SingleTargetObjective(Objective):
    """An objective focusing on a single target."""

    _target: Target = field(validator=instance_of(Target), alias="target")
    """The single target considered by the objective."""

    @override
    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)

        fields = [
            to_string("Type", self.__class__.__name__, single_line=True),
            to_string("Targets", pretty_print_df(targets_df)),
        ]

        return to_string("Objective", *fields)

    @override
    @property
    def targets(self) -> tuple[Target, ...]:
        return (self._target,)

    @override
    def transform(
        self,
        df: pd.DataFrame | None = None,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool | None = None,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # >>>>>>>>>> Deprecation
        if not ((df is None) ^ (data is None)):
            raise ValueError(
                "Provide the dataframe to be transformed as argument to `df`."
            )

        if data is not None:
            df = data
            warnings.warn(
                "Providing the dataframe via the `data` argument is deprecated and "
                "will be removed in a future version. Please pass your dataframe "
                "as positional argument instead.",
                DeprecationWarning,
            )

        # Mypy does not infer from the above that `df` must be a dataframe here
        assert isinstance(df, pd.DataFrame)

        if allow_extra is None:
            allow_extra = True
            if set(df.columns) - {p.name for p in self.targets}:
                warnings.warn(
                    "For backward compatibility, the new `allow_extra` flag is set "
                    "to `True` when left unspecified. However, this behavior will be "
                    "changed in a future version. If you want to invoke the old "
                    "behavior, please explicitly set `allow_extra=True`.",
                    DeprecationWarning,
                )
        # <<<<<<<<<< Deprecation

        # Even for a single target, it is convenient to use the existing validation
        # machinery instead of re-implementing it
        get_transform_objects(
            df, [self._target], allow_missing=allow_missing, allow_extra=allow_extra
        )

        target_data = df[self._target.name].copy()

        out = self._target.transform(target_data).to_frame()

        # TODO: Remove hotfix (https://github.com/emdgroup/baybe/issues/460)
        if (
            isinstance(t := self._target, NumericalTarget)
            and t.mode is TargetMode.MIN
            and t.bounds.is_bounded
        ):
            out = -out
        return out


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
