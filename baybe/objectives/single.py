"""Functionality for single-target objectives."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import NonGaussianityError
from baybe.objectives.base import Objective
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget
from baybe.transformations.basic import AffineTransformation, IdentityTransformation
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import pretty_print_df

if TYPE_CHECKING:
    from botorch.acquisition.objective import (
        MCAcquisitionObjective,
        ScalarizedPosteriorTransform,
    )


@define(frozen=True, slots=False)
class SingleTargetObjective(Objective):
    """An objective focusing on a single target."""

    is_multi_output: ClassVar[bool] = False
    # See base class.

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
    @property
    def output_names(self) -> tuple[str, ...]:
        return (self._target.name,)

    @override
    @property
    def supports_partial_measurements(self) -> bool:
        return False

    @override
    def to_botorch(self) -> MCAcquisitionObjective:
        from botorch.acquisition.objective import IdentityMCObjective

        from baybe.objectives.botorch import ChainedMCObjective

        if isinstance(self._target, NumericalTarget):
            return ChainedMCObjective(super().to_botorch(), IdentityMCObjective())

        return IdentityMCObjective()

    @override
    def to_botorch_posterior_transform(self) -> ScalarizedPosteriorTransform | None:
        if not (
            isinstance((t := self._target), NumericalTarget)
            and isinstance(
                (tr := t.transformation), (IdentityTransformation, AffineTransformation)
            )
        ):
            raise NonGaussianityError(
                f"Converting an objective of type '{type(self).__name__}' is only "
                f"possible when the transformation result is Gaussian, that is, "
                f"when the target is of type '{NumericalTarget.__name__}' and the "
                f"assigned transformation is affine."
            )

        if isinstance(tr, IdentityTransformation):
            return None

        return (tr if not t.minimize else tr.negate()).to_botorch_posterior_transform()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
