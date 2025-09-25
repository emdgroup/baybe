"""Functionality for desirability objectives."""

from __future__ import annotations

import gc
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field, fields
from attrs.validators import deep_iterable, gt, instance_of, min_len
from typing_extensions import override

from baybe.exceptions import IncompatibilityError
from baybe.objectives.base import Objective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.validation import validate_target_names
from baybe.targets import NumericalTarget
from baybe.targets.base import Target
from baybe.targets.numerical import UncertainBool
from baybe.utils.basic import to_tuple
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import pretty_print_df
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective
    from torch import Tensor

_OUTPUT_NAME = "Desirability"
"""The name of the output column produced by the desirability transform."""


def _geometric_mean(x: Tensor, /, weights: Tensor, dim: int = -1) -> Tensor:
    """Calculate the geometric mean of an array along a given dimension.

    Args:
        x: A tensor containing the values for the mean computation.
        weights: A tensor of weights whose shape must be broadcastable to the shape
            of the input tensor.
        dim: The dimension along which to compute the geometric mean.

    Returns:
        A tensor containing the weighted geometric means.
    """
    import torch

    # Ensure x is a floating-point tensor
    if not torch.is_floating_point(x):
        x = x.float()

    # Normalize weights
    normalized_weights = weights / torch.sum(weights)

    # Add epsilon to avoid log(0)
    eps = torch.finfo(x.dtype).eps
    log_tensor = torch.log(x + eps)

    # Compute the weighted log sum
    weighted_log_sum = torch.sum(log_tensor * normalized_weights.unsqueeze(0), dim=dim)

    # Convert back from log domain
    return torch.exp(weighted_log_sum)


@define(frozen=True, slots=False)
class DesirabilityObjective(Objective):
    """An objective scalarizing multiple targets using desirability values."""

    is_multi_output: ClassVar[bool] = False
    # See base class.

    _targets: tuple[NumericalTarget, ...] = field(
        converter=to_tuple,
        validator=[
            min_len(2),
            deep_iterable(member_validator=instance_of(NumericalTarget)),
            validate_target_names,
        ],
        alias="targets",
    )
    "The targets considered by the objective."

    weights: tuple[float, ...] = field(
        converter=lambda w: cattrs.structure(w, tuple[float, ...]),
        validator=deep_iterable(member_validator=[finite_float, gt(0.0)]),
    )
    """The weights to balance the different targets.
    By default, all targets are considered equally important."""

    scalarizer: Scalarizer = field(default=Scalarizer.GEOM_MEAN, converter=Scalarizer)
    """The mechanism to scalarize the weighted desirability values of all targets."""

    require_normalization: bool = field(
        default=True, validator=instance_of(bool), kw_only=True
    )
    """Controls if the targets must be normalized."""

    as_pre_transformation: bool = field(
        default=False, validator=instance_of(bool), kw_only=True
    )
    """Controls if the desirability computation is applied as a pre-transformation."""

    @weights.default
    def _default_weights(self) -> tuple[float, ...]:
        """Create unit weights for all targets."""
        return tuple(1.0 for _ in range(len(self.targets)))

    @_targets.validator
    def _validate_targets(self, _, targets) -> None:  # noqa: DOC101, DOC103
        # Validate non-negativity when using geometric mean
        if self.scalarizer is Scalarizer.GEOM_MEAN and (
            negative := {t.name for t in targets if t.get_codomain().lower < 0}
        ):
            raise ValueError(
                f"Using '{Scalarizer.GEOM_MEAN}' for '{self.__class__.__name__}' "
                f"requires that all targets are transformed to a non-negative range. "
                f"However, the images of the following targets cover negative values: "
                f"{negative}."
            )

        # Validate normalization
        if self.require_normalization and (
            unnormalized := {
                t.name for t in targets if t.is_normalized is not UncertainBool.TRUE
            }
        ):
            raise ValueError(
                f"By default, '{self.__class__.__name__}' only accepts normalized "
                f"targets but the following targets are either not normalized or their "
                f"normalization status is unclear because the image "
                f"of the underlying transformation is unknown: {unnormalized}. "
                f"Either normalize your targets (e.g. using their "
                f"'{NumericalTarget.normalize.__name__}' method / by specifying "
                f"a suitable target transformation) or explicitly set "
                f"'{DesirabilityObjective.__name__}."
                f"{fields(DesirabilityObjective).require_normalization.name}' to "
                f"'False' to allow unnormalized targets."
            )

    @weights.validator
    def _validate_weights(self, _, weights) -> None:  # noqa: DOC101, DOC103
        if (lw := len(weights)) != (lt := len(self.targets)):
            raise ValueError(
                f"If custom weights are specified, there must be one for each target. "
                f"Specified number of targets: {lt}. Specified number of weights: {lw}."
            )

    @override
    @property
    def targets(self) -> tuple[NumericalTarget, ...]:
        return self._targets

    @override
    @property
    def _modeled_quantity_names(self) -> tuple[str, ...]:
        return (
            self.output_names
            if self.as_pre_transformation
            else tuple(t.name for t in self.targets)
        )

    @override
    @property
    def output_names(self) -> tuple[str, ...]:
        return (_OUTPUT_NAME,)

    @override
    @property
    def supports_partial_measurements(self) -> bool:
        return not self.as_pre_transformation

    @cached_property
    def _normalized_weights(self) -> np.ndarray:
        """The normalized target weights."""
        return np.asarray(self.weights) / np.sum(self.weights)

    @override
    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)
        targets_df["Weight"] = self.weights

        fields = [
            to_string("Type", self.__class__.__name__, single_line=True),
            to_string("Targets", pretty_print_df(targets_df)),
            to_string("Scalarizer", self.scalarizer.name, single_line=True),
        ]

        return to_string("Objective", *fields)

    @override
    @property
    def _oriented_targets(self) -> tuple[Target, ...]:
        # For desirability, we do not only negate but also shift by 1 so that
        # normalized minimization targets are still mapped to [0, 1] instead of [-1, 0]
        # to enable geometric averaging.
        return tuple(
            t.negate() + 1 if isinstance(t, NumericalTarget) and t.minimize else t
            for t in self.targets
        )

    @override
    @property
    def _full_transformation(self) -> MCAcquisitionObjective:
        return self._to_botorch_full()

    @override
    def to_botorch(self) -> MCAcquisitionObjective:
        if self.as_pre_transformation:
            return NumericalTarget(_OUTPUT_NAME).to_objective().to_botorch()
        else:
            return self._to_botorch_full()

    def _to_botorch_full(self) -> MCAcquisitionObjective:
        """Create a BoTorch objective conducting the full desirability transform.

        Full transformation means:
            1. Starting from the raw target values
            2. Applying the individual target transformations
            3. Scalarizing the transformed values into a desirability score

        This differs from the regular :meth:`to_botorch` in that the entire
        transformation step is represented end-to-end by the returned objective, whereas
        the former only captures the part of the transformation starting from the point
        where the surrogate model(s) are applied (i.e. which may or may not include the
        desirability scalarization step, depending on the  chosen`as_pre_transformation`
        setting).
        """
        import torch
        from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective

        from baybe.objectives.botorch import ChainedMCObjective

        if self.scalarizer is Scalarizer.MEAN:
            outer = LinearMCObjective(torch.tensor(self._normalized_weights))

        elif self.scalarizer is Scalarizer.GEOM_MEAN:
            outer = GenericMCObjective(
                lambda samples, X: _geometric_mean(
                    samples, torch.tensor(self._normalized_weights)
                )
            )

        else:
            raise NotImplementedError(
                f"No scalarization mechanism defined for '{self.scalarizer.name}'."
            )

        inner = super().to_botorch()
        return ChainedMCObjective(inner, outer)

    @override
    def _pre_transform(
        self,
        df: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        if not self.as_pre_transformation:
            return super()._pre_transform(
                df, allow_missing=allow_missing, allow_extra=allow_extra
            )

        if allow_missing:
            raise IncompatibilityError(
                f"Setting 'allow_missing=True' is not supported for "
                f"'{self.__class__.__name__}.{self._pre_transform.__name__}' when "
                f"'{fields(self.__class__).as_pre_transformation.name}=True' since "
                f"the involved desirability computation requires all target columns "
                f"to be present."
            )

        return self.transform(df, allow_missing=False, allow_extra=allow_extra)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
