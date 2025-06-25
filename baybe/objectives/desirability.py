"""Functionality for desirability objectives."""

from __future__ import annotations

import gc
import warnings
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar

import cattrs
import numpy as np
import pandas as pd
from attrs import define, field, fields
from attrs.validators import deep_iterable, gt, instance_of, min_len
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.objectives.enum import Scalarizer
from baybe.objectives.validation import validate_target_names
from baybe.targets import NumericalTarget
from baybe.targets.numerical import UncertainBool
from baybe.utils.basic import to_tuple
from baybe.utils.conversion import to_string
from baybe.utils.dataframe import get_transform_objects, pretty_print_df, to_tensor
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective
    from torch import Tensor

_OUTPUT_NAME = "Desirability"
"""The name of output column produced by the desirability transform."""


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

    require_normalization: bool = field(default=True, validator=instance_of(bool))
    """Boolean flag controlling whether the targets must be normalized."""

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
                f"'True' to allow unnormalized targets."
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
    def outputs(self) -> tuple[str, ...]:
        return (_OUTPUT_NAME,)

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
    def to_botorch(self) -> MCAcquisitionObjective:
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
                "Provide the dataframe to be transformed as first positional argument."
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

        targets = get_transform_objects(
            df, self.targets, allow_missing=allow_missing, allow_extra=allow_extra
        )

        import torch

        with torch.no_grad():
            transformed = self.to_botorch()(to_tensor(df[[t.name for t in targets]]))

        return pd.DataFrame(transformed.numpy(), columns=self.outputs, index=df.index)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
