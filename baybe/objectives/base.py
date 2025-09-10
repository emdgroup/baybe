"""Base classes for all objectives."""

from __future__ import annotations

import gc
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from attrs import define, field

from baybe.serialization.mixin import SerialMixin
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import is_all_instance
from baybe.utils.dataframe import get_transform_objects, to_tensor
from baybe.utils.metadata import Metadata, to_metadata

if TYPE_CHECKING:
    from botorch.acquisition.objective import MCAcquisitionObjective


# TODO: Reactive slots in all classes once cached_property is supported:
#   https://github.com/python-attrs/attrs/issues/164


@define(frozen=True, slots=False)
class Objective(ABC, SerialMixin):
    """Abstract base class for all objectives."""

    is_multi_output: ClassVar[bool]
    """Class variable indicating if the objective produces multiple outputs."""

    metadata: Metadata = field(
        factory=Metadata,
        converter=lambda x: to_metadata(x, Metadata),
        kw_only=True,
    )
    """Optional metadata containing description and other information."""

    @property
    def description(self) -> str | None:
        """The description of the objective."""
        return self.metadata.description

    @property
    @abstractmethod
    def targets(self) -> tuple[Target, ...]:
        """The targets included in the objective."""

    @property
    def _modeled_quantity_names(self) -> tuple[str, ...]:
        """The names of the quantities returned by the pre-transformation."""
        return tuple(t.name for t in self.targets)

    @property
    def _n_models(self) -> int:
        """The number of models used in the objective.

        Corresponds to the number of dimensions after the pre-transformation.
        """
        return len(self._modeled_quantity_names)

    @property
    def _is_multi_model(self) -> bool:
        """Check if the objective relies on multiple surrogate models."""
        return self._n_models > 1

    @property
    @abstractmethod
    def output_names(self) -> tuple[str, ...]:
        """The names of the outputs of the objective."""

    @property
    def n_outputs(self) -> int:
        """The number of outputs of the objective."""
        return len(self.output_names)

    @property
    @abstractmethod
    def supports_partial_measurements(self) -> bool:
        """Boolean indicating if the objective accepts partial target measurements."""

    @property
    def _oriented_targets(self) -> tuple[Target, ...]:
        """The targets with optional negation transformation for minimization."""
        return tuple(
            t.negate() if isinstance(t, NumericalTarget) and t.minimize else t
            for t in self.targets
        )

    @property
    def _full_transformation(self) -> MCAcquisitionObjective:
        """The end-to-end transformation applied, from targets to objective values."""
        return self.to_botorch()

    def to_botorch(self) -> MCAcquisitionObjective:
        """Convert to BoTorch representation."""
        if not is_all_instance(self._oriented_targets, NumericalTarget):
            raise NotImplementedError(
                "Conversion to BoTorch is only supported for numerical targets."
            )

        import torch
        from botorch.acquisition.multi_objective.objective import (
            GenericMCMultiOutputObjective,
        )

        return GenericMCMultiOutputObjective(
            lambda samples, X: torch.stack(
                [
                    t.transformation.to_botorch_objective()(samples[..., i])
                    for i, t in enumerate(self._oriented_targets)
                ],
                dim=-1,
            )
        )

    def _pre_transform(
        self,
        df: pd.DataFrame,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool = False,
    ) -> pd.DataFrame:
        """Pre-transform the target values prior to predictive modeling.

        For details on the method arguments, see :meth:`transform`.
        """
        # By default, we just pipe through the unmodified target values
        targets = get_transform_objects(
            df, self.targets, allow_missing=allow_missing, allow_extra=allow_extra
        )
        return df[[t.name for t in targets]]

    def transform(
        self,
        df: pd.DataFrame | None = None,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool | None = None,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Evaluate the objective on the target columns of the given dataframe.

        Args:
            df: The dataframe to be transformed. The allowed columns of the dataframe
                are dictated by the ``allow_missing`` and ``allow_extra`` flags.
            allow_missing: If ``False``, each target of the objective must have
                exactly one corresponding column in the given dataframe. If ``True``,
                the dataframe may contain only a subset of target columns.
            allow_extra: If ``False``, each column present in the dataframe must
                correspond to exactly one target of the objective. If ``True``, the
                dataframe may contain additional non-target-related columns, which
                will be ignored.
                The ``None`` default value is for temporary backward compatibility only
                and will be removed in a future version.
            data: Ignore! For backward compatibility only.

        Raises:
            ValueError: If dataframes are passed to both ``df`` and ``data``.

        Returns:
            A dataframe containing the objective values for the given input dataframe.
        """
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
            df,
            self._oriented_targets,  # <-- important to use oriented version
            allow_missing=allow_missing,
            allow_extra=allow_extra,
        )

        import torch

        with torch.no_grad():
            transformed = self._full_transformation(
                to_tensor(df[[t.name for t in targets]])
            )

        return pd.DataFrame(
            transformed.numpy(), columns=self.output_names, index=df.index
        )


def to_objective(x: Target | Objective, /) -> Objective:
    """Convert a target into an objective (with objective passthrough)."""
    return x if isinstance(x, Objective) else x.to_objective()


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
