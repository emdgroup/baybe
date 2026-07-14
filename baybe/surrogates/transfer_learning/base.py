"""Shared base for source/target transfer-learning surrogates."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from attrs import define, evolve, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.exceptions import IncompatibleSearchSpaceError
from baybe.surrogates.base import Surrogate
from baybe.surrogates.gaussian_process.core import (
    GaussianProcessSurrogate,
    _ModelContext,
)

if TYPE_CHECKING:
    import pandas as pd
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter
    from baybe.searchspace.core import SearchSpace


@define
class _SourceTargetTransferSurrogate(Surrogate, ABC):
    """Base class for transfer learning with one source and one target task.

    Handles the shared machinery for transfer-learning surrogates that build a target
    model from a single source model: validating that the search space describes
    exactly one source and one target task, splitting the measurements accordingly,
    fitting a single-task source Gaussian process on the reduced (task-free) search
    space, and stripping the task column from candidates at prediction time.

    Subclasses implement how the target model is built from the fitted source model
    (:meth:`_fit_target`) and how the posterior is assembled (``_posterior``).
    """

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    base_surrogate: GaussianProcessSurrogate = field(
        factory=GaussianProcessSurrogate,
        validator=instance_of(GaussianProcessSurrogate),
    )
    """The Gaussian process configuration used for the source and target models."""

    _source_gp: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The single-task GP trained on the source data. Available after fitting."""

    _numerical_indices: list[int] | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The comp-rep column indices of the non-task inputs. Available after fitting."""

    @override
    @staticmethod
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type[InputTransform] | None:
        """Disable base-class input scaling; the inner GPs handle scaling themselves.

        Args:
            parameter: The parameter for which a scaler would otherwise be created.

        Returns:
            Always ``None`` to signal that no base-class input scaling is applied.
        """
        return None

    @override
    @staticmethod
    def _make_target_scaler_factory() -> type[OutcomeTransform] | None:
        """Disable base-class output scaling; the inner GPs handle scaling themselves.

        Returns:
            Always ``None`` to signal that no base-class output scaling is applied.
        """
        return None

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the source GP and delegate target-model construction to the subclass.

        Args:
            train_x: Computational-representation inputs prepared by the base class.
                Not used directly, since the inner GPs are refitted from the stored
                measurements over the reduced search space.
            train_y: Target values prepared by the base class. Not used directly, for
                the same reason as ``train_x``.
        """
        assert self._searchspace is not None  # provided by base class
        assert self._objective is not None  # provided by base class
        assert self._measurements is not None  # provided by base class

        reduced_searchspace, source_measurements, target_measurements = (
            self._split_measurements()
        )

        self._source_gp = evolve(self.base_surrogate)
        self._source_gp.fit(reduced_searchspace, self._objective, source_measurements)

        context = _ModelContext(self._searchspace, self._objective, self._measurements)
        self._numerical_indices = context.numerical_indices

        self._fit_target(
            reduced_searchspace,
            self._objective,
            source_measurements,
            target_measurements,
        )

    def _split_measurements(self) -> tuple[SearchSpace, pd.DataFrame, pd.DataFrame]:
        """Validate the task setup and split measurements into source and target.

        Returns:
            The reduced (task-free) search space, the source measurements, and the
            target measurements.

        Raises:
            IncompatibleSearchSpaceError: If the search space has no task parameter,
                does not describe exactly one source and one target task, or lacks
                measurements for the source or the target task.
        """
        assert self._searchspace is not None
        assert self._measurements is not None

        searchspace = self._searchspace
        measurements = self._measurements

        task_param = searchspace._task_parameter
        if task_param is None:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires a search space that contains a "
                f"task parameter."
            )

        active_values = set(task_param.active_values)
        source_values = set(task_param.values) - active_values
        if len(active_values) != 1 or len(source_values) != 1:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' currently supports exactly one source "
                f"and one target task, but the task parameter describes "
                f"{len(source_values)} source and {len(active_values)} target value(s)."
            )
        (target_value,) = active_values
        (source_value,) = source_values

        task_name = task_param.name
        source_measurements = measurements[measurements[task_name] == source_value]
        target_measurements = measurements[measurements[task_name] == target_value]
        if source_measurements.empty or target_measurements.empty:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires measurements for both the "
                f"source and the target task, but received "
                f"{len(source_measurements)} source and "
                f"{len(target_measurements)} target measurement(s)."
            )

        reduced_searchspace = searchspace._drop_parameters({task_name})
        return reduced_searchspace, source_measurements, target_measurements

    def _strip_task(self, candidates_comp_scaled: Tensor) -> Tensor:
        """Remove the task column from full comp-rep candidates.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full (task-aware) search space.

        Returns:
            The candidates restricted to the non-task columns, matching the reduced
            space on which the inner GPs were trained.
        """
        import torch

        assert self._numerical_indices is not None  # set during fitting
        indices = torch.tensor(self._numerical_indices, dtype=torch.long)
        return candidates_comp_scaled.index_select(-1, indices)

    @abstractmethod
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        source_measurements: pd.DataFrame,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Build and fit the target model from the fitted source GP.

        Args:
            reduced_searchspace: The task-free search space for the inner GPs.
            objective: The objective (a single modeled quantity after replication).
            source_measurements: The measurements belonging to the source task.
            target_measurements: The measurements belonging to the target task.
        """


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
