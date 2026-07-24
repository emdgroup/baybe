"""Shared base for source/target transfer-learning surrogates."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

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
    from botorch.posteriors import Posterior
    from torch import Tensor

    from baybe.objectives.base import Objective
    from baybe.parameters.base import Parameter
    from baybe.searchspace.core import SearchSpace


@define
class _SourceTargetTransferSurrogate(Surrogate, ABC):
    """Base class for transfer learning from one or more source tasks to a target task.

    Handles the machinery shared by transfer-learning surrogates that build a target
    model from one or more source models: validating the task configuration, splitting
    the measurements by task, fitting a single-task source Gaussian process for each
    source task that has data (on the reduced, task-free search space), and stripping
    the task column from candidates at prediction time.

    Subclasses implement how the target model is built from the fitted source models
    (:meth:`_fit_target`) and how the posterior is assembled (``_posterior``). They may
    also restrict the number of allowed source tasks via :attr:`_max_sources`.

    Cold start: the target task may have no measurements yet (a transfer-learning
    campaign switches to the predictive surrogate as soon as *any* measurements exist,
    including source ones). Subclasses are expected to fall back to a source-informed
    prediction in that regime; :meth:`_source_only_posterior` provides the single-source
    fallback.
    """

    supports_transfer_learning: ClassVar[bool] = True
    # See base class.

    _max_sources: ClassVar[int | None] = None
    """The maximum number of source tasks the surrogate supports (``None`` = no limit).

    Subclasses restricting themselves to a single source (e.g. mean/residual transfer)
    set this to ``1``.
    """

    base_surrogate: GaussianProcessSurrogate = field(
        factory=GaussianProcessSurrogate,
        validator=instance_of(GaussianProcessSurrogate),
    )
    """The Gaussian process configuration used for the source and target models."""

    _source_gps: tuple[GaussianProcessSurrogate, ...] = field(
        init=False, factory=tuple, eq=False, repr=False
    )
    """The single-task GPs trained on the source data, ordered by task value.

    Only source tasks that actually have measurements are included. Available after
    fitting.
    """

    _numerical_indices: list[int] | None = field(
        init=False, default=None, eq=False, repr=False
    )
    """The comp-rep column indices of the non-task inputs. Available after fitting."""

    @property
    def _source_gp(self) -> GaussianProcessSurrogate:
        """The single source GP, for single-source subclasses.

        Returns:
            The unique fitted source Gaussian process.

        Raises:
            RuntimeError: If there is not exactly one fitted source GP.
        """
        if len(self._source_gps) != 1:
            raise RuntimeError(
                f"'{self.__class__.__name__}' expected exactly one source GP, but "
                f"found {len(self._source_gps)}."
            )
        return self._source_gps[0]

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
        """Fit the source GPs and delegate target-model construction to the subclass.

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

        reduced_searchspace, sources, target_measurements = self._split_measurements()

        source_gps = []
        for _, source_measurements in sources:
            source_gp = evolve(self.base_surrogate)
            source_gp.fit(reduced_searchspace, self._objective, source_measurements)
            source_gps.append(source_gp)
        self._source_gps = tuple(source_gps)

        context = _ModelContext(self._searchspace, self._objective, self._measurements)
        self._numerical_indices = context.numerical_indices

        self._fit_target(reduced_searchspace, self._objective, target_measurements)

    def _split_measurements(
        self,
    ) -> tuple[SearchSpace, list[tuple[Any, pd.DataFrame]], pd.DataFrame]:
        """Validate the task configuration and split measurements by task.

        Returns:
            The reduced (task-free) search space, an ordered list of
            ``(task_value, measurements)`` pairs for the source tasks that have data
            (in the order they appear in the task parameter's values), and the
            target-task measurements (which may be empty).

        Raises:
            IncompatibleSearchSpaceError: If the search space has no task parameter,
                does not describe exactly one target task and at least one source task,
                exceeds the surrogate's supported number of source tasks, or if none of
                the source tasks has any measurements.
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
        if len(active_values) != 1:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires exactly one active (target) "
                f"task value, but the task parameter describes {len(active_values)}."
            )
        if len(source_values) < 1:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires at least one source "
                f"(non-active) task value, but the task parameter describes none."
            )
        if self._max_sources is not None and len(source_values) > self._max_sources:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' supports at most {self._max_sources} "
                f"source task(s), but the task parameter describes "
                f"{len(source_values)}."
            )
        (target_value,) = active_values

        task_name = task_param.name
        sources = [
            (value, subset)
            for value in task_param.values
            if value in source_values
            if not (subset := measurements[measurements[task_name] == value]).empty
        ]
        if not sources:
            raise IncompatibleSearchSpaceError(
                f"'{self.__class__.__name__}' requires measurements for at least one "
                f"source task, but none were provided."
            )
        target_measurements = measurements[measurements[task_name] == target_value]

        reduced_searchspace = searchspace._drop_parameters({task_name})
        return reduced_searchspace, sources, target_measurements

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

    def _source_only_posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Return the (single) source GP's posterior on task-stripped candidates.

        This is the cold-start fallback used by single-source subclasses when the
        target task has no measurements yet: with no target data, the source model is
        the best available estimate of the target.

        Args:
            candidates_comp_scaled: Candidate points in the computational representation
                of the full search space (including the task column).

        Returns:
            The source Gaussian process's posterior at the given candidates.
        """
        reduced_candidates = self._strip_task(candidates_comp_scaled)
        return self._source_gp._posterior(reduced_candidates)

    @abstractmethod
    def _fit_target(
        self,
        reduced_searchspace: SearchSpace,
        objective: Objective,
        target_measurements: pd.DataFrame,
    ) -> None:
        """Build and fit the target model from the fitted source GP(s).

        The fitted source GPs are available via :attr:`_source_gps` (or
        :attr:`_source_gp` for single-source subclasses). The target measurements may
        be empty, in which case the subclass is expected to arrange a source-informed
        cold-start prediction.

        Args:
            reduced_searchspace: The task-free search space for the inner GPs.
            objective: The objective (a single modeled quantity after replication).
            target_measurements: The measurements belonging to the target task (may be
                empty).
        """


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
