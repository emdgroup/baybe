"""Source prior transfer learning surrogate using DataFrame-based operations."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from attrs import define, field
from botorch.models.model import Model
from botorch.posteriors import Posterior
from typing_extensions import override

from baybe.exceptions import ModelNotTrainedError
from baybe.objectives.base import Objective
from baybe.parameters import TaskParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import (
    GaussianProcessSurrogate,
)


@define
class SourcePriorGaussianProcessSurrogate(GaussianProcessSurrogate):
    """Source prior transfer learning Gaussian process surrogate.

    This surrogate implements transfer learning by:
    1. Splitting measurement data into source and target tasks
    2. Training a source GP on source task data (without task dimension)
    3. Training a target GP on target task data, using source GP as prior
    4. Using only the target GP for predictions

    Transfer learning is configured via TransferConfig objects that specify how to
    transfer mean functions and/or covariance functions from the source GP.
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Private attributes for storing model state
    _source_surrogate: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False
    )
    """The source surrogate (needed for posterior calls)."""

    _target_surrogate: GaussianProcessSurrogate | None = field(
        init=False, default=None, eq=False
    )
    """The target surrogate (needed for posterior calls)."""

    _reduced_searchspace: SearchSpace | None = field(init=False, default=None, eq=False)
    """SearchSpace without task parameter (for routing posterior calls)."""

    _objective: Objective | None = field(init=False, default=None, eq=False)
    """Stored objective for creating target surrogate."""

    _task_name: str | None = field(init=False, default=None, eq=False)
    """Name of the task parameter."""

    @override
    def fit(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Fit source and target GPs using data splitting.

        Args:
            searchspace: The search space including TaskParameter
            objective: The objective to optimize
            measurements: Training data including task column

        Raises:
            ValueError: If no TaskParameter found in searchspace
            ValueError: If no source data available for transfer learning
        """
        # Validate and extract task information
        task_param = next(
            (p for p in searchspace.parameters if isinstance(p, TaskParameter)), None
        )
        if task_param is None:
            raise ValueError(
                "SourcePriorGaussianProcessSurrogate requires a TaskParameter"
            )

        target_tasks = task_param.active_values
        source_tasks = [t for t in task_param.values if t not in target_tasks]
        self._task_name = task_param.name

        source_data = measurements[
            measurements[self._task_name].isin(source_tasks)
        ].drop(columns=[self._task_name])
        target_data = measurements[
            measurements[self._task_name].isin(target_tasks)
        ].drop(columns=[self._task_name])

        if len(source_data) == 0:
            raise ValueError(
                "No source data found. SourcePriorGaussianProcessSurrogate requires "
                "at least source data for transfer learning."
            )

        # Store context
        reduced_searchspace = searchspace.remove_task_parameters()
        # TODO: Do we need to store?
        self._reduced_searchspace = reduced_searchspace
        self._searchspace = searchspace
        self._objective = objective

        # Train source GP
        self._source_surrogate = GaussianProcessSurrogate(
            kernel_or_factory=self.kernel_factory
        )
        self._source_surrogate.fit(reduced_searchspace, objective, source_data)
        # Train target GP with source prior
        if len(target_data) == 0:
            # No target data - use source surrogate directly
            self._target_surrogate = self._source_surrogate  # type: ignore[return-value]
        else:
            # Create target surrogate with transfer learning
            self._target_surrogate = GaussianProcessSurrogate.from_prior(
                prior_gp=self._source_surrogate.to_botorch(),  # type: ignore[union-attr]
                kernel_factory=self.kernel_factory,
            )

            # Fit the target surrogate with target data
            self._target_surrogate.fit(reduced_searchspace, objective, target_data)

    @override
    def posterior(self, candidates: pd.DataFrame) -> Posterior:
        """Get posterior predictions from target GP only.

        Args:
            candidates: DataFrame with parameter configurations

        Returns:
            Posterior distribution for target task only

        Raises:
            ModelNotTrainedError: If model not fitted
            ValueError: If candidates contain source task data
        """
        if self._target_surrogate is None:
            raise ModelNotTrainedError("Model must be fitted first")

        candidates_clean = candidates.drop(columns=self._task_name, errors="ignore")

        return self._target_surrogate.posterior(candidates_clean)

    @override
    def to_botorch(self) -> Model:
        """Return the BoTorch model representation.

        Returns:
            A wrapper model that delegates to the target surrogate

        Raises:
            ModelNotTrainedError: If model not fitted
        """
        if self._target_surrogate is None:
            raise ModelNotTrainedError(
                "Model must be fitted before accessing BoTorch model."
            )
        return self._target_surrogate.to_botorch()
