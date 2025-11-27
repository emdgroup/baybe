"""Abstract base class for source prior surrogates."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from attrs import define, field
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

from baybe.exceptions import ModelNotTrainedError
from baybe.parameters import TaskParameter
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate


class SourcePriorWrapperModel(Model):
    """BoTorch Model wrapper for SourcePriorGaussianProcessSurrogate._to_botorch."""

    def __init__(
        self, source_prior_surrogate: SourcePriorGaussianProcessSurrogate
    ) -> None:
        """Initialize the wrapper model.

        Args:
            source_prior_surrogate: The SourcePriorGaussianProcessSurrogate to wrap.
        """
        super().__init__()
        self.surrogate: SourcePriorGaussianProcessSurrogate = source_prior_surrogate

    @property
    def num_outputs(self) -> int:
        """Number of outputs of the model."""
        return 1

    def posterior(self, X: Tensor, **kwargs: Any) -> Posterior:
        """Compute posterior distribution.

        Args:
            X: Input tensor with shape (..., n_points, n_features + 1)
               where the last feature is the task index.
            **kwargs: Additional keyword arguments.

        Returns:
            Posterior distribution over the input points.
        """
        return self.surrogate._posterior(X)

    @property
    def train_inputs(self) -> tuple[Tensor, ...]:
        """Return training inputs from the target GP.

        Returns:
            Tuple of training input tensors.

        Raises:
            RuntimeError: If model not fitted.
        """
        if self.surrogate._target_gp is None:
            raise RuntimeError("Model not fitted")
        return self.surrogate._target_gp.train_inputs

    @property
    def train_targets(self) -> Tensor:
        """Return training targets from the target GP.

        Returns:
            Training target tensor.

        Raises:
            RuntimeError: If model not fitted.
        """
        if self.surrogate._target_gp is None:
            raise RuntimeError("Model not fitted")
        return self.surrogate._target_gp.train_targets


@define
class SourcePriorGaussianProcessSurrogate(GaussianProcessSurrogate):
    """Source prior transfer learning Gaussian process surrogate.

    This surrogate implements transfer learning by:
    1. Training a source GP on source task data (without task dimension)
    2. Using the source GP as a mean prior for the target GP
    3. Training the target GP on all data (source + target) with source-informed priors
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Input dim of the problem
    input_dim: int | None = field(default=None)
    """Dimensionality of the input space (excluding task feature)."""

    use_prior_mean: bool = field(default=True)
    """Whether to use the source GP as prior mean for the target GP."""

    use_prior_kernel: bool = field(default=False)
    """Whether to use the source GP's kernel as prior kernel for the target GP."""

    # Private attributes for storing model state
    _model: Model | None = field(init=False, default=None, eq=False)
    """The actual SourcePriorModel instance."""

    _target_task_id: int | None = field(init=False, default=None, eq=False)
    """Numeric ID of the target task."""

    _task_column_idx: int | None = field(init=False, default=None, eq=False)
    """Column index of task feature in BayBE's computational representation."""

    _source_gp: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """Fitted source Gaussian Process model."""

    _target_gp: SingleTaskGP | None = field(init=False, default=None, eq=False)
    """Fitted target Gaussian Process model with source prior."""

    def _identify_target_task(self) -> tuple[int, float]:
        """Identify the TaskParameter and return its column index and target value.

        This function identifies the TaskParameter within the search space, retrieves
        its active value, and returns both the column index of the TaskParameter and
        the computational representation value for the active value. This is useful
        for filtering tensor rows that correspond to the target task.

        Returns:
            A tuple containing:
            - task_idx (int): The column index of the TaskParameter in the computational
                            representation of the search space
            - target_value (float): The computational representation value for the
                        active value of the TaskParameter (used for filtering)
        """
        searchspace = self._searchspace  # type: ignore[assignment]
        # Find the TaskParameter in the search space
        task_params = [
            p for p in searchspace.parameters if isinstance(p, TaskParameter)
        ]

        task_param = task_params[0]

        # Get the active value
        active_value = task_param.active_values[0]

        # Get the index of the TaskParameter in the computational representation
        task_idx = searchspace.task_idx  # type: ignore[assignment]

        # Get the computational representation value for the active value
        # TaskParameter uses INT encoding, so comp_df has a single column with
        # integer values
        comp_df = task_param.comp_df

        # Extract the single computational representation value
        target_value = float(comp_df.loc[active_value].iloc[0])

        return task_idx, target_value

    def _validate_transfer_learning_context(self) -> None:
        """Validate that we have a proper transfer learning setup.

        Raises:
            ValueError: If no task parameter found in search space.
            ValueError: If input dimensions don't match expected format.
        """
        if self._searchspace.task_idx is None:
            raise ValueError(
                "No task parameter found in search space. "
                "SourcePriorGaussianProcessSurrogate requires a TaskParameter "
                "for transfer learning."
            )
        # Set input_dim if not provided at initialization
        if self.input_dim is None:
            self.input_dim = len(self._searchspace.comp_rep_columns) - 1

        # Validate that we have the expected number of feature dimensions
        expected_total_dims = self.input_dim + 1  # features + task
        actual_total_dims = len(self._searchspace.comp_rep_columns)

        if actual_total_dims != expected_total_dims:
            raise ValueError(
                f"Expected {expected_total_dims} total dimensions "
                f"({self.input_dim} features + 1 task), "
                f"but got {actual_total_dims} from search space."
            )

    def _extract_task_data(
        self,
        X: Tensor,
        Y: Tensor | None = None,
        task_feature: int = -1,
        target_task: int = 0,
    ) -> tuple[list[tuple[Tensor, Tensor | None]], tuple[Tensor, Tensor | None]]:
        """Extract source and target data from multi-task format.

        Args:
            X: Input data including task indices, shape (n_total, input_dim + 1).
            Y: Output data, shape (n_total, 1).
            task_feature: Index of the task feature dimension.
            target_task: Task ID for the target task.

        Returns:
            Tuple of (source_data_list, target_data_tuple).
        """
        # Extract task indices
        task_indices = X[:, task_feature].long()

        # Extract input features (without task index)
        if task_feature == -1:
            X_features = X[:, :-1]
        else:
            X_features = torch.cat(
                [X[:, :task_feature], X[:, task_feature + 1 :]], dim=1
            )

        # Get unique task IDs and sort them
        unique_tasks = torch.unique(task_indices)
        source_tasks = unique_tasks[unique_tasks != target_task].sort().values

        # Extract source data
        source_data = []
        for task_id in source_tasks:
            mask = task_indices == task_id
            X_task = X_features[mask]
            Y_task = Y[mask] if Y is not None else None
            source_data.append((X_task, Y_task))

        # Extract target data
        target_mask = task_indices == target_task
        X_target = X_features[target_mask]
        Y_target = Y[target_mask] if Y is not None else None

        return source_data, (X_target, Y_target)

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the transfer learning model.

        This method handles the common training workflow for all transfergpbo models:
        1. Validate transfer learning context
        2. Identify target task from TaskParameter
        3. Fit source GP on the source task data
        4. Fit target GP on the target task data using source GP posterior mean
           as prior mean.

        Args:
            train_x: Training inputs in BayBE's computational representation.
                    Shape: (n_points, n_features + 1) last column may be task indices.
            train_y: Training targets. Shape: (n_points, 1).

        Raises:
            ValueError: If received empty training data or if context is invalid.
        """
        # FIXME[typing]: It seems there is currently no better way to inform the type
        #   checker that the attribute is available at the time of the function call
        assert self._searchspace is not None

        # Check if we receive empty data
        if train_x.shape[0] == 0 or train_y.shape[0] == 0:
            raise ValueError(
                f"Received empty training data! "
                f"train_x.shape={train_x.shape}, train_y.shape={train_y.shape}"
            )

        # 1. Validate transfer learning context
        self._validate_transfer_learning_context()

        # 2. Identify target task from TaskParameter active_values
        self._task_column_idx, self._target_task_id = self._identify_target_task()

        source_data, (X_target, Y_target) = self._extract_task_data(
            train_x, train_y, self._task_column_idx, self._target_task_id
        )

        if len(source_data) == 0:
            raise ValueError(
                "No source data found. SourcePriorGaussianProcessSurrogate requires "
                "at least source data for transfer learning."
            )

        source_data = source_data[0]
        X_source, Y_source = source_data

        # Remove task parameter from searchspace before training the GPs
        reduced_searchspace = self._searchspace.remove_task_parameters()

        # 1. Create and fit source GP
        source_surrogate = GaussianProcessSurrogate(
            kernel_or_factory=self.kernel_factory
        )
        source_surrogate._searchspace = reduced_searchspace
        source_surrogate._fit(X_source, Y_source)
        self._source_gp = source_surrogate.to_botorch()

        # 2. Train GP based on available data and transfer mode
        if X_target.shape[0] == 0:
            # Use copy of source GP if no target data available
            print("No target data provided. Using copy of source GP as target GP.")
            from copy import deepcopy

            self._target_gp = deepcopy(self._source_gp)
        else:
            # Create target GP from prior based on transfer mode
            if self.use_prior_mean and not self.use_prior_kernel:
                target_surrogate = GaussianProcessSurrogate.from_prior_gp(
                    prior_gp=self._source_gp,
                    transfer_mode="mean",
                    kernel_factory=self.kernel_factory,
                )
            elif self.use_prior_kernel and not self.use_prior_mean:
                target_surrogate = GaussianProcessSurrogate.from_prior_gp(
                    prior_gp=self._source_gp, transfer_mode="kernel"
                )
            else:
                target_surrogate = GaussianProcessSurrogate(
                    kernel_or_factory=self.kernel_factory
                )

            # Fit target GP
            target_surrogate._searchspace = reduced_searchspace
            target_surrogate._fit(X_target, Y_target)
            self._target_gp = target_surrogate.to_botorch()

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute posterior predictions for target task only.

        Args:
            candidates_comp_scaled: Candidate points in computational representation.
                                   Must contain only target task data.

        Returns:
            Posterior distribution for the candidate points.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
            ValueError: If candidates contain source task data.
        """
        if self._target_gp is None:
            raise ModelNotTrainedError(
                "Model must be fitted before making predictions. Call fit() first."
            )

        # Check for source task data and reject it
        task_indices = candidates_comp_scaled[:, self._task_column_idx].long()
        if torch.any(task_indices != self._target_task_id).item():
            raise ValueError(
                "SourcePriorGaussianProcessSurrogate only supports predictions on "
                "target task data. Found source task data in candidates."
            )

        # Extract target task features by removing task column
        if self._task_column_idx == -1:
            target_features = candidates_comp_scaled[:, :-1]
        else:
            target_features = torch.cat(
                [
                    candidates_comp_scaled[:, : self._task_column_idx],
                    candidates_comp_scaled[:, self._task_column_idx + 1 :],
                ],
                dim=1,
            )

        return self._target_gp.posterior(target_features)

    @override
    def to_botorch(self) -> Model:
        """Return the BoTorch model representation.

        Returns:
            A wrapper model that can handle both source and target predictions.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._target_gp is None:
            raise ModelNotTrainedError(
                "Model must be fitted before accessing BoTorch model."
            )
        return SourcePriorWrapperModel(self)
